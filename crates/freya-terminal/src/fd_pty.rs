//! Adapter that wraps a raw file descriptor as a [`portable_pty::MasterPty`].
//!
//! This allows freya-terminal to drive a PTY that was opened elsewhere (e.g. by
//! a daemon) instead of always spawning its own via `native_pty_system()`.

use std::{
    cell::RefCell,
    fs::File,
    io::{Read, Write},
    mem,
    os::unix::io::{FromRawFd, RawFd},
    path::PathBuf,
};

use anyhow::bail;
use portable_pty::{MasterPty, PtySize};

/// Duplicate `fd` with close-on-exec set, then ensure the shared open-file
/// description is blocking so idle PTYs do not surface `WouldBlock` to the
/// reader loop (#3 residual P1/P2).
fn dup_cloexec_blocking(fd: RawFd) -> Result<RawFd, anyhow::Error> {
    // SAFETY: caller must pass a valid open fd; F_DUPFD_CLOEXEC returns a new
    // independent descriptor or -1.
    let duped = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if duped < 0 {
        bail!(
            "fcntl(F_DUPFD_CLOEXEC) failed: {:?}",
            std::io::Error::last_os_error()
        );
    }
    // O_NONBLOCK lives on the open-file description shared by all dups of the
    // master; clear it so portable_pty/reader loops that treat WouldBlock as
    // fatal do not tear down the terminal on idle.
    // SAFETY: `duped` is a valid open descriptor we just created.
    let flags = unsafe { libc::fcntl(duped, libc::F_GETFL) };
    if flags < 0 {
        // SAFETY: close the failed-prep descriptor to avoid leaking it.
        unsafe { libc::close(duped) };
        bail!(
            "fcntl(F_GETFL) failed: {:?}",
            std::io::Error::last_os_error()
        );
    }
    if flags & libc::O_NONBLOCK != 0 {
        // SAFETY: clear O_NONBLOCK on a live open descriptor we own.
        if unsafe { libc::fcntl(duped, libc::F_SETFL, flags & !libc::O_NONBLOCK) } != 0 {
            unsafe { libc::close(duped) };
            bail!(
                "fcntl(F_SETFL clear O_NONBLOCK) failed: {:?}",
                std::io::Error::last_os_error()
            );
        }
    }
    Ok(duped)
}

/// A [`MasterPty`] implementation backed by an owned raw file descriptor.
///
/// The fd is **owned** by this struct: it will be closed on [`Drop`].
pub struct RawFdMasterPty {
    fd: RawFd,
    took_writer: RefCell<bool>,
}

impl RawFdMasterPty {
    /// Wrap an existing PTY master file descriptor.
    ///
    /// # Safety
    /// The caller must ensure `fd` is a valid, open PTY master file descriptor
    /// and that ownership is being transferred to this struct (it will be closed
    /// on drop).
    pub unsafe fn from_fd(fd: RawFd) -> Self {
        Self {
            fd,
            took_writer: RefCell::new(false),
        }
    }
}

impl Drop for RawFdMasterPty {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

impl MasterPty for RawFdMasterPty {
    fn resize(&self, size: PtySize) -> Result<(), anyhow::Error> {
        let ws = libc::winsize {
            ws_row: size.rows,
            ws_col: size.cols,
            ws_xpixel: size.pixel_width,
            ws_ypixel: size.pixel_height,
        };
        if unsafe { libc::ioctl(self.fd, libc::TIOCSWINSZ as _, &ws as *const _) } != 0 {
            bail!(
                "ioctl(TIOCSWINSZ) failed: {:?}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    fn get_size(&self) -> Result<PtySize, anyhow::Error> {
        let mut ws: libc::winsize = unsafe { mem::zeroed() };
        if unsafe { libc::ioctl(self.fd, libc::TIOCGWINSZ as _, &mut ws as *mut _) } != 0 {
            bail!(
                "ioctl(TIOCGWINSZ) failed: {:?}",
                std::io::Error::last_os_error()
            );
        }
        Ok(PtySize {
            rows: ws.ws_row,
            cols: ws.ws_col,
            pixel_width: ws.ws_xpixel,
            pixel_height: ws.ws_ypixel,
        })
    }

    fn try_clone_reader(&self) -> Result<Box<dyn Read + Send>, anyhow::Error> {
        let duped = dup_cloexec_blocking(self.fd)?;
        Ok(Box::new(unsafe { File::from_raw_fd(duped) }))
    }

    fn take_writer(&self) -> Result<Box<dyn Write + Send>, anyhow::Error> {
        if *self.took_writer.borrow() {
            bail!("cannot take writer more than once");
        }
        // Flip the flag only after a successful dup so a failed dup does not
        // permanently poison the master (#3 residual).
        let duped = dup_cloexec_blocking(self.fd)?;
        *self.took_writer.borrow_mut() = true;
        Ok(Box::new(unsafe { File::from_raw_fd(duped) }))
    }

    fn process_group_leader(&self) -> Option<libc::pid_t> {
        match unsafe { libc::tcgetpgrp(self.fd) } {
            pid if pid > 0 => Some(pid),
            _ => None,
        }
    }

    fn as_raw_fd(&self) -> Option<RawFd> {
        Some(self.fd)
    }

    fn tty_name(&self) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use portable_pty::{PtySize, native_pty_system};

    fn open_adapter() -> (portable_pty::PtyPair, RawFdMasterPty) {
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("openpty");
        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0, "dup failed");
        let adapter = unsafe { RawFdMasterPty::from_fd(duped) };
        (pair, adapter)
    }

    #[test]
    fn resize_on_real_pty() {
        let (_pair, adapter) = open_adapter();
        adapter
            .resize(PtySize {
                rows: 40,
                cols: 120,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("resize");
        let size = adapter.get_size().expect("get_size");
        assert_eq!(size.rows, 40);
        assert_eq!(size.cols, 120);
    }

    #[test]
    fn try_clone_reader_and_take_writer() {
        let (_pair, adapter) = open_adapter();
        let _reader = adapter.try_clone_reader().expect("try_clone_reader");
        let _writer = adapter.take_writer().expect("take_writer");
        assert!(adapter.take_writer().is_err());
    }

    #[test]
    fn clones_set_cloexec_and_clear_nonblock() {
        let (_pair, adapter) = open_adapter();
        // Mark the master ofd non-blocking; clones must clear it.
        let flags = unsafe { libc::fcntl(adapter.as_raw_fd().expect("fd"), libc::F_GETFL) };
        assert!(flags >= 0);
        assert_eq!(
            unsafe { libc::fcntl(adapter.as_raw_fd().expect("fd"), libc::F_SETFL, flags | libc::O_NONBLOCK) },
            0
        );

        let reader = adapter.try_clone_reader().expect("reader");
        // Drop the boxed reader after inspecting its fd via as_raw_fd on File.
        // We re-dup via try_clone path and check the master ofd is now blocking.
        drop(reader);
        let flags_after = unsafe { libc::fcntl(adapter.as_raw_fd().expect("fd"), libc::F_GETFL) };
        assert!(flags_after >= 0);
        assert_eq!(
            flags_after & libc::O_NONBLOCK,
            0,
            "clone path must clear O_NONBLOCK on the shared open-file description"
        );

        // Fresh clone: FD_CLOEXEC must be set on the duplicate.
        let duped = dup_cloexec_blocking(adapter.as_raw_fd().expect("fd")).expect("dup_cloexec");
        let fd_flags = unsafe { libc::fcntl(duped, libc::F_GETFD) };
        assert!(fd_flags >= 0);
        assert_ne!(
            fd_flags & libc::FD_CLOEXEC,
            0,
            "duplicate must have FD_CLOEXEC"
        );
        unsafe { libc::close(duped) };
    }
}
