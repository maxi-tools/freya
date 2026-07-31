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

/// Duplicate `fd` with close-on-exec set on the new descriptor only.
///
/// Does **not** change open-file description status flags (`O_NONBLOCK`, etc.):
/// those are shared with every other holder of the same OFD (for example a
/// daemon that still has a copy of the master). Callers that need a blocking
/// reader should pass a blocking descriptor or handle `WouldBlock` themselves.
pub(crate) fn dup_cloexec(fd: RawFd) -> Result<RawFd, anyhow::Error> {
    // SAFETY: caller must pass a valid open fd; F_DUPFD_CLOEXEC returns a new
    // independent descriptor or -1.
    let duped = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if duped < 0 {
        bail!(
            "fcntl(F_DUPFD_CLOEXEC) failed: {:?}",
            std::io::Error::last_os_error()
        );
    }
    Ok(duped)
}

/// Set `FD_CLOEXEC` on an existing descriptor (per-fd flag, not OFD status).
pub(crate) fn set_cloexec(fd: RawFd) -> Result<(), anyhow::Error> {
    // SAFETY: `fd` is a live open descriptor at the ownership-transfer boundary.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        bail!(
            "fcntl(F_GETFD) failed: {:?}",
            std::io::Error::last_os_error()
        );
    }
    if flags & libc::FD_CLOEXEC != 0 {
        return Ok(());
    }
    // SAFETY: F_SETFD only touches per-descriptor flags for `fd`.
    if unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } != 0 {
        bail!(
            "fcntl(F_SETFD FD_CLOEXEC) failed: {:?}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
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
    /// Sets `FD_CLOEXEC` on the adopted descriptor so a later `exec` in this
    /// process cannot leak the PTY master. Status flags on the open-file
    /// description (`O_NONBLOCK`, etc.) are left unchanged so concurrent
    /// holders of the same OFD are not disrupted.
    ///
    /// # Safety
    /// The caller must ensure `fd` is a valid, open PTY master file descriptor
    /// and that ownership is being transferred to this struct (it will be closed
    /// on drop).
    pub unsafe fn from_fd(fd: RawFd) -> Self {
        // Best-effort: if CLOEXEC cannot be set, still take ownership so the
        // caller does not lose the fd; the flag is set when possible.
        let _ = set_cloexec(fd);
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
        let duped = dup_cloexec(self.fd)?;
        Ok(Box::new(unsafe { File::from_raw_fd(duped) }))
    }

    fn take_writer(&self) -> Result<Box<dyn Write + Send>, anyhow::Error> {
        if *self.took_writer.borrow() {
            bail!("cannot take writer more than once");
        }
        // Flip the flag only after a successful dup so a failed dup does not
        // permanently poison the master.
        let duped = dup_cloexec(self.fd)?;
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
    use crate::fd_pty::{dup_cloexec, set_cloexec, RawFdMasterPty};
    use portable_pty::{native_pty_system, MasterPty, PtySize};
    use std::os::unix::io::AsRawFd;

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
        // Clear CLOEXEC so from_fd must re-apply it.
        let flags = unsafe { libc::fcntl(duped, libc::F_GETFD) };
        assert!(flags >= 0);
        assert_eq!(
            unsafe { libc::fcntl(duped, libc::F_SETFD, flags & !libc::FD_CLOEXEC) },
            0
        );
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
    fn from_fd_sets_cloexec_on_adopted_descriptor() {
        let (_pair, adapter) = open_adapter();
        let fd = adapter.as_raw_fd().expect("fd");
        let fd_flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert!(fd_flags >= 0);
        assert_ne!(
            fd_flags & libc::FD_CLOEXEC,
            0,
            "adopted master must have FD_CLOEXEC"
        );
    }

    #[test]
    fn clones_set_cloexec_without_mutating_shared_status_flags() {
        let (_pair, adapter) = open_adapter();
        // Mark the master OFD non-blocking; clones must NOT clear it (shared OFD).
        let flags = unsafe { libc::fcntl(adapter.as_raw_fd().expect("fd"), libc::F_GETFL) };
        assert!(flags >= 0);
        assert_eq!(
            unsafe {
                libc::fcntl(
                    adapter.as_raw_fd().expect("fd"),
                    libc::F_SETFL,
                    flags | libc::O_NONBLOCK,
                )
            },
            0
        );

        let _reader = adapter.try_clone_reader().expect("reader");
        let flags_after = unsafe { libc::fcntl(adapter.as_raw_fd().expect("fd"), libc::F_GETFL) };
        assert!(flags_after >= 0);
        assert_ne!(
            flags_after & libc::O_NONBLOCK,
            0,
            "clone path must not clear O_NONBLOCK on the shared open-file description"
        );

        // Fresh clone: FD_CLOEXEC must be set on the duplicate.
        let duped = dup_cloexec(adapter.as_raw_fd().expect("fd")).expect("dup_cloexec");
        let fd_flags = unsafe { libc::fcntl(duped, libc::F_GETFD) };
        assert!(fd_flags >= 0);
        assert_ne!(
            fd_flags & libc::FD_CLOEXEC,
            0,
            "duplicate must have FD_CLOEXEC"
        );
        unsafe { libc::close(duped) };
    }

    #[test]
    fn set_cloexec_is_idempotent() {
        let (_pair, adapter) = open_adapter();
        let fd = adapter.as_raw_fd().expect("fd");
        set_cloexec(fd).expect("first");
        set_cloexec(fd).expect("second");
    }
}
