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

/// A [`MasterPty`] implementation backed by a raw file descriptor.
///
/// The fd is **owned** by this struct — it will be closed on [`Drop`].
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
        let duped = unsafe { libc::dup(self.fd) };
        if duped < 0 {
            bail!("dup() failed: {:?}", std::io::Error::last_os_error());
        }
        Ok(Box::new(unsafe { File::from_raw_fd(duped) }))
    }

    fn take_writer(&self) -> Result<Box<dyn Write + Send>, anyhow::Error> {
        if *self.took_writer.borrow() {
            bail!("cannot take writer more than once");
        }
        // Ownership flag flips only after a successful dup so a failed dup does
        // not permanently poison the master (disposition fix for PR #3).
        let duped = unsafe { libc::dup(self.fd) };
        if duped < 0 {
            bail!("dup() failed: {:?}", std::io::Error::last_os_error());
        }
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

    #[test]
    fn resize_on_real_pty() {
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
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize::default()).expect("openpty");

        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0);

        let adapter = unsafe { RawFdMasterPty::from_fd(duped) };

        let _reader = adapter.try_clone_reader().expect("try_clone_reader");
        let _writer = adapter.take_writer().expect("take_writer");

        assert!(adapter.take_writer().is_err());
    }

    #[test]
    fn take_writer_failed_dup_does_not_poison_flag() {
        // Closed fd: first take_writer must fail without setting took_writer,
        // so a subsequent take_writer on a valid re-wrap path still works.
        // Here we only assert the flag stays false after a failed dup on a
        // deliberately invalid fd path by using a closed fd master.
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize::default()).expect("openpty");
        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0);
        // Close the duped fd before wrapping so dup() inside take_writer fails.
        unsafe { libc::close(duped) };
        let adapter = unsafe { RawFdMasterPty::from_fd(duped) };
        assert!(adapter.take_writer().is_err());
        // Flag must remain false so a second attempt is still a "dup failed"
        // rather than "cannot take writer more than once".
        let err = adapter.take_writer().unwrap_err().to_string();
        assert!(
            err.contains("dup()"),
            "expected dup failure after failed first take, got: {err}"
        );
        // Avoid double-close of already-closed fd in Drop: leak by forgetting
        // would still close; Drop closes self.fd which is already closed — that
        // is safe on Unix (close on invalid may EBADF). We accept that.
        std::mem::forget(adapter);
    }
}
