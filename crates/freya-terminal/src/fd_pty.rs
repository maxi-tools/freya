//! Adapter that wraps a raw file descriptor as a [`portable_pty::MasterPty`].
//!
//! This allows freya-terminal to drive a PTY that was opened elsewhere (e.g. by
//! a daemon) instead of always spawning its own via `native_pty_system()`.

use std::{
    cell::RefCell,
    fs::File,
    io::{Read, Write},
    mem,
    os::unix::io::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    path::PathBuf,
};

use anyhow::bail;
use portable_pty::{MasterPty, PtySize};

/// A [`MasterPty`] implementation backed by an owned file descriptor.
///
/// Ownership is enforced by the type system rather than by a comment:
/// constructing this struct consumes an [`OwnedFd`], so the compiler prevents
/// the caller from using or closing the descriptor afterwards. It will be
/// closed when this struct (and thus the `OwnedFd`) is dropped.
pub struct RawFdMasterPty {
    fd: OwnedFd,
    took_writer: RefCell<bool>,
}

impl RawFdMasterPty {
    /// Wrap an existing PTY master file descriptor, taking ownership of it.
    ///
    /// Passing an [`OwnedFd`] transfers ownership by value — the compiler
    /// prevents the caller from using or closing the descriptor afterwards —
    /// so this constructor needs no `unsafe` and no separate ownership
    /// documentation contract; the type itself mechanically enforces it.
    pub fn from_owned_fd(fd: OwnedFd) -> Self {
        Self {
            fd,
            took_writer: RefCell::new(false),
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
        // SAFETY: `self.fd` is a valid, open descriptor owned by `self` for
        // the duration of this call; `ws` is a fully-initialized, live local
        // and `TIOCSWINSZ` only reads through the pointer for the call.
        if unsafe { libc::ioctl(self.fd.as_raw_fd(), libc::TIOCSWINSZ as _, &ws as *const _) } != 0
        {
            bail!(
                "ioctl(TIOCSWINSZ) failed: {:?}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    fn get_size(&self) -> Result<PtySize, anyhow::Error> {
        // SAFETY: `libc::winsize` is a plain-old-data struct of integer
        // fields, so an all-zero bit pattern is a valid value; it is fully
        // overwritten by the ioctl below before being read.
        let mut ws: libc::winsize = unsafe { mem::zeroed() };
        // SAFETY: `self.fd` is a valid, open descriptor owned by `self` for
        // the duration of this call, and `ws` is a valid, live mutable
        // local; `TIOCGWINSZ` only writes through the pointer for the call.
        if unsafe {
            libc::ioctl(
                self.fd.as_raw_fd(),
                libc::TIOCGWINSZ as _,
                &mut ws as *mut _,
            )
        } != 0
        {
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
        // SAFETY: `self.fd` is a valid, open descriptor owned by `self` for
        // the lifetime of this call; `dup()` only reads it and returns a new,
        // independent descriptor on success (or -1 on failure, checked below).
        let duped = unsafe { libc::dup(self.fd.as_raw_fd()) };
        if duped < 0 {
            bail!("dup() failed: {:?}", std::io::Error::last_os_error());
        }
        // SAFETY: `duped` was just returned by the successful `dup()` above,
        // so it is a valid, open, and otherwise-unowned descriptor; wrapping
        // it in a `File` gives it exactly one owner, which closes it on drop.
        Ok(Box::new(unsafe { File::from_raw_fd(duped) }))
    }

    fn take_writer(&self) -> Result<Box<dyn Write + Send>, anyhow::Error> {
        if *self.took_writer.borrow() {
            bail!("cannot take writer more than once");
        }
        // Ownership flag flips only after a successful dup so a failed dup does
        // not permanently poison the master (disposition fix for PR #3).
        // SAFETY: `self.fd` is a valid, open descriptor owned by `self` for
        // the lifetime of this call; `dup()` only reads it and returns a new,
        // independent descriptor on success (or -1 on failure, checked below).
        let duped = unsafe { libc::dup(self.fd.as_raw_fd()) };
        if duped < 0 {
            bail!("dup() failed: {:?}", std::io::Error::last_os_error());
        }
        *self.took_writer.borrow_mut() = true;
        // SAFETY: `duped` was just returned by the successful `dup()` above,
        // so it is a valid, open, and otherwise-unowned descriptor; wrapping
        // it in a `File` gives it exactly one owner, which closes it on drop.
        Ok(Box::new(unsafe { File::from_raw_fd(duped) }))
    }

    fn process_group_leader(&self) -> Option<libc::pid_t> {
        // SAFETY: `self.fd` is a valid, open descriptor owned by `self` for
        // the duration of this call; `tcgetpgrp()` only reads it and returns
        // a pid or -1 on error.
        match unsafe { libc::tcgetpgrp(self.fd.as_raw_fd()) } {
            pid if pid > 0 => Some(pid),
            _ => None,
        }
    }

    fn as_raw_fd(&self) -> Option<RawFd> {
        Some(self.fd.as_raw_fd())
    }

    fn tty_name(&self) -> Option<PathBuf> {
        None
    }
}

#[cfg(test)]
mod tests {
    use portable_pty::{CommandBuilder, PtySize, native_pty_system};

    use super::*;

    #[test]
    fn resize_on_real_pty() {
        let _guard = crate::test_support::PTY_FD_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        // SAFETY: `raw_fd` is a valid, open pty master descriptor owned by
        // `pair.master` for the scope of this test; `dup()` only reads it and
        // returns a new, independent descriptor on success.
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0, "dup failed");

        // SAFETY: `duped` was just returned by the successful `dup()` above,
        // so it is a valid, open, and otherwise-unowned descriptor; wrapping
        // it in an `OwnedFd` gives it exactly one owner.
        let adapter = RawFdMasterPty::from_owned_fd(unsafe { OwnedFd::from_raw_fd(duped) });

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
        let _guard = crate::test_support::PTY_FD_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize::default()).expect("openpty");

        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        // SAFETY: `raw_fd` is a valid, open pty master descriptor owned by
        // `pair.master` for the scope of this test; `dup()` only reads it and
        // returns a new, independent descriptor on success.
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0);

        // SAFETY: `duped` was just returned by the successful `dup()` above,
        // so it is a valid, open, and otherwise-unowned descriptor; wrapping
        // it in an `OwnedFd` gives it exactly one owner.
        let adapter = RawFdMasterPty::from_owned_fd(unsafe { OwnedFd::from_raw_fd(duped) });

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
        let _guard = crate::test_support::PTY_FD_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize::default()).expect("openpty");
        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        // SAFETY: `raw_fd` is a valid, open pty master descriptor owned by
        // `pair.master` for the scope of this test; `dup()` only reads it and
        // returns a new, independent descriptor on success.
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0);
        // SAFETY: `duped` is a valid, open descriptor from the `dup()` above
        // that is not yet owned by anything else (not wrapped into an
        // `OwnedFd` yet), so closing it directly here is sound; this
        // deliberately makes the fd number stale before it is wrapped below,
        // which is exactly the failure this test wants to exercise.
        // Close the duped fd before wrapping so dup() inside take_writer fails.
        unsafe { libc::close(duped) };
        // SAFETY: `duped` is a valid (if now-stale) fd number that is not
        // owned by anything else; wrapping it in an `OwnedFd` gives it
        // exactly one owner. The adapter is `mem::forget`'d below instead of
        // dropped normally, so this deliberately-invalid fd is never
        // double-closed.
        let adapter = RawFdMasterPty::from_owned_fd(unsafe { OwnedFd::from_raw_fd(duped) });
        assert!(adapter.take_writer().is_err());
        // Flag must remain false so a second attempt is still a "dup failed"
        // rather than "cannot take writer more than once". Matched via `match`
        // rather than `unwrap_err()` since the writer's success type (`Box<dyn
        // Write + Send>`) has no `Debug` impl, so `unwrap_err()` would not
        // compile (it requires `T: Debug` to format the `Ok` case on panic).
        match adapter.take_writer() {
            Ok(_) => panic!("expected dup() to keep failing on an already-closed fd"),
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains("dup()"),
                    "expected dup failure after failed first take, got: {err}"
                );
            }
        }
        // Avoid double-close of already-closed fd in Drop: leak by forgetting
        // would still close; Drop closes self.fd which is already closed — that
        // is safe on Unix (close on invalid may EBADF). We accept that.
        std::mem::forget(adapter);
    }

    #[test]
    fn lifecycle_spawn_resize_kill_reap() {
        // Exercises the public RawFdMasterPty + real spawned child lifecycle
        // end-to-end: spawn, resize the pty, kill the child, and reap it so no
        // zombie/orphan process remains behind.
        let _guard = crate::test_support::PTY_FD_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows: 24,
                cols: 80,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("openpty");

        let mut cmd = CommandBuilder::new("sleep");
        cmd.arg("5");
        let mut child = pair.slave.spawn_command(cmd).expect("spawn_command");
        let pid = child.process_id().expect("process_id") as libc::pid_t;

        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        // SAFETY: `raw_fd` is a valid, open pty master descriptor owned by
        // `pair.master` for the scope of this test; `dup()` only reads it and
        // returns a new, independent descriptor on success.
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0, "dup failed");
        // SAFETY: `duped` was just returned by the successful `dup()` above,
        // so it is a valid, open, and otherwise-unowned descriptor; wrapping
        // it in an `OwnedFd` gives it exactly one owner.
        let adapter = RawFdMasterPty::from_owned_fd(unsafe { OwnedFd::from_raw_fd(duped) });

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

        child.kill().expect("kill");
        child.wait().expect("wait (reap)");

        // Fully reaped: a further wait on this pid must report ECHILD (no
        // unreaped child left behind), not the process still being alive.
        let mut status = 0;
        // SAFETY: `pid` is a valid pid obtained above and `status` is a
        // valid, live mutable local; `waitpid()` only reads `pid` and writes
        // through `status` for the duration of this call.
        let wait_result = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(wait_result, -1, "child should already be reaped");
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD),
            "expected ECHILD: no unreaped child should remain for this pid"
        );
    }

    #[test]
    fn process_group_leader_returns_child_pid() {
        // A freshly spawned child on the pty slave becomes both the session
        // leader and the tty's foreground process group leader, so
        // process_group_leader() must report its pid.
        let _guard = crate::test_support::PTY_FD_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize::default()).expect("openpty");

        let mut cmd = CommandBuilder::new("sleep");
        cmd.arg("5");
        let mut child = pair.slave.spawn_command(cmd).expect("spawn_command");
        let pid = child.process_id().expect("process_id") as libc::pid_t;

        let raw_fd = pair.master.as_raw_fd().expect("as_raw_fd");
        // SAFETY: `raw_fd` is a valid, open pty master descriptor owned by
        // `pair.master` for the scope of this test; `dup()` only reads it and
        // returns a new, independent descriptor on success.
        let duped = unsafe { libc::dup(raw_fd) };
        assert!(duped >= 0, "dup failed");
        // SAFETY: `duped` was just returned by the successful `dup()` above,
        // so it is a valid, open, and otherwise-unowned descriptor; wrapping
        // it in an `OwnedFd` gives it exactly one owner.
        let adapter = RawFdMasterPty::from_owned_fd(unsafe { OwnedFd::from_raw_fd(duped) });

        assert_eq!(adapter.process_group_leader(), Some(pid));

        let _ = child.kill();
        let _ = child.wait();
    }
}
