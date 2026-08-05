use std::{cell::RefCell, io::Read, path::PathBuf, rc::Rc, time::Instant};

use freya_core::{
    notify::ArcNotify,
    prelude::{Platform, TaskHandle, UserEvent, spawn_forever},
};
use futures_lite::AsyncReadExt;
use keyboard_types::Modifiers;
use portable_pty::{CommandBuilder, MasterPty, PtySize, native_pty_system};
use termwiz::escape::{
    Action, CSI, OperatingSystemCommand,
    csi::{Cursor, Device},
    parser::Parser as TermwizParser,
};
use vt100::Parser;

use crate::{
    buffer::TerminalBuffer,
    handle::{TerminalCleaner, TerminalError, TerminalHandle, TerminalId},
};

/// Query the maximum scrollback available without disturbing the viewport.
/// Saves current scrollback, queries max, and restores.
pub(crate) fn query_max_scrollback(parser: &mut Parser) -> usize {
    let saved = parser.screen().scrollback();
    parser.screen_mut().set_scrollback(usize::MAX);
    let max = parser.screen().scrollback();
    parser.screen_mut().set_scrollback(saved);
    max
}

/// Extract visible cells from the parser at the current scrollback position.
pub(crate) fn extract_buffer(
    parser: &Parser,
    scroll_offset: usize,
    total_scrollback: usize,
) -> TerminalBuffer {
    let (rows, cols) = parser.screen().size();
    let rows_vec: Vec<Vec<vt100::Cell>> = (0..rows)
        .map(|r| {
            (0..cols)
                .filter_map(|c| parser.screen().cell(r, c).cloned())
                .collect()
        })
        .collect();
    let (cur_r, cur_c) = parser.screen().cursor_position();
    TerminalBuffer {
        rows: rows_vec,
        cursor_row: cur_r as usize,
        cursor_col: cur_c as usize,
        cols: cols as usize,
        rows_count: rows as usize,
        selection: None,
        scroll_offset,
        total_scrollback,
        cursor_visible: !parser.screen().hide_cursor(),
    }
}

/// Attach a freshly spawned child to its master, wiring up the shared
/// terminal lifecycle.
///
/// If setup fails (e.g. a failed `take_writer()`/`try_clone_reader()`
/// `dup()`), the already-spawned child is killed and reaped here so it isn't
/// left orphaned/zombied by the early return.
pub(crate) fn attach_spawned_child(
    id: TerminalId,
    master: Box<dyn MasterPty + Send>,
    mut child: Box<dyn portable_pty::Child + Send + Sync>,
    scrollback_size: usize,
) -> Result<TerminalHandle, TerminalError> {
    setup_terminal_from_master(id, master, scrollback_size).inspect_err(|_| {
        let _ = child.kill();
        let _ = child.wait();
    })
}

/// Spawn a PTY and return a TerminalHandle.
pub(crate) fn spawn_pty(
    id: TerminalId,
    command: CommandBuilder,
    scrollback_size: usize,
) -> Result<TerminalHandle, TerminalError> {
    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize::default())
        .map_err(|e| TerminalError::PtyError(e.to_string()))?;

    let child = pair
        .slave
        .spawn_command(command)
        .map_err(|e| TerminalError::PtyError(e.to_string()))?;

    attach_spawned_child(id, pair.master, child, scrollback_size)
}

/// Shared, ref-counted terminal state created before any I/O is wired up.
///
/// Bundled into one struct purely to keep [`setup_terminal_from_master`]'s
/// delegation to the helpers below manageable. Pure grouping of the same
/// fields that used to be separate local variables; no behavior change.
struct TerminalState {
    buffer: Rc<RefCell<TerminalBuffer>>,
    parser: Rc<RefCell<Parser>>,
    writer: Rc<RefCell<Option<Box<dyn std::io::Write + Send>>>>,
    closer_notifier: ArcNotify,
    output_notifier: ArcNotify,
    title_notifier: ArcNotify,
    cwd: Rc<RefCell<Option<PathBuf>>>,
    title: Rc<RefCell<Option<String>>>,
    clipboard_content: Rc<RefCell<Option<String>>>,
    clipboard_notifier: ArcNotify,
}

/// Allocate the buffer, parser, writer cell, and notifiers shared by the
/// reader and PTY tasks and by the resulting [`TerminalHandle`].
fn init_terminal_state(scrollback_size: usize, size: PtySize) -> TerminalState {
    // Zero is "unset" on many PTY masters, not a 1x1 map. Fall back to the
    // historical default instead of max(1) (#8 cubic P2).
    let rows = if size.rows == 0 { 24 } else { size.rows };
    let cols = if size.cols == 0 { 80 } else { size.cols };
    TerminalState {
        buffer: Rc::new(RefCell::new(TerminalBuffer::default())),
        parser: Rc::new(RefCell::new(Parser::new(rows, cols, scrollback_size))),
        writer: Rc::new(RefCell::new(None)),
        closer_notifier: ArcNotify::new(),
        output_notifier: ArcNotify::new(),
        title_notifier: ArcNotify::new(),
        cwd: Rc::new(RefCell::new(None)),
        title: Rc::new(RefCell::new(None)),
        clipboard_content: Rc::new(RefCell::new(None)),
        clipboard_notifier: ArcNotify::new(),
    }
}

/// Take the writer and reader halves out of `master`, storing the writer in
/// `writer` and wrapping `master` itself for shared, resizable access.
///
/// Returns the async-wrapped reader alongside the shared master handle.
fn wire_master_io(
    master: Box<dyn MasterPty + Send>,
    writer: &Rc<RefCell<Option<Box<dyn std::io::Write + Send>>>>,
) -> Result<
    (
        blocking::Unblock<Box<dyn Read + Send>>,
        Rc<RefCell<Box<dyn MasterPty + Send>>>,
    ),
    TerminalError,
> {
    let master_writer = master
        .take_writer()
        .map_err(|e| TerminalError::PtyError(e.to_string()))?;
    *writer.borrow_mut() = Some(master_writer);

    let reader = master
        .try_clone_reader()
        .map_err(|e| TerminalError::PtyError(e.to_string()))?;
    let reader = blocking::Unblock::new(reader);

    Ok((reader, Rc::new(RefCell::new(master))))
}

/// Spawn the task that rebuilds `state.buffer` from `state.parser` each time
/// the PTY task (see [`spawn_pty_task`]) signals new output on `update_rx`,
/// and marks the terminal closed (dropping the writer, notifying
/// `state.closer_notifier`) once the channel closes, i.e. the PTY task has
/// exited.
fn spawn_reader_task(
    mut update_rx: futures_channel::mpsc::UnboundedReceiver<()>,
    state: &TerminalState,
    platform: Platform,
) -> TaskHandle {
    let parser = state.parser.clone();
    let buffer = state.buffer.clone();
    let closer_notifier = state.closer_notifier.clone();
    let writer = state.writer.clone();
    spawn_forever(async move {
        use futures_lite::StreamExt;
        while let Some(()) = update_rx.next().await {
            let mut parser = parser.borrow_mut();
            let total_scrollback = query_max_scrollback(&mut parser);

            let mut buffer = buffer.borrow_mut();
            let old_total_scrollback = buffer.total_scrollback;
            let delta = total_scrollback.saturating_sub(old_total_scrollback);
            parser.screen_mut().set_scrollback(buffer.scroll_offset);
            let mut new_buffer = extract_buffer(&parser, buffer.scroll_offset, total_scrollback);
            parser.screen_mut().set_scrollback(0);

            new_buffer.selection = buffer.selection.take().map(|mut selection| {
                selection.start_scroll = selection.start_scroll.saturating_add(delta);
                selection.end_scroll = selection.end_scroll.saturating_add(delta);
                selection
            });
            *buffer = new_buffer;
            platform.send(UserEvent::RequestRedraw);
        }
        // Channel closed: PTY exited
        *writer.borrow_mut() = None;
        closer_notifier.notify();
        platform.send(UserEvent::RequestRedraw);
    })
}

/// Spawn the task that reads raw bytes from `reader`, feeds them to the
/// VT100 parser, detects terminal queries/OSC sequences via termwiz, writes
/// any required responses back through `state.writer`, and pings
/// `update_tx` / `state.output_notifier` after each chunk so the reader task
/// (see [`spawn_reader_task`]) can rebuild the buffer.
fn spawn_pty_task(
    mut reader: blocking::Unblock<Box<dyn Read + Send>>,
    update_tx: futures_channel::mpsc::UnboundedSender<()>,
    state: &TerminalState,
) -> TaskHandle {
    let writer = state.writer.clone();
    let parser = state.parser.clone();
    let output_notifier = state.output_notifier.clone();
    let cwd = state.cwd.clone();
    let title = state.title.clone();
    let title_notifier = state.title_notifier.clone();
    let clipboard_content = state.clipboard_content.clone();
    let clipboard_notifier = state.clipboard_notifier.clone();
    spawn_forever(async move {
        let mut tw_parser = TermwizParser::new();
        loop {
            let mut buf = [0u8; 4096];

            match reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    let data = &buf[..n];

                    parser.borrow_mut().process(data);

                    // Use termwiz to detect terminal queries and OSC sequences
                    let actions = tw_parser.parse_as_vec(data);
                    let mut responses: Vec<Vec<u8>> = Vec::new();

                    for action in actions {
                        match action {
                            Action::CSI(CSI::Device(dev)) => match *dev {
                                Device::RequestPrimaryDeviceAttributes => {
                                    responses.push(b"\x1b[?62;22c".to_vec());
                                }
                                Device::RequestSecondaryDeviceAttributes => {
                                    responses.push(b"\x1b[>0;0;0c".to_vec());
                                }
                                Device::StatusReport => {
                                    responses.push(b"\x1b[0n".to_vec());
                                }
                                _ => {}
                            },
                            Action::CSI(CSI::Cursor(Cursor::RequestActivePositionReport)) => {
                                let p = parser.borrow();
                                let (row, col) = p.screen().cursor_position();
                                let response = format!("\x1b[{};{}R", row + 1, col + 1);
                                responses.push(response.into_bytes());
                            }
                            Action::OperatingSystemCommand(osc) => match *osc {
                                OperatingSystemCommand::CurrentWorkingDirectory(url) => {
                                    // Strip file:// prefix if present
                                    let path = if let Some(stripped) = url.strip_prefix("file://") {
                                        // file:///path or file://hostname/path
                                        if let Some(rest) = stripped.strip_prefix('/') {
                                            PathBuf::from(format!("/{rest}"))
                                        } else if let Some((_host, path)) = stripped.split_once('/')
                                        {
                                            PathBuf::from(format!("/{path}"))
                                        } else {
                                            PathBuf::from(stripped)
                                        }
                                    } else {
                                        PathBuf::from(url)
                                    };
                                    *cwd.borrow_mut() = Some(path);
                                }
                                OperatingSystemCommand::SetWindowTitle(t)
                                | OperatingSystemCommand::SetIconNameAndWindowTitle(t) => {
                                    *title.borrow_mut() = Some(t);
                                    title_notifier.notify();
                                }
                                OperatingSystemCommand::SetSelection(_sel, text) => {
                                    *clipboard_content.borrow_mut() = Some(text);
                                    clipboard_notifier.notify();
                                }
                                _ => {}
                            },
                            _ => {}
                        }
                    }

                    if !responses.is_empty()
                        && let Some(writer) = &mut *writer.borrow_mut()
                    {
                        for response in responses {
                            let _ = writer.write_all(&response);
                        }
                        let _ = writer.flush();
                    }

                    let _ = update_tx.unbounded_send(());
                    output_notifier.notify();
                }
                Err(_) => break,
            }
        }
    })
}

/// Wire up a [`MasterPty`] (reader, writer, async tasks) into a [`TerminalHandle`].
///
/// Shared post-PTY-creation path used by both [`spawn_pty`] and
/// [`TerminalHandle::from_fd`] (daemon-provided fd; no child spawn/cleanup).
pub(crate) fn setup_terminal_from_master(
    id: TerminalId,
    master: Box<dyn MasterPty + Send>,
    scrollback_size: usize,
) -> Result<TerminalHandle, TerminalError> {
    let (update_tx, update_rx) = futures_channel::mpsc::unbounded::<()>();

    // Seed the VT parser from the master's real geometry before any reader
    // task starts (daemon-provided fds may not be 24x80).
    // Treat missing or zero-dimension geometry as unset → 24x80 default.
    let size = master
        .get_size()
        .ok()
        .filter(|s| s.rows > 0 && s.cols > 0)
        .unwrap_or(PtySize {
            rows: 24,
            cols: 80,
            pixel_width: 0,
            pixel_height: 0,
        });
    let state = init_terminal_state(scrollback_size, size);
    let (reader, master) = wire_master_io(master, &state.writer)?;

    let platform = Platform::get();
    let reader_task = spawn_reader_task(update_rx, &state, platform);
    let pty_task = spawn_pty_task(reader, update_tx, &state);

    let TerminalState {
        buffer,
        parser,
        writer,
        closer_notifier,
        output_notifier,
        title_notifier,
        cwd,
        title,
        clipboard_content,
        clipboard_notifier,
    } = state;

    Ok(TerminalHandle {
        closer_notifier: closer_notifier.clone(),
        cleaner: Rc::new(TerminalCleaner {
            writer: writer.clone(),
            reader_task,
            pty_task,
            closer_notifier,
        }),
        id,
        buffer,
        parser,
        writer,
        master,
        cwd,
        title,
        title_notifier,
        clipboard_content,
        clipboard_notifier,
        output_notifier,
        last_write_time: Rc::new(RefCell::new(Instant::now())),
        pressed_button: Rc::new(RefCell::new(None)),
        modifiers: Rc::new(RefCell::new(Modifiers::empty())),
        scroll_velocity: Rc::new(RefCell::new(0.0)),
        scroll_accumulator: Rc::new(RefCell::new(0.0)),
    })
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::io::{AsRawFd, FromRawFd, OwnedFd};

    use portable_pty::{CommandBuilder, MasterPty, PtySize, native_pty_system};

    use crate::fd_pty::RawFdMasterPty;
    use crate::handle::TerminalId;
    use crate::pty::{attach_spawned_child, setup_terminal_from_master};

    #[test]
    fn attach_spawned_child_kills_and_reaps_child_when_setup_fails() {
        // Real child with a master whose writer was already taken, so
        // setup fails at take_writer without closing the underlying fd.
        let _guard = crate::test_support::PTY_FD_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize::default()).expect("openpty");

        let mut cmd = CommandBuilder::new("sleep");
        cmd.arg("5");
        let child = pair.slave.spawn_command(cmd).expect("spawn_command");
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
        let failing_master = RawFdMasterPty::from_owned_fd(unsafe { OwnedFd::from_raw_fd(duped) });
        let _pre_taken_writer = failing_master.take_writer().expect("first take_writer");

        let result = attach_spawned_child(TerminalId::new(), Box::new(failing_master), child, 100);
        assert!(
            result.is_err(),
            "setup must fail against a master whose writer was already taken"
        );

        // Child must already be reaped: waitpid reports ECHILD.
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
}
