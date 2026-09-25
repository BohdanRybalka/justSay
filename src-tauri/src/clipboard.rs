//! Every clipboard write the app makes. The text lands in this machine's
//! clipboard and its local history, marked so the OS never syncs it to the
//! user's other devices: Windows Cloud Clipboard skips an item carrying
//! `CanUploadToCloudClipboard` = 0, and Apple Universal Clipboard skips
//! contents prepared as current-host-only.

/// Put `text` on the clipboard, kept on this device. The error names what
/// failed, for the log; the page shows its own wording.
#[tauri::command]
pub fn write_clipboard_text(text: String) -> Result<(), String> {
    platform::write(&text)
}

#[cfg(windows)]
mod platform {
    use std::{thread, time::Duration};

    /// The registered format Windows reads to decide whether an item may be
    /// uploaded to Cloud Clipboard; a DWORD 0 forbids it and leaves Win+V
    /// history alone.
    const CLOUD_UPLOAD_FORMAT: &str = "CanUploadToCloudClipboard";

    /// Another app can hold the clipboard for a moment; Chromium and Firefox
    /// retry the open this many times, this far apart.
    const OPEN_ATTEMPTS: usize = 5;
    const OPEN_RETRY_DELAY: Duration = Duration::from_millis(5);

    /// The two operations a write makes on an open clipboard.
    pub(super) trait Board {
        fn set_text(&mut self, text: &str) -> Result<(), String>;
        fn set_format(&mut self, name: &str, data: &[u8]) -> Result<(), String>;
    }

    /// Replace the clipboard with `text` and mark it out of Cloud Clipboard
    /// while the clipboard is still open, so no reader sees it unmarked.
    pub(super) fn write_kept_on_this_device(
        board: &mut impl Board,
        text: &str,
    ) -> Result<(), String> {
        board.set_text(text)?;
        board.set_format(CLOUD_UPLOAD_FORMAT, &0u32.to_ne_bytes())
    }

    /// The system clipboard, open for as long as this value lives.
    struct OpenClipboard {
        _held: clipboard_win::Clipboard,
    }

    impl OpenClipboard {
        fn open() -> Result<Self, String> {
            let mut attempts_left = OPEN_ATTEMPTS;
            loop {
                match clipboard_win::Clipboard::new() {
                    Ok(clipboard) => return Ok(Self { _held: clipboard }),
                    Err(e) if attempts_left <= 1 => {
                        return Err(format!("the clipboard stayed busy — {}", e))
                    }
                    Err(_) => {
                        attempts_left -= 1;
                        thread::sleep(OPEN_RETRY_DELAY);
                    }
                }
            }
        }
    }

    impl Board for OpenClipboard {
        fn set_text(&mut self, text: &str) -> Result<(), String> {
            clipboard_win::raw::set_string(text)
                .map_err(|e| format!("writing the text failed — {}", e))
        }

        fn set_format(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
            let format = clipboard_win::register_format(name)
                .ok_or_else(|| format!("registering {} failed", name))?;
            clipboard_win::raw::set_without_clear(format.get(), data)
                .map_err(|e| format!("writing {} failed — {}", name, e))
        }
    }

    pub(super) fn write(text: &str) -> Result<(), String> {
        write_kept_on_this_device(&mut OpenClipboard::open()?, text)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use objc2_app_kit::{NSPasteboard, NSPasteboardContentsOptions, NSPasteboardTypeString};
    use objc2_foundation::NSString;

    /// The two operations a write makes on the general pasteboard.
    pub(super) trait Board {
        fn prepare(&mut self, options: NSPasteboardContentsOptions);
        fn set_string(&mut self, text: &str) -> Result<(), String>;
    }

    /// Clear the pasteboard as current-host-only, then write `text`: the
    /// option applies to the contents written after it, so the order matters.
    pub(super) fn write_kept_on_this_device(
        board: &mut impl Board,
        text: &str,
    ) -> Result<(), String> {
        board.prepare(NSPasteboardContentsOptions::CurrentHostOnly);
        board.set_string(text)
    }

    struct GeneralPasteboard;

    impl Board for GeneralPasteboard {
        fn prepare(&mut self, options: NSPasteboardContentsOptions) {
            NSPasteboard::generalPasteboard().prepareForNewContentsWithOptions(options);
        }

        fn set_string(&mut self, text: &str) -> Result<(), String> {
            let written = NSPasteboard::generalPasteboard()
                .setString_forType(&NSString::from_str(text), unsafe { NSPasteboardTypeString });
            if written {
                Ok(())
            } else {
                Err("the pasteboard refused the text".into())
            }
        }
    }

    pub(super) fn write(text: &str) -> Result<(), String> {
        write_kept_on_this_device(&mut GeneralPasteboard, text)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::platform::{write_kept_on_this_device, Board};

    #[derive(Debug, PartialEq)]
    enum Call {
        Text(String),
        Format(String, Vec<u8>),
    }

    #[derive(Default)]
    struct RecordingBoard {
        calls: Vec<Call>,
        refuse_text: bool,
    }

    impl Board for RecordingBoard {
        fn set_text(&mut self, text: &str) -> Result<(), String> {
            if self.refuse_text {
                return Err("refused".into());
            }
            self.calls.push(Call::Text(text.into()));
            Ok(())
        }

        fn set_format(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
            self.calls.push(Call::Format(name.into(), data.into()));
            Ok(())
        }
    }

    #[test]
    fn the_text_is_written_then_marked_out_of_cloud_clipboard() {
        let mut board = RecordingBoard::default();
        write_kept_on_this_device(&mut board, "привіт").unwrap();
        assert_eq!(
            board.calls,
            vec![
                Call::Text("привіт".into()),
                Call::Format("CanUploadToCloudClipboard".into(), vec![0, 0, 0, 0]),
            ]
        );
    }

    #[test]
    fn a_refused_text_is_reported_and_nothing_is_marked() {
        let mut board = RecordingBoard {
            refuse_text: true,
            ..Default::default()
        };
        assert!(write_kept_on_this_device(&mut board, "text").is_err());
        assert!(board.calls.is_empty());
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::platform::{write_kept_on_this_device, Board};
    use objc2_app_kit::NSPasteboardContentsOptions;

    #[derive(Debug, PartialEq)]
    enum Call {
        Prepare(NSPasteboardContentsOptions),
        Text(String),
    }

    #[derive(Default)]
    struct RecordingBoard {
        calls: Vec<Call>,
        refuse_text: bool,
    }

    impl Board for RecordingBoard {
        fn prepare(&mut self, options: NSPasteboardContentsOptions) {
            self.calls.push(Call::Prepare(options));
        }

        fn set_string(&mut self, text: &str) -> Result<(), String> {
            if self.refuse_text {
                return Err("refused".into());
            }
            self.calls.push(Call::Text(text.into()));
            Ok(())
        }
    }

    #[test]
    fn the_pasteboard_is_prepared_current_host_only_before_the_text() {
        let mut board = RecordingBoard::default();
        write_kept_on_this_device(&mut board, "привіт").unwrap();
        assert_eq!(
            board.calls,
            vec![
                Call::Prepare(NSPasteboardContentsOptions::CurrentHostOnly),
                Call::Text("привіт".into()),
            ]
        );
    }

    #[test]
    fn a_refused_text_is_reported() {
        let mut board = RecordingBoard {
            refuse_text: true,
            ..Default::default()
        };
        assert!(write_kept_on_this_device(&mut board, "text").is_err());
    }
}
