//! Every clipboard write the app makes. The text or image lands in this
//! machine's clipboard and its local history, marked so the OS never syncs it
//! to the user's other devices: Windows Cloud Clipboard skips an item carrying
//! `CanUploadToCloudClipboard` = 0, and Apple Universal Clipboard skips
//! contents prepared as current-host-only.

/// Put `text` on the clipboard, kept on this device. The error names what
/// failed, for the log; the page shows its own wording.
#[tauri::command]
pub fn write_clipboard_text(text: String) -> Result<(), String> {
    platform::write(&text)
}

/// Put the PNG image `png` on the clipboard, kept on this device, in the
/// formats the platform's paste targets read.
pub fn write_png(png: &[u8]) -> Result<(), String> {
    platform::write_png(png)
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

    /// The registered format Chromium, Office and Qt read a PNG image from.
    const PNG_FORMAT: &str = "PNG";

    /// The operations a write makes on an open clipboard.
    pub(super) trait Board {
        fn set_text(&mut self, text: &str) -> Result<(), String>;
        fn set_image(&mut self, png: &[u8], dib: &[u8]) -> Result<(), String>;
        fn set_format(&mut self, name: &str, data: &[u8]) -> Result<(), String>;
        fn empty(&mut self);
    }

    /// Replace the clipboard with `text` and mark it out of Cloud Clipboard
    /// while the clipboard is still open, so no reader sees it unmarked. Text
    /// that cannot be marked is taken off again rather than left to sync.
    pub(super) fn write_kept_on_this_device(
        board: &mut impl Board,
        text: &str,
    ) -> Result<(), String> {
        board.set_text(text)?;
        mark_kept_on_this_device(board)
    }

    /// Replace the clipboard with an image as `PNG` and as a device-independent
    /// bitmap, then mark it as text is marked. An image written in part is
    /// taken off again.
    pub(super) fn write_image_kept_on_this_device(
        board: &mut impl Board,
        png: &[u8],
        dib: &[u8],
    ) -> Result<(), String> {
        board.set_image(png, dib).inspect_err(|_| board.empty())?;
        mark_kept_on_this_device(board)
    }

    fn mark_kept_on_this_device(board: &mut impl Board) -> Result<(), String> {
        board
            .set_format(CLOUD_UPLOAD_FORMAT, &0u32.to_ne_bytes())
            .inspect_err(|_| board.empty())
    }

    /// A `CF_DIB` of the straight-alpha RGBA `pixels`, laid over white: GDI
    /// readers such as Paint ignore alpha and would paint clear pixels black.
    pub(super) fn dib_on_white(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
        const HEADER_SIZE: u32 = 40;
        let image_size = width * height * 4;
        let mut dib = Vec::with_capacity((HEADER_SIZE + image_size) as usize);
        dib.extend_from_slice(&HEADER_SIZE.to_le_bytes());
        dib.extend_from_slice(&(width as i32).to_le_bytes());
        dib.extend_from_slice(&(height as i32).to_le_bytes());
        dib.extend_from_slice(&1u16.to_le_bytes());
        dib.extend_from_slice(&32u16.to_le_bytes());
        dib.extend_from_slice(&0u32.to_le_bytes());
        dib.extend_from_slice(&image_size.to_le_bytes());
        dib.extend_from_slice(&[0; 16]);
        let over_white = |channel: u8, alpha: u8| {
            ((channel as u32 * alpha as u32 + 255 * (255 - alpha as u32) + 127) / 255) as u8
        };
        for row in pixels.chunks_exact(width as usize * 4).rev() {
            for pixel in row.chunks_exact(4) {
                let alpha = pixel[3];
                dib.extend_from_slice(&[
                    over_white(pixel[2], alpha),
                    over_white(pixel[1], alpha),
                    over_white(pixel[0], alpha),
                    255,
                ]);
            }
        }
        dib
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

        fn set_image(&mut self, png: &[u8], dib: &[u8]) -> Result<(), String> {
            clipboard_win::raw::empty()
                .map_err(|e| format!("emptying the clipboard failed — {}", e))?;
            self.set_format(PNG_FORMAT, png)?;
            clipboard_win::raw::set_without_clear(clipboard_win::formats::CF_DIB, dib)
                .map_err(|e| format!("writing the bitmap failed — {}", e))
        }

        fn set_format(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
            let format = clipboard_win::register_format(name)
                .ok_or_else(|| format!("registering {} failed", name))?;
            clipboard_win::raw::set_without_clear(format.get(), data)
                .map_err(|e| format!("writing {} failed — {}", name, e))
        }

        fn empty(&mut self) {
            if let Err(e) = clipboard_win::raw::empty() {
                log::warn!("Emptying the clipboard after a failed write failed — {}", e);
            }
        }
    }

    pub(super) fn write(text: &str) -> Result<(), String> {
        write_kept_on_this_device(&mut OpenClipboard::open()?, text)
    }

    pub(super) fn write_png(png: &[u8]) -> Result<(), String> {
        let image = tauri::image::Image::from_bytes(png)
            .map_err(|e| format!("reading the image failed — {}", e))?;
        let dib = dib_on_white(image.rgba(), image.width(), image.height());
        write_image_kept_on_this_device(&mut OpenClipboard::open()?, png, &dib)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use objc2_app_kit::{
        NSPasteboard, NSPasteboardContentsOptions, NSPasteboardTypePNG, NSPasteboardTypeString,
    };
    use objc2_foundation::{NSData, NSString};

    /// The operations a write makes on the general pasteboard.
    pub(super) trait Board {
        fn prepare(&mut self, options: NSPasteboardContentsOptions);
        fn set_string(&mut self, text: &str) -> Result<(), String>;
        fn set_png(&mut self, png: &[u8]) -> Result<(), String>;
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

    /// Clear the pasteboard as current-host-only, then write `png` as
    /// `public.png`, which AppKit's image readers accept.
    pub(super) fn write_image_kept_on_this_device(
        board: &mut impl Board,
        png: &[u8],
    ) -> Result<(), String> {
        board.prepare(NSPasteboardContentsOptions::CurrentHostOnly);
        board.set_png(png)
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

        fn set_png(&mut self, png: &[u8]) -> Result<(), String> {
            let written = NSPasteboard::generalPasteboard()
                .setData_forType(Some(&NSData::with_bytes(png)), unsafe {
                    NSPasteboardTypePNG
                });
            if written {
                Ok(())
            } else {
                Err("the pasteboard refused the image".into())
            }
        }
    }

    pub(super) fn write(text: &str) -> Result<(), String> {
        write_kept_on_this_device(&mut GeneralPasteboard, text)
    }

    pub(super) fn write_png(png: &[u8]) -> Result<(), String> {
        write_image_kept_on_this_device(&mut GeneralPasteboard, png)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::platform::{
        dib_on_white, write_image_kept_on_this_device, write_kept_on_this_device, Board,
    };

    #[derive(Debug, PartialEq)]
    enum Call {
        Text(String),
        Image(Vec<u8>, Vec<u8>),
        Format(String, Vec<u8>),
        Empty,
    }

    #[derive(Default)]
    struct RecordingBoard {
        calls: Vec<Call>,
        refuse_text: bool,
        refuse_image: bool,
        refuse_format: bool,
    }

    impl Board for RecordingBoard {
        fn set_text(&mut self, text: &str) -> Result<(), String> {
            if self.refuse_text {
                return Err("refused".into());
            }
            self.calls.push(Call::Text(text.into()));
            Ok(())
        }

        fn set_image(&mut self, png: &[u8], dib: &[u8]) -> Result<(), String> {
            if self.refuse_image {
                return Err("refused".into());
            }
            self.calls.push(Call::Image(png.into(), dib.into()));
            Ok(())
        }

        fn set_format(&mut self, name: &str, data: &[u8]) -> Result<(), String> {
            if self.refuse_format {
                return Err("refused".into());
            }
            self.calls.push(Call::Format(name.into(), data.into()));
            Ok(())
        }

        fn empty(&mut self) {
            self.calls.push(Call::Empty);
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

    #[test]
    fn text_that_cannot_be_marked_is_taken_off_the_clipboard() {
        let mut board = RecordingBoard {
            refuse_format: true,
            ..Default::default()
        };
        assert!(write_kept_on_this_device(&mut board, "text").is_err());
        assert_eq!(board.calls, vec![Call::Text("text".into()), Call::Empty]);
    }

    #[test]
    fn the_image_is_written_then_marked_out_of_cloud_clipboard() {
        let mut board = RecordingBoard::default();
        write_image_kept_on_this_device(&mut board, b"png", b"dib").unwrap();
        assert_eq!(
            board.calls,
            vec![
                Call::Image(b"png".to_vec(), b"dib".to_vec()),
                Call::Format("CanUploadToCloudClipboard".into(), vec![0, 0, 0, 0]),
            ]
        );
    }

    #[test]
    fn an_image_written_in_part_is_taken_off_the_clipboard() {
        let mut board = RecordingBoard {
            refuse_image: true,
            ..Default::default()
        };
        assert!(write_image_kept_on_this_device(&mut board, b"png", b"dib").is_err());
        assert_eq!(board.calls, vec![Call::Empty]);
    }

    #[test]
    fn the_bitmap_is_bottom_up_bgr_laid_over_white() {
        let top_red_opaque = [255, 0, 0, 255];
        let bottom_clear = [0, 0, 0, 0];
        let dib = dib_on_white(&[top_red_opaque, bottom_clear].concat(), 1, 2);

        assert_eq!(dib.len(), 40 + 8);
        assert_eq!(&dib[0..4], &40u32.to_le_bytes());
        assert_eq!(&dib[4..8], &1i32.to_le_bytes());
        assert_eq!(&dib[8..12], &2i32.to_le_bytes());
        assert_eq!(&dib[14..16], &32u16.to_le_bytes());
        assert_eq!(&dib[20..24], &8u32.to_le_bytes());
        assert_eq!(&dib[40..], &[255, 255, 255, 255, 0, 0, 255, 255]);
    }

    #[test]
    fn a_half_clear_pixel_is_blended_halfway_to_white() {
        let dib = dib_on_white(&[0, 0, 0, 128], 1, 1);
        assert_eq!(&dib[40..], &[127, 127, 127, 255]);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::platform::{write_image_kept_on_this_device, write_kept_on_this_device, Board};
    use objc2_app_kit::NSPasteboardContentsOptions;

    #[derive(Debug, PartialEq)]
    enum Call {
        Prepare(NSPasteboardContentsOptions),
        Text(String),
        Png(Vec<u8>),
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

        fn set_png(&mut self, png: &[u8]) -> Result<(), String> {
            self.calls.push(Call::Png(png.into()));
            Ok(())
        }
    }

    #[test]
    fn the_pasteboard_is_prepared_current_host_only_before_the_image() {
        let mut board = RecordingBoard::default();
        write_image_kept_on_this_device(&mut board, b"png").unwrap();
        assert_eq!(
            board.calls,
            vec![
                Call::Prepare(NSPasteboardContentsOptions::CurrentHostOnly),
                Call::Png(b"png".to_vec()),
            ]
        );
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
