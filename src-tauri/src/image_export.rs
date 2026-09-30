//! The month's card as an image, copied to the clipboard or saved where the
//! user picks. The page sends the PNG as a raw IPC body and never holds a
//! path: the save dialog is opened and the file written here (ADR 087).

use std::path::Path;

use tauri::ipc::{InvokeBody, Request};
use tauri_plugin_dialog::DialogExt;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const SUGGESTED_NAME_HEADER: &str = "x-suggested-name";
const FALLBACK_NAME: &str = "JustSay.png";

fn png_body<'a>(request: &'a Request<'_>) -> Result<&'a [u8], String> {
    match request.body() {
        InvokeBody::Raw(bytes) => png_only(bytes),
        InvokeBody::Json(_) => Err("the body is not raw bytes".into()),
    }
}

fn png_only(bytes: &[u8]) -> Result<&[u8], String> {
    if bytes.starts_with(PNG_SIGNATURE) {
        Ok(bytes)
    } else {
        Err("the body is not a PNG image".into())
    }
}

fn suggested_name(request: &Request<'_>) -> String {
    file_name_only(
        request
            .headers()
            .get(SUGGESTED_NAME_HEADER)
            .and_then(|value| value.to_str().ok()),
    )
}

/// The last path component of the page's suggested file name, so the
/// suggestion can name a file and never a folder.
fn file_name_only(suggested: Option<&str>) -> String {
    suggested
        .and_then(|name| Path::new(name).file_name())
        .and_then(|name| name.to_str())
        .unwrap_or(FALLBACK_NAME)
        .to_owned()
}

/// Put the PNG body on the clipboard, kept on this device.
#[tauri::command]
pub fn copy_image(request: Request<'_>) -> Result<(), String> {
    crate::clipboard::write_png(png_body(&request)?)
}

/// Ask where to save the PNG body and write it there. `false` when the user
/// closed the dialog without choosing.
#[tauri::command]
pub async fn save_image(window: tauri::Window, request: Request<'_>) -> Result<bool, String> {
    let png = png_body(&request)?.to_vec();
    let name = suggested_name(&request);
    tauri::async_runtime::spawn_blocking(move || {
        let Some(picked) = window
            .dialog()
            .file()
            .set_parent(&window)
            .set_file_name(name)
            .add_filter("PNG image", &["png"])
            .blocking_save_file()
        else {
            return Ok(false);
        };
        let path = picked
            .into_path()
            .map_err(|e| format!("the chosen place is not a file path — {}", e))?;
        write_image(&path, &png).map(|()| true)
    })
    .await
    .map_err(|e| format!("the save dialog stopped — {}", e))?
}

fn write_image(path: &Path, png: &[u8]) -> Result<(), String> {
    std::fs::write(path, png).map_err(|e| format!("writing {} failed — {}", path.display(), e))
}

#[cfg(test)]
mod tests {
    use super::{file_name_only, png_only, write_image};

    #[test]
    fn only_png_bytes_are_accepted() {
        assert!(png_only(b"\x89PNG\r\n\x1a\nrest").is_ok());
        assert!(png_only(b"GIF89a").is_err());
        assert!(png_only(b"").is_err());
    }

    #[test]
    fn the_suggestion_names_a_file_and_never_a_folder() {
        assert_eq!(
            file_name_only(Some("JustSay-September-2026.png")),
            "JustSay-September-2026.png"
        );
        assert_eq!(file_name_only(Some("../../Windows/evil.png")), "evil.png");
        assert_eq!(file_name_only(Some("..")), "JustSay.png");
        assert_eq!(file_name_only(None), "JustSay.png");
    }

    #[test]
    fn the_file_holds_exactly_the_bytes_given() {
        let folder = std::env::temp_dir().join(format!("justsay-image-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("month.png");
        let png = b"\x89PNG\r\n\x1a\n\x00\xffpixels".to_vec();

        write_image(&path, &png).unwrap();
        let written = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();

        assert_eq!(written, png);
    }

    #[test]
    fn a_path_that_cannot_be_written_is_reported() {
        let missing = std::env::temp_dir()
            .join(format!("justsay-missing-{}", uuid::Uuid::new_v4()))
            .join("month.png");

        assert!(write_image(&missing, b"png").is_err());
    }
}
