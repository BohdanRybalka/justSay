//! An audio file the user picks in the system dialog, opened from the ring or
//! the tray. The main window is told the file's name and size under a token
//! and fetches the bytes with that token once it accepts the file, so the page
//! never holds a path and has no filesystem permission (ADR 087).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::Serialize;
use tauri::ipc::Response;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

const AUDIO_EXTENSIONS: [&str; 13] = [
    ".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac", ".m4a", ".mp4", ".aac", ".opus", ".wma",
    ".aiff", ".aif",
];

const UNREADABLE: &str = "This file can't be read";

/// What the main window hears about a picked file.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PickedFile {
    token: String,
    name: String,
    size: u64,
}

/// The one file picked last, waiting for the main window to take it. A newer
/// pick replaces an older one nobody took.
#[derive(Default)]
pub struct PickedFiles {
    waiting: Mutex<Option<(String, PathBuf)>>,
}

impl PickedFiles {
    fn keep(&self, path: &Path) -> Result<PickedFile, String> {
        let size = std::fs::metadata(path)
            .map_err(|e| format!("reading the size of {} failed — {}", path.display(), e))?
            .len();
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .ok_or_else(|| format!("{} names no file", path.display()))?;
        let token = uuid::Uuid::new_v4().to_string();
        *self.lock() = Some((token.clone(), path.to_path_buf()));
        Ok(PickedFile { token, name, size })
    }

    fn take(&self, token: &str) -> Option<PathBuf> {
        let mut waiting = self.lock();
        match waiting.as_ref() {
            Some((kept, _)) if kept == token => waiting.take().map(|(_, path)| path),
            _ => None,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<(String, PathBuf)>> {
        self.waiting.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Bring the main window forward and open the system file dialog over it.
/// A picked file is announced to that window; a cancelled dialog does nothing.
pub fn pick(app: &AppHandle) {
    crate::show_settings(app);
    let Some(window) = app.get_webview_window("settings") else {
        return;
    };
    let extensions = AUDIO_EXTENSIONS.map(|extension| extension.trim_start_matches('.'));
    let app = app.clone();
    window
        .dialog()
        .file()
        .set_parent(&window)
        .add_filter("Audio", &extensions)
        .pick_file(move |picked| {
            let Some(picked) = picked else {
                return;
            };
            let announced = picked
                .into_path()
                .map_err(|e| format!("the pick is not a file path — {}", e))
                .and_then(|path| announce(&app, &path));
            if let Err(e) = announced {
                log::warn!("The picked file was not handed to the main window: {}", e);
            }
        });
}

fn announce(app: &AppHandle, path: &Path) -> Result<(), String> {
    let file = app.state::<PickedFiles>().keep(path)?;
    app.emit_to("settings", "file-picked", file)
        .map_err(|e| format!("telling the main window failed — {}", e))
}

/// The ring's "Transcribe a file" petal.
#[tauri::command]
pub fn pick_audio_file(app: AppHandle) {
    pick(&app);
}

/// The bytes of the file announced under `token`, as a raw body. A token is
/// good for one take.
#[tauri::command]
pub fn take_picked_file(picked: State<'_, PickedFiles>, token: String) -> Result<Response, String> {
    let path = picked.take(&token).ok_or(UNREADABLE)?;
    read_picked(&path).map(Response::new)
}

fn read_picked(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| {
        log::warn!("Reading the picked file {} failed: {}", path.display(), e);
        UNREADABLE.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::{read_picked, PickedFiles, UNREADABLE};
    use std::path::PathBuf;

    fn audio_file(bytes: &[u8]) -> (PathBuf, PathBuf) {
        let folder = std::env::temp_dir().join(format!("justsay-picked-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("Interview 3.m4a");
        std::fs::write(&path, bytes).unwrap();
        (folder, path)
    }

    #[test]
    fn a_kept_file_is_announced_by_its_name_and_size_and_never_its_folder() {
        let (folder, path) = audio_file(b"twelve bytes");
        let picked = PickedFiles::default().keep(&path).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();

        assert_eq!(picked.name, "Interview 3.m4a");
        assert_eq!(picked.size, 12);
        assert!(!picked.token.is_empty());
    }

    #[test]
    fn the_token_hands_over_the_picked_path_once() {
        let (folder, path) = audio_file(b"audio");
        let files = PickedFiles::default();
        let picked = files.keep(&path).unwrap();

        assert_eq!(files.take("another-token"), None);
        assert_eq!(files.take(&picked.token), Some(path.clone()));
        assert_eq!(files.take(&picked.token), None);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_newer_pick_replaces_one_nobody_took() {
        let (folder, path) = audio_file(b"audio");
        let files = PickedFiles::default();
        let older = files.keep(&path).unwrap();
        let newer = files.keep(&path).unwrap();

        assert_eq!(files.take(&older.token), None);
        assert_eq!(files.take(&newer.token), Some(path));
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn a_file_that_is_gone_is_neither_kept_nor_read() {
        let (folder, path) = audio_file(b"audio");
        std::fs::remove_dir_all(&folder).unwrap();

        assert!(PickedFiles::default().keep(&path).is_err());
        assert_eq!(read_picked(&path), Err(UNREADABLE.to_string()));
    }

    #[test]
    fn the_bytes_read_are_the_file_s_bytes() {
        let (folder, path) = audio_file(b"\x00\xffaudio");
        let bytes = read_picked(&path).unwrap();
        std::fs::remove_dir_all(&folder).unwrap();

        assert_eq!(bytes, b"\x00\xffaudio");
    }
}
