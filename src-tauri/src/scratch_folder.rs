//! The folder the backend keeps temporary audio in, opened in the OS file
//! manager at the user's request. Rust-side only, so the page needs no shell grant.

use std::process::Command;

#[cfg(windows)]
const FILE_MANAGER: &str = "explorer";
#[cfg(target_os = "macos")]
const FILE_MANAGER: &str = "open";
#[cfg(not(any(windows, target_os = "macos")))]
const FILE_MANAGER: &str = "xdg-open";

/// Creates the folder when no recording has made it yet, then shows it.
/// Explorer exits with 1 even when it opened the window, so only a failed
/// launch counts as an error.
#[tauri::command(async)]
pub fn open_scratch_folder() -> Result<(), String> {
    let dir = crate::backend::scratch_dir()
        .ok_or_else(|| "There is no home folder to find the audio in".to_string())?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Could not create {}: {}", dir.display(), e))?;
    Command::new(FILE_MANAGER)
        .arg(&dir)
        .status()
        .map(|_| ())
        .map_err(|e| format!("Could not open {}: {}", dir.display(), e))
}
