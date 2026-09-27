//! The folder the backend keeps temporary audio in, opened in the OS file
//! manager at the user's request. Rust-side only, so the page needs no shell grant.

use std::path::Path;
use std::process::Command;

#[cfg(windows)]
const FILE_MANAGER: &str = "explorer";
#[cfg(target_os = "macos")]
const FILE_MANAGER: &str = "open";
#[cfg(not(any(windows, target_os = "macos")))]
const FILE_MANAGER: &str = "xdg-open";

/// Explorer reads `/` as a switch and exits with 1 even after opening the
/// window, so on Windows the path is given with `\` only and its exit code is not read.
fn file_manager_path(dir: &Path) -> String {
    let path = dir.to_string_lossy();
    if cfg!(windows) {
        path.replace('/', "\\")
    } else {
        path.into_owned()
    }
}

/// Creates the folder when no recording has made it yet, then shows it.
#[tauri::command(async)]
pub fn open_scratch_folder() -> Result<(), String> {
    let dir = crate::backend::scratch_dir()
        .ok_or_else(|| "There is no home folder to find the audio in".to_string())?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Could not create {}: {}", dir.display(), e))?;
    let status = Command::new(FILE_MANAGER)
        .arg(file_manager_path(&dir))
        .status()
        .map_err(|e| format!("Could not open {}: {}", dir.display(), e))?;
    if cfg!(windows) || status.success() {
        Ok(())
    } else {
        Err(format!("Could not open {}: {}", dir.display(), status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explorer_gets_a_path_with_backslashes_only() {
        let path = file_manager_path(Path::new("D:/scratch/js\\tmp"));
        if cfg!(windows) {
            assert_eq!(path, "D:\\scratch\\js\\tmp");
        } else {
            assert_eq!(path, "D:/scratch/js\\tmp");
        }
    }
}
