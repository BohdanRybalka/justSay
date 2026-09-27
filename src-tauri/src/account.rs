//! Who is using this computer, as the OS knows them. The main window shows
//! this name in its account row until the user can set their own.

/// The account's full name, or its login name when the OS holds no full name
/// for it (a local Windows account often has none). Empty only when the OS
/// answers neither.
#[tauri::command(async)]
pub fn os_display_name() -> String {
    display_name(whoami::realname().ok(), whoami::username().ok())
}

fn display_name(real_name: Option<String>, login_name: Option<String>) -> String {
    [real_name, login_name]
        .into_iter()
        .flatten()
        .map(|name| name.trim().to_string())
        .find(|name| !name.is_empty())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::display_name;

    #[test]
    fn prefers_the_full_name() {
        assert_eq!(
            display_name(Some("Bohdan Rybalka".into()), Some("admin".into())),
            "Bohdan Rybalka"
        );
    }

    #[test]
    fn falls_back_to_the_login_name_when_the_full_name_is_missing_or_blank() {
        assert_eq!(display_name(None, Some("admin".into())), "admin");
        assert_eq!(display_name(Some("  ".into()), Some("admin".into())), "admin");
    }

    #[test]
    fn is_empty_when_the_os_answers_neither() {
        assert_eq!(display_name(None, None), "");
    }
}
