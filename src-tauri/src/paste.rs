//! Dictated text typed into the app the user is working in. The text goes on
//! the clipboard first (kept on this device, see `clipboard.rs`), then one
//! paste chord goes to whichever app has focus — the widget never takes it, so
//! that is the app the user was typing in. Windows silently drops the chord
//! for apps running as administrator; the text is still on the clipboard.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use enigo::{Direction, Enigo, Key, Settings};
use serde::Serialize;
use tauri::AppHandle;

/// Where the text ended up, for the widget's label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PasteOutcome {
    Pasted,
    Copied,
    Failed,
}

/// The default shortcut is Ctrl+Alt+V: a chord sent while the user still
/// holds Alt reaches the app as Ctrl+Alt+V and starts a new recording, and
/// `SendInput` does not lift keys already down. So the chord waits this long
/// for the user's hands to leave the modifiers, then gives up to "copied".
const MODIFIER_RELEASE_BUDGET: Duration = Duration::from_secs(1);
const MODIFIER_POLL_INTERVAL: Duration = Duration::from_millis(15);

/// How long the modifier stays down around V; apps that poll key state
/// instead of reading the event stream miss a shorter chord.
const CHORD_HOLD: Duration = Duration::from_millis(100);

/// Put `text` on the clipboard and paste it into the focused app.
#[tauri::command(async)]
pub fn paste_text(app: AppHandle, text: String) -> PasteOutcome {
    deliver(
        || crate::clipboard::write_clipboard_text(text),
        || {
            wait_for_released_modifiers(
                platform::modifiers_held,
                thread::sleep,
                MODIFIER_RELEASE_BUDGET,
                MODIFIER_POLL_INTERVAL,
            )
        },
        || on_main_thread(&app, send_paste_chord),
    )
}

/// Whether this app may send key events. Only macOS asks: its Accessibility
/// permission gates them.
#[tauri::command]
pub fn paste_permission_granted() -> bool {
    platform::may_send_keys()
}

/// Opens System Settings at Privacy & Security → Accessibility.
#[tauri::command(async)]
pub fn open_accessibility_settings() -> Result<(), String> {
    platform::open_accessibility_settings()
}

fn deliver(
    write: impl FnOnce() -> Result<(), String>,
    modifiers_released: impl FnOnce() -> bool,
    chord: impl FnOnce() -> Result<(), String>,
) -> PasteOutcome {
    if let Err(e) = write() {
        log::warn!("Dictated text did not reach the clipboard — {}", e);
        return PasteOutcome::Failed;
    }
    if !modifiers_released() {
        log::info!("A modifier key stayed down, so the text was only copied");
        return PasteOutcome::Copied;
    }
    match chord() {
        Ok(()) => PasteOutcome::Pasted,
        Err(e) => {
            log::warn!("The paste chord was not sent, so the text was only copied — {}", e);
            PasteOutcome::Copied
        }
    }
}

/// Polls `held` until it reports no modifier down; false once `budget` has
/// been slept through with one still held.
fn wait_for_released_modifiers(
    held: impl Fn() -> bool,
    mut sleep: impl FnMut(Duration),
    budget: Duration,
    interval: Duration,
) -> bool {
    let mut waited = Duration::ZERO;
    while held() {
        if waited >= budget {
            return false;
        }
        sleep(interval);
        waited += interval;
    }
    true
}

/// Recent macOS asserts that keyboard-layout lookups run on the main thread,
/// and the chord looks up the key that types "v"; both platforms send from there.
fn on_main_thread(
    app: &AppHandle,
    task: impl FnOnce() -> Result<(), String> + Send + 'static,
) -> Result<(), String> {
    let (done, result) = mpsc::channel();
    app.run_on_main_thread(move || {
        let _ = done.send(task());
    })
    .map_err(|e| format!("the main thread was not reachable — {}", e))?;
    result
        .recv()
        .map_err(|_| "the main thread dropped the chord".to_string())?
}

/// The one operation a chord makes on a keyboard.
trait Keys {
    fn key(&mut self, key: Key, direction: Direction) -> Result<(), String>;
}

impl Keys for Enigo {
    fn key(&mut self, key: Key, direction: Direction) -> Result<(), String> {
        enigo::Keyboard::key(self, key, direction).map_err(|e| e.to_string())
    }
}

/// Modifier down, key clicked, `hold`, modifier up — the modifier is lifted
/// even when the click failed, so no key is left stuck down.
fn press_chord(
    keys: &mut impl Keys,
    modifier: Key,
    key: Key,
    hold: impl FnOnce(),
) -> Result<(), String> {
    keys.key(modifier, Direction::Press)?;
    let clicked = keys.key(key, Direction::Click);
    hold();
    let released = keys.key(modifier, Direction::Release);
    clicked.and(released)
}

fn send_paste_chord() -> Result<(), String> {
    let mut enigo = Enigo::new(&Settings::default())
        .map_err(|e| format!("key events are not allowed — {}", e))?;
    let (modifier, key) = platform::paste_keys();
    press_chord(&mut enigo, modifier, key, || thread::sleep(CHORD_HOLD))
}

#[cfg(windows)]
mod platform {
    use enigo::Key;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_CONTROL, VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
    };

    const MODIFIERS: [u16; 5] = [VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN];
    const KEY_DOWN: u16 = 0x8000;

    /// The virtual key V: apps read Ctrl+V by virtual key, so it pastes on
    /// every layout, Cyrillic included.
    const VK_V: u32 = 0x56;

    pub(super) fn modifiers_held() -> bool {
        MODIFIERS
            .iter()
            .any(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } as u16 & KEY_DOWN != 0)
    }

    pub(super) fn paste_keys() -> (Key, Key) {
        (Key::Control, Key::Other(VK_V))
    }

    pub(super) fn may_send_keys() -> bool {
        true
    }

    pub(super) fn open_accessibility_settings() -> Result<(), String> {
        Err("Windows has no Accessibility permission to grant".into())
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::ffi::c_void;
    use std::process::Command;

    use enigo::Key;

    const ACCESSIBILITY_PANE: &str =
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility";

    const HID_SYSTEM_STATE: i32 = 1;
    const SHIFT: u64 = 0x0002_0000;
    const CONTROL: u64 = 0x0004_0000;
    const OPTION: u64 = 0x0008_0000;
    const COMMAND: u64 = 0x0010_0000;
    const MODIFIERS: u64 = SHIFT | CONTROL | OPTION | COMMAND;

    /// `kVK_ANSI_V`, the key under V on a US keyboard.
    pub(super) const ANSI_V: u16 = 9;
    const HIGHEST_KEY_CODE: u16 = 127;

    const KEY_ACTION_DISPLAY: u16 = 3;
    const NO_DEAD_KEYS: u32 = 1;
    /// `cmdKey >> 8`: layouts map keys under Command separately, and Cyrillic
    /// ones map them to Latin letters there.
    const COMMAND_HELD: u32 = 1;

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGEventSourceFlagsState(state_id: i32) -> u64;
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrusted() -> u8;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFDataGetBytePtr(data: *const c_void) -> *const u8;
        fn CFRelease(object: *const c_void);
    }

    #[link(name = "Carbon", kind = "framework")]
    extern "C" {
        static kTISPropertyUnicodeKeyLayoutData: *const c_void;
        fn TISCopyCurrentKeyboardLayoutInputSource() -> *mut c_void;
        fn TISGetInputSourceProperty(source: *mut c_void, key: *const c_void) -> *const c_void;
        fn LMGetKbdType() -> u8;
        fn UCKeyTranslate(
            layout: *const u8,
            key_code: u16,
            key_action: u16,
            modifier_state: u32,
            keyboard_type: u32,
            options: u32,
            dead_key_state: *mut u32,
            max_length: usize,
            actual_length: *mut usize,
            unicode: *mut u16,
        ) -> i32;
    }

    pub(super) fn modifiers_held() -> bool {
        let flags = unsafe { CGEventSourceFlagsState(HID_SYSTEM_STATE) };
        flags & MODIFIERS != 0
    }

    /// Main thread only: the layout lookup asserts it.
    pub(super) fn paste_keys() -> (Key, Key) {
        (Key::Meta, Key::Other(u32::from(key_typing_v_in_current_layout())))
    }

    /// The key that types "v" with Command held in the current layout, so
    /// Dvorak pastes too; the ANSI V key when none does.
    pub(super) fn key_typing_v(typed: impl Fn(u16) -> Option<char>) -> u16 {
        (0..=HIGHEST_KEY_CODE)
            .find(|&code| typed(code) == Some('v'))
            .unwrap_or(ANSI_V)
    }

    fn key_typing_v_in_current_layout() -> u16 {
        unsafe {
            let source = TISCopyCurrentKeyboardLayoutInputSource();
            if source.is_null() {
                return ANSI_V;
            }
            let data = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData);
            let code = if data.is_null() {
                ANSI_V
            } else {
                let layout = CFDataGetBytePtr(data);
                key_typing_v(|code| translate(layout, code))
            };
            CFRelease(source);
            code
        }
    }

    unsafe fn translate(layout: *const u8, code: u16) -> Option<char> {
        let mut dead_keys = 0u32;
        let mut length = 0usize;
        let mut unicode = [0u16; 4];
        let status = UCKeyTranslate(
            layout,
            code,
            KEY_ACTION_DISPLAY,
            COMMAND_HELD,
            u32::from(LMGetKbdType()),
            NO_DEAD_KEYS,
            &mut dead_keys,
            unicode.len(),
            &mut length,
            unicode.as_mut_ptr(),
        );
        if status != 0 || length != 1 {
            return None;
        }
        char::decode_utf16([unicode[0]]).next()?.ok()
    }

    pub(super) fn may_send_keys() -> bool {
        unsafe { AXIsProcessTrusted() != 0 }
    }

    pub(super) fn open_accessibility_settings() -> Result<(), String> {
        let status = Command::new("open")
            .arg(ACCESSIBILITY_PANE)
            .status()
            .map_err(|e| format!("Could not open System Settings: {}", e))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("Could not open System Settings: {}", status))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    fn text_on_the_clipboard_and_released_keys_is_pasted() {
        let order = RefCell::new(Vec::new());
        let outcome = deliver(
            || {
                order.borrow_mut().push("write");
                Ok(())
            },
            || true,
            || {
                order.borrow_mut().push("chord");
                Ok(())
            },
        );
        assert_eq!(outcome, PasteOutcome::Pasted);
        assert_eq!(*order.borrow(), vec!["write", "chord"]);
    }

    #[test]
    fn a_failed_clipboard_write_sends_no_chord() {
        let chord_sent = Cell::new(false);
        let outcome = deliver(
            || Err("busy".into()),
            || true,
            || {
                chord_sent.set(true);
                Ok(())
            },
        );
        assert_eq!(outcome, PasteOutcome::Failed);
        assert!(!chord_sent.get());
    }

    #[test]
    fn a_modifier_still_down_leaves_the_text_copied_without_a_chord() {
        let chord_sent = Cell::new(false);
        let outcome = deliver(
            || Ok(()),
            || false,
            || {
                chord_sent.set(true);
                Ok(())
            },
        );
        assert_eq!(outcome, PasteOutcome::Copied);
        assert!(!chord_sent.get());
    }

    #[test]
    fn a_refused_chord_leaves_the_text_copied() {
        let outcome = deliver(|| Ok(()), || true, || Err("no permission".into()));
        assert_eq!(outcome, PasteOutcome::Copied);
    }

    #[test]
    fn the_wait_ends_as_soon_as_the_keys_are_up() {
        let polls = Cell::new(0);
        let slept = RefCell::new(Vec::new());
        let released = wait_for_released_modifiers(
            || {
                polls.set(polls.get() + 1);
                polls.get() <= 3
            },
            |d| slept.borrow_mut().push(d),
            Duration::from_secs(1),
            Duration::from_millis(15),
        );
        assert!(released);
        assert_eq!(slept.borrow().len(), 3);
    }

    #[test]
    fn keys_held_past_the_budget_give_up() {
        let slept = RefCell::new(Duration::ZERO);
        let released = wait_for_released_modifiers(
            || true,
            |d| *slept.borrow_mut() += d,
            Duration::from_millis(100),
            Duration::from_millis(25),
        );
        assert!(!released);
        assert_eq!(*slept.borrow(), Duration::from_millis(100));
    }

    #[test]
    fn keys_already_up_need_no_wait() {
        let slept = Cell::new(false);
        assert!(wait_for_released_modifiers(
            || false,
            |_| slept.set(true),
            Duration::from_secs(1),
            Duration::from_millis(15),
        ));
        assert!(!slept.get());
    }

    #[derive(Default)]
    struct RecordingKeys {
        calls: Vec<String>,
        refuse_click: bool,
    }

    impl Keys for RecordingKeys {
        fn key(&mut self, key: Key, direction: Direction) -> Result<(), String> {
            if self.refuse_click && direction == Direction::Click {
                return Err("refused".into());
            }
            self.calls.push(format!("{:?} {:?}", key, direction));
            Ok(())
        }
    }

    #[test]
    fn the_chord_holds_the_modifier_around_the_key() {
        let mut keys = RecordingKeys::default();
        press_chord(&mut keys, Key::Control, Key::Other(0x56), || {}).unwrap();
        assert_eq!(
            keys.calls,
            vec!["Control Press", "Other(86) Click", "Control Release"]
        );
    }

    #[test]
    fn a_failed_key_still_lifts_the_modifier() {
        let mut keys = RecordingKeys {
            refuse_click: true,
            ..Default::default()
        };
        let held = Cell::new(false);
        let result = press_chord(&mut keys, Key::Control, Key::Other(0x56), || held.set(true));
        assert!(result.is_err());
        assert!(held.get());
        assert_eq!(keys.calls, vec!["Control Press", "Control Release"]);
    }

    #[cfg(windows)]
    #[test]
    fn windows_pastes_with_ctrl_and_the_virtual_key_v() {
        assert_eq!(platform::paste_keys(), (Key::Control, Key::Other(0x56)));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_pastes_with_the_key_that_types_v() {
        let dvorak_v = 47;
        assert_eq!(
            platform::key_typing_v(|code| (code == dvorak_v).then_some('v')),
            dvorak_v
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_falls_back_to_the_ansi_v_key() {
        assert_eq!(platform::key_typing_v(|_| Some('м')), platform::ANSI_V);
    }
}
