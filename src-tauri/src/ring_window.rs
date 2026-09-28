//! The ring window: five actions around the cursor, opened by a right-click on
//! the widget. It is built once and kept hidden, and like the widget it never
//! takes focus. While it is open a loop sends the page the pointer's offset
//! from the centre, which the page turns into the hovered petal, and closes the
//! ring on a mouse press beyond its reach. Escape is a global shortcut held
//! only while the ring is open, so the key never reaches the user's app.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};
use std::time::Duration;

use serde::Serialize;
use tauri::{webview::WebviewWindowBuilder, AppHandle, Emitter, Manager, WebviewUrl};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};

#[cfg(target_os = "macos")]
use tauri_nspanel::tauri_panel;

const RING_LABEL: &str = "ring";

/// The window's logical side: the petals, the guide and the longest label.
const WINDOW_SIZE: f64 = 200.0;

/// Logical distance kept between the ring's centre and the work area's top.
const EDGE_MARGIN_TOP: f64 = 70.0;

/// Logical distance kept from the sides and the bottom, where the label hangs
/// 58px below the centre and is up to about 180px wide.
const EDGE_MARGIN_SIDES_AND_BOTTOM: f64 = 96.0;

/// Logical distance from the centre within which a press belongs to the ring:
/// a petal's centre sits 38px out and a hovered petal's edge about 18px beyond.
const RING_REACH: f64 = 56.0;

/// The pointer loop's period, about 60 times a second.
const POINTER_POLL: Duration = Duration::from_millis(16);

static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// The ring on screen, if any. `generation` tells one opening from the next,
/// so a loop or a close meant for an earlier opening leaves a newer one alone.
static OPEN_RING: Mutex<Option<OpenRing>> = Mutex::new(None);

#[cfg(target_os = "macos")]
tauri_panel! {
    panel!(RingPanel {
        config: {
            can_become_key_window: false,
            can_become_main_window: false,
            is_floating_panel: true,
            hides_on_deactivate: false
        }
    })
}

/// A point in screen units: physical pixels on Windows, points on macOS.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}

/// A monitor's work area in the same screen units as `Point`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Area {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

/// Where the ring window goes: its top-left corner and its side, in screen units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Frame {
    x: f64,
    y: f64,
    size: f64,
}

/// The cursor and the work area of the monitor under it, with `scale` screen
/// units per logical pixel.
struct Screen {
    cursor: Point,
    work_area: Area,
    scale: f64,
}

#[derive(Clone, Copy)]
struct OpenRing {
    generation: u64,
    centre: Point,
    scale: f64,
}

/// Payload of the event that tells the page where the pointer is, in logical
/// pixels from the ring's centre.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct RingPointer {
    x: f64,
    y: f64,
}

/// Build the hidden ring window. On macOS it becomes a non-activating panel
/// that never becomes key, the same as the widget.
pub fn build(app: &tauri::App) -> tauri::Result<()> {
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let ring = WebviewWindowBuilder::new(app, "ring", WebviewUrl::App("/ring.html".into()))
        .title("")
        .inner_size(WINDOW_SIZE, WINDOW_SIZE)
        .resizable(false)
        .visible(false)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .skip_taskbar(true)
        .shadow(false)
        .focusable(false)
        .focused(false)
        .accept_first_mouse(true)
        .build()?;

    #[cfg(target_os = "macos")]
    {
        use tauri_nspanel::{StyleMask, WebviewWindowExt};

        let panel = ring.to_panel::<RingPanel>()?;
        panel
            .add_style_mask(StyleMask::empty().nonactivating_panel().into())
            .map_err(|e| tauri::Error::Anyhow(e.into()))?;
    }

    Ok(())
}

/// Open the ring centred on the cursor, kept inside the work area of the
/// monitor under it. Opening it while it is open moves it to the cursor.
#[tauri::command]
pub fn open_ring(app: AppHandle) {
    let Some(ring) = app.get_webview_window(RING_LABEL) else {
        return;
    };
    let Some(screen) = platform::screen_under_cursor(&app) else {
        log::warn!("Reading the cursor or its monitor failed, so the ring stays closed");
        return;
    };
    let centre = ring_centre(screen.cursor, screen.work_area, screen.scale);
    if let Err(e) = platform::show_at(&app, &ring, ring_frame(centre, screen.scale)) {
        log::warn!("Showing the ring failed: {}", e);
        return;
    }

    let generation = NEXT_GENERATION.fetch_add(1, Ordering::SeqCst);
    *open_ring_state() = Some(OpenRing { generation, centre, scale: screen.scale });
    hold_escape(&app);
    if let Err(e) = app.emit_to("ring", "ring-opened", ()) {
        log::warn!("Telling the ring it opened failed: {}", e);
    }
    follow_the_pointer(app, generation);
}

/// Close the ring; the page calls this after a petal is picked.
#[tauri::command]
pub fn close_ring(app: AppHandle) {
    close(&app, None);
}

fn open_ring_state() -> std::sync::MutexGuard<'static, Option<OpenRing>> {
    OPEN_RING.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Close the ring on the main thread, where the window and the shortcut live.
/// `generation` limits it to that opening; `None` closes whatever is open.
fn close(app: &AppHandle, generation: Option<u64>) {
    let handle = app.clone();
    let result = app.run_on_main_thread(move || {
        {
            let mut open = open_ring_state();
            match *open {
                Some(ring) if generation.map_or(true, |g| g == ring.generation) => *open = None,
                _ => return,
            }
        }
        if let Some(ring) = handle.get_webview_window(RING_LABEL) {
            if let Err(e) = platform::hide(&handle, &ring) {
                log::warn!("Hiding the ring failed: {}", e);
            }
        }
        release_escape(&handle);
        if let Err(e) = handle.emit_to("ring", "ring-closed", ()) {
            log::warn!("Telling the ring it closed failed: {}", e);
        }
    });
    if let Err(e) = result {
        log::warn!("Closing the ring failed: {}", e);
    }
}

fn escape() -> Shortcut {
    Shortcut::new(None, Code::Escape)
}

/// Take Escape for as long as the ring is open. The plugin runs the handler
/// under its own lock, which releasing the shortcut needs, so the close is
/// handed to another thread.
fn hold_escape(app: &AppHandle) {
    let shortcuts = app.global_shortcut();
    if shortcuts.is_registered(escape()) {
        return;
    }
    let registered = shortcuts.on_shortcut(escape(), |app, _shortcut, event| {
        if event.state == ShortcutState::Pressed {
            let app = app.clone();
            std::thread::spawn(move || close(&app, None));
        }
    });
    if let Err(e) = registered {
        log::warn!("Escape could not be taken, so only a click closes the ring: {}", e);
    }
}

fn release_escape(app: &AppHandle) {
    let shortcuts = app.global_shortcut();
    if !shortcuts.is_registered(escape()) {
        return;
    }
    if let Err(e) = shortcuts.unregister(escape()) {
        log::warn!("Releasing Escape after the ring closed failed: {}", e);
    }
}

/// While this opening lasts, send the page the pointer whenever it moves, and
/// close the ring when a mouse button goes down beyond its reach.
fn follow_the_pointer(app: AppHandle, generation: u64) {
    std::thread::spawn(move || {
        let mut was_down = platform::mouse_button_down();
        let mut sent: Option<RingPointer> = None;
        loop {
            std::thread::sleep(POINTER_POLL);
            let Some(ring) = *open_ring_state() else {
                return;
            };
            if ring.generation != generation {
                return;
            }
            let Some(cursor) = platform::cursor(&app) else {
                continue;
            };
            let pointer = pointer_offset(cursor, ring.centre, ring.scale);
            let down = platform::mouse_button_down();
            if pressed_beyond_reach(was_down, down, pointer) {
                close(&app, Some(generation));
                return;
            }
            was_down = down;
            if sent != Some(pointer) {
                if let Err(e) = app.emit_to("ring", "ring-pointer", pointer) {
                    log::warn!("Telling the ring where the pointer is failed: {}", e);
                }
                sent = Some(pointer);
            }
        }
    });
}

/// The ring's centre: the cursor, moved inside the work area far enough for
/// the petals and the label to stay whole.
fn ring_centre(cursor: Point, work_area: Area, scale: f64) -> Point {
    let side = EDGE_MARGIN_SIDES_AND_BOTTOM * scale;
    let left = work_area.x + side;
    let right = work_area.x + work_area.width - side;
    let top = work_area.y + EDGE_MARGIN_TOP * scale;
    let bottom = work_area.y + work_area.height - side;
    Point {
        x: cursor.x.min(right).max(left),
        y: cursor.y.min(bottom).max(top),
    }
}

/// The window around a centre: the page draws the ring in the window's middle.
fn ring_frame(centre: Point, scale: f64) -> Frame {
    let size = WINDOW_SIZE * scale;
    Frame { x: centre.x - size / 2.0, y: centre.y - size / 2.0, size }
}

/// The pointer's offset from the ring's centre in whole logical pixels, so
/// sub-pixel jitter sends the page nothing.
fn pointer_offset(cursor: Point, centre: Point, scale: f64) -> RingPointer {
    RingPointer {
        x: ((cursor.x - centre.x) / scale).round(),
        y: ((cursor.y - centre.y) / scale).round(),
    }
}

/// A button that went down on this tick, beyond the ring's reach. One already
/// held when the ring opened, the right-click that opened it, does not count.
fn pressed_beyond_reach(was_down: bool, down: bool, pointer: RingPointer) -> bool {
    down && !was_down && pointer.x.hypot(pointer.y) > RING_REACH
}

/// Bring the ring's panel to the front without making it key, or take it away.
#[cfg(target_os = "macos")]
fn order_panel(app: &AppHandle, front: bool) -> Result<(), String> {
    use tauri_nspanel::ManagerExt;

    let panel = app.get_webview_panel(RING_LABEL).map_err(|e| format!("{:?}", e))?;
    if front {
        panel.show();
    } else {
        panel.hide();
    }
    Ok(())
}

#[cfg(windows)]
mod platform {
    use super::{Area, Frame, Point, Screen};
    use tauri::{AppHandle, WebviewWindow};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, HWND_TOPMOST, SET_WINDOW_POS_FLAGS, SWP_HIDEWINDOW, SWP_NOACTIVATE,
        SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SWP_SHOWWINDOW,
    };

    const KEY_DOWN: u16 = 0x8000;

    pub(super) fn cursor(app: &AppHandle) -> Option<Point> {
        let cursor = app.cursor_position().ok()?;
        Some(Point { x: cursor.x, y: cursor.y })
    }

    pub(super) fn screen_under_cursor(app: &AppHandle) -> Option<Screen> {
        let cursor = cursor(app)?;
        let monitor = app
            .monitor_from_point(cursor.x, cursor.y)
            .ok()
            .flatten()
            .or_else(|| app.primary_monitor().ok().flatten())?;
        let area = monitor.work_area();
        Some(Screen {
            cursor,
            work_area: Area {
                x: f64::from(area.position.x),
                y: f64::from(area.position.y),
                width: f64::from(area.size.width),
                height: f64::from(area.size.height),
            },
            scale: monitor.scale_factor(),
        })
    }

    /// Placed, sized and shown in one `SetWindowPos` in physical pixels, then
    /// placed again: tao answers the move to a monitor of another scale by
    /// moving the window itself. Neither call activates it.
    pub(super) fn show_at(_app: &AppHandle, ring: &WebviewWindow, frame: Frame) -> Result<(), String> {
        let (x, y, size) = (frame.x.round() as i32, frame.y.round() as i32, frame.size.round() as i32);
        set_window_pos(ring, x, y, size, SWP_NOACTIVATE | SWP_SHOWWINDOW)?;
        set_window_pos(ring, x, y, size, SWP_NOACTIVATE)
    }

    pub(super) fn hide(_app: &AppHandle, ring: &WebviewWindow) -> Result<(), String> {
        set_window_pos(
            ring,
            0,
            0,
            0,
            SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_HIDEWINDOW,
        )
    }

    fn set_window_pos(
        ring: &WebviewWindow,
        x: i32,
        y: i32,
        size: i32,
        flags: SET_WINDOW_POS_FLAGS,
    ) -> Result<(), String> {
        let hwnd = ring.hwnd().map_err(|e| e.to_string())?;
        let placed = unsafe { SetWindowPos(hwnd.0 as _, HWND_TOPMOST, x, y, size, size, flags) };
        if placed == 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        Ok(())
    }

    pub(super) fn mouse_button_down() -> bool {
        [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON]
            .iter()
            .any(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } as u16 & KEY_DOWN != 0)
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::{Area, Frame, Point, Screen};
    use objc2_app_kit::NSEvent;
    use tauri::{AppHandle, LogicalPosition, WebviewWindow};

    /// The cursor in points. tao reports it scaled by the primary display's
    /// factor, whatever display it is on.
    pub(super) fn cursor(app: &AppHandle) -> Option<Point> {
        let cursor = app.cursor_position().ok()?;
        let scale = app.primary_monitor().ok().flatten()?.scale_factor();
        Some(Point { x: cursor.x / scale, y: cursor.y / scale })
    }

    /// Everything in points, where one logical pixel is one unit on every
    /// display; tao reports a work area scaled by its own display's factor.
    pub(super) fn screen_under_cursor(app: &AppHandle) -> Option<Screen> {
        let cursor = cursor(app)?;
        let monitor = app
            .monitor_from_point(cursor.x, cursor.y)
            .ok()
            .flatten()
            .or_else(|| app.primary_monitor().ok().flatten())?;
        let scale = monitor.scale_factor();
        let area = monitor.work_area();
        Some(Screen {
            cursor,
            work_area: Area {
                x: f64::from(area.position.x) / scale,
                y: f64::from(area.position.y) / scale,
                width: f64::from(area.size.width) / scale,
                height: f64::from(area.size.height) / scale,
            },
            scale: 1.0,
        })
    }

    pub(super) fn show_at(app: &AppHandle, ring: &WebviewWindow, frame: Frame) -> Result<(), String> {
        ring.set_position(LogicalPosition::new(frame.x, frame.y))
            .map_err(|e| e.to_string())?;
        super::order_panel(app, true)
    }

    pub(super) fn hide(app: &AppHandle, _ring: &WebviewWindow) -> Result<(), String> {
        super::order_panel(app, false)
    }

    pub(super) fn mouse_button_down() -> bool {
        NSEvent::pressedMouseButtons() != 0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        pointer_offset, pressed_beyond_reach, ring_centre, ring_frame, Area, Frame, Point,
        RingPointer,
    };

    const FULL_HD: Area = Area { x: 0.0, y: 0.0, width: 1920.0, height: 1032.0 };

    fn centre(x: f64, y: f64, area: Area, scale: f64) -> Point {
        ring_centre(Point { x, y }, area, scale)
    }

    #[test]
    fn away_from_the_edges_the_ring_opens_on_the_cursor() {
        assert_eq!(centre(900.0, 500.0, FULL_HD, 1.0), Point { x: 900.0, y: 500.0 });
    }

    #[test]
    fn near_an_edge_the_ring_opens_far_enough_inside_to_be_whole() {
        assert_eq!(centre(3.0, 2.0, FULL_HD, 1.0), Point { x: 96.0, y: 70.0 });
        assert_eq!(centre(1919.0, 1031.0, FULL_HD, 1.0), Point { x: 1824.0, y: 936.0 });
        assert_eq!(centre(96.0, 70.0, FULL_HD, 1.0), Point { x: 96.0, y: 70.0 });
        assert_eq!(centre(95.0, 69.0, FULL_HD, 1.0), Point { x: 96.0, y: 70.0 });
    }

    #[test]
    fn the_margins_scale_to_physical_pixels() {
        let area = Area { x: 0.0, y: 0.0, width: 3840.0, height: 2064.0 };
        assert_eq!(centre(0.0, 0.0, area, 1.5), Point { x: 144.0, y: 105.0 });
        assert_eq!(centre(3840.0, 2064.0, area, 1.5), Point { x: 3696.0, y: 1920.0 });
        assert_eq!(
            ring_frame(Point { x: 1000.0, y: 600.0 }, 1.5),
            Frame { x: 850.0, y: 450.0, size: 300.0 }
        );
    }

    #[test]
    fn a_monitor_left_of_or_above_the_primary_keeps_the_ring_on_itself() {
        let left_monitor = Area { x: -1920.0, y: -300.0, width: 1920.0, height: 1080.0 };
        assert_eq!(centre(-1.0, -299.0, left_monitor, 1.0), Point { x: -96.0, y: -230.0 });
        assert_eq!(centre(-1900.0, 770.0, left_monitor, 1.0), Point { x: -1824.0, y: 684.0 });
        assert_eq!(
            ring_frame(Point { x: -96.0, y: -230.0 }, 1.0),
            Frame { x: -196.0, y: -330.0, size: 200.0 }
        );
    }

    #[test]
    fn the_pointer_is_sent_in_whole_logical_pixels_from_the_centre() {
        let centre = Point { x: 1000.0, y: 600.0 };
        assert_eq!(
            pointer_offset(Point { x: 1000.0, y: 543.0 }, centre, 1.5),
            RingPointer { x: 0.0, y: -38.0 }
        );
        assert_eq!(
            pointer_offset(Point { x: 1000.4, y: 599.8 }, centre, 1.0),
            RingPointer { x: 0.0, y: -0.0 }
        );
    }

    #[test]
    fn only_a_fresh_press_beyond_the_petals_closes_the_ring() {
        let far = RingPointer { x: 60.0, y: 0.0 };
        let on_a_petal = RingPointer { x: 0.0, y: -38.0 };
        let hovered_petal_edge = RingPointer { x: 0.0, y: -56.0 };

        assert!(pressed_beyond_reach(false, true, far));
        assert!(!pressed_beyond_reach(true, true, far), "the right-click still held from opening");
        assert!(!pressed_beyond_reach(false, false, far));
        assert!(!pressed_beyond_reach(false, true, on_a_petal));
        assert!(!pressed_beyond_reach(false, true, hovered_petal_edge));
    }
}
