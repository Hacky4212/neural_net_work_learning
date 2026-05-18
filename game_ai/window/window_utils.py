from __future__ import annotations

import ctypes
from dataclasses import dataclass
from ctypes import wintypes


user32 = ctypes.windll.user32
SW_SHOW = 5
SW_RESTORE = 9


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]
    minimized: bool = False


EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def find_window(title: str) -> int | None:
    hwnd = user32.FindWindowW(None, title)
    if hwnd == 0:
        return None
    return int(hwnd)


def list_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(wintypes.HWND(hwnd)):
            return True

        length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
        if length == 0:
            return True

        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(wintypes.HWND(hwnd), buffer, length + 1)
        title = buffer.value.strip()
        rect = get_window_rect(int(hwnd))
        if title and rect:
            windows.append(
                WindowInfo(
                    hwnd=int(hwnd),
                    title=title,
                    rect=rect,
                    minimized=is_minimized(int(hwnd)),
                )
            )
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return windows


def resolve_window(title: str, fallback_keywords: tuple[str, ...] = ()) -> WindowInfo | None:
    hwnd = find_window(title)
    if hwnd is not None:
        rect = get_window_rect(hwnd)
        if rect is not None:
            return WindowInfo(hwnd=hwnd, title=title, rect=rect, minimized=is_minimized(hwnd))

    windows = list_windows()
    for window in windows:
        if title in window.title and _is_usable_window(window):
            return window

    for window in windows:
        if title in window.title:
            return window

    for keyword in fallback_keywords:
        for window in windows:
            if keyword in window.title and _is_usable_window(window):
                return window

    for keyword in fallback_keywords:
        for window in windows:
            if keyword in window.title:
                return window

    return None


def _is_usable_window(window: WindowInfo) -> bool:
    left, top, right, bottom = window.rect
    return not window.minimized and right > left and bottom > top and right > 0 and bottom > 0


def get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    ok = user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    if not ok:
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)


def get_client_screen_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    ok = user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect))
    if not ok:
        return None

    origin = wintypes.POINT(0, 0)
    ok = user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(origin))
    if not ok:
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None

    return (origin.x, origin.y, origin.x + width, origin.y + height)


def is_minimized(hwnd: int) -> bool:
    return bool(user32.IsIconic(wintypes.HWND(hwnd)))


def restore_window(hwnd: int) -> None:
    if is_minimized(hwnd):
        user32.ShowWindow(wintypes.HWND(hwnd), SW_RESTORE)
    else:
        user32.ShowWindow(wintypes.HWND(hwnd), SW_SHOW)


def focus_window(title: str) -> bool:
    window = resolve_window(title)
    if window is None:
        return False

    restore_window(window.hwnd)
    return bool(user32.SetForegroundWindow(wintypes.HWND(window.hwnd)))


def focus_resolved_window(title: str, fallback_keywords: tuple[str, ...]) -> bool:
    window = resolve_window(title, fallback_keywords)
    if window is None:
        return False

    restore_window(window.hwnd)
    return bool(user32.SetForegroundWindow(wintypes.HWND(window.hwnd)))


def window_to_screen(
    title: str,
    x: int,
    y: int,
    fallback_keywords: tuple[str, ...] = (),
    *,
    area: str = "window",
    reference_width: int = 0,
    reference_height: int = 0,
    scale: bool = False,
) -> tuple[int, int] | None:
    window = resolve_window(title, fallback_keywords)
    if window is None:
        return None

    bounds = window.rect
    if area == "client":
        bounds = get_client_screen_rect(window.hwnd) or bounds

    return resolve_point_in_bounds(x, y, bounds, reference_width, reference_height, scale)


def resolve_point_in_bounds(
    x: int,
    y: int,
    bounds: tuple[int, int, int, int],
    reference_width: int = 0,
    reference_height: int = 0,
    scale: bool = False,
) -> tuple[int, int]:
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top

    offset_x = x
    offset_y = y
    if scale and reference_width > 0 and reference_height > 0 and width > 0 and height > 0:
        offset_x = round(x * width / reference_width)
        offset_y = round(y * height / reference_height)

    return (
        _clamp(left + offset_x, left, right - 1),
        _clamp(top + offset_y, top, bottom - 1),
    )


def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value
