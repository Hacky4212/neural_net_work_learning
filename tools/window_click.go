package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

type rect struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

type point struct {
	X int32
	Y int32
}

type bounds struct {
	Left   int32
	Top    int32
	Right  int32
	Bottom int32
}

var (
	user32                 = syscall.NewLazyDLL("user32.dll")
	procSetProcessDPIAware = user32.NewProc("SetProcessDPIAware")
	procFindWindowW        = user32.NewProc("FindWindowW")
	procEnumWindows        = user32.NewProc("EnumWindows")
	procIsWindowVisible    = user32.NewProc("IsWindowVisible")
	procGetWindowTextLenW  = user32.NewProc("GetWindowTextLengthW")
	procGetWindowTextW     = user32.NewProc("GetWindowTextW")
	procGetWindowRect      = user32.NewProc("GetWindowRect")
	procGetClientRect      = user32.NewProc("GetClientRect")
	procClientToScreen     = user32.NewProc("ClientToScreen")
	procIsIconic           = user32.NewProc("IsIconic")
	procShowWindow         = user32.NewProc("ShowWindow")
	procSetForeground      = user32.NewProc("SetForegroundWindow")
	procSetCursorPos       = user32.NewProc("SetCursorPos")
	procMouseEvent         = user32.NewProc("mouse_event")
	procScreenToClient     = user32.NewProc("ScreenToClient")
	procPostMessageW       = user32.NewProc("PostMessageW")
)

const (
	swRestore         = 9
	mouseLeftDown     = 0x0002
	mouseLeftUp       = 0x0004
	wmLButtonDown     = 0x0201
	wmLButtonUp       = 0x0202
	mkLButton         = 0x0001
	defaultClickDelay = 80 * time.Millisecond
)

func main() {
	title := flag.String("title", "", "target window title")
	x := flag.Int("x", -1, "reference-space window x coordinate")
	y := flag.Int("y", -1, "reference-space window y coordinate")
	refW := flag.Int("refw", 0, "reference window width")
	refH := flag.Int("refh", 0, "reference window height")
	area := flag.String("area", "client", "client or window")
	scale := flag.Bool("scale", true, "scale reference coordinates to the current window size")
	focus := flag.Bool("focus", true, "restore and focus the target window before clicking")
	mode := flag.String("mode", "cursor", "cursor or message")
	dryRun := flag.Bool("dry-run", false, "print coordinates without clicking")
	flag.Parse()

	if *title == "" || *x < 0 || *y < 0 {
		exitf("usage: window_click.exe -title \"MIRMG(1)\" -x 640 -y 360 -refw 1280 -refh 720")
	}
	if *scale && (*refW <= 0 || *refH <= 0) {
		exitf("refw and refh are required when scale=true")
	}
	if *area != "client" && *area != "window" {
		exitf("area must be client or window")
	}
	if !isClickMode(*mode) {
		exitf("mode must be cursor or message")
	}

	// Makes window rectangles and cursor coordinates use physical pixels.
	procSetProcessDPIAware.Call()

	hwnd, actualTitle, err := findWindow(*title)
	if err != nil {
		exitf(err.Error())
	}

	if *focus && *mode == "cursor" {
		restoreAndFocus(hwnd)
		time.Sleep(defaultClickDelay)
	}

	targetBounds, err := getBounds(hwnd, *area)
	if err != nil {
		exitf(err.Error())
	}

	screenX, screenY := resolvePoint(*x, *y, *refW, *refH, *scale, targetBounds)
	clientX, clientY, err := screenToClient(hwnd, screenX, screenY)
	if err != nil {
		exitf(err.Error())
	}

	fmt.Printf(
		"title=%q mode=%s area=%s bounds=%d,%d,%d,%d ref=%dx%d input=%d,%d screen=%d,%d client=%d,%d\n",
		actualTitle,
		*mode,
		*area,
		targetBounds.Left,
		targetBounds.Top,
		targetBounds.Right,
		targetBounds.Bottom,
		*refW,
		*refH,
		*x,
		*y,
		screenX,
		screenY,
		clientX,
		clientY,
	)

	if *dryRun {
		return
	}

	if *mode == "message" {
		if err := messageClick(hwnd, clientX, clientY); err != nil {
			exitf(err.Error())
		}
		return
	}

	cursorClick(screenX, screenY)
}

func findWindow(title string) (uintptr, string, error) {
	titlePtr, _ := syscall.UTF16PtrFromString(title)
	hwnd, _, _ := procFindWindowW.Call(0, uintptr(unsafe.Pointer(titlePtr)))
	if hwnd != 0 {
		return hwnd, windowTitle(hwnd), nil
	}

	var exact uintptr
	var exactTitle string
	var partial uintptr
	var partialTitle string

	callback := syscall.NewCallback(func(hwnd uintptr, _ uintptr) uintptr {
		visible, _, _ := procIsWindowVisible.Call(hwnd)
		if visible == 0 {
			return 1
		}

		text := windowTitle(hwnd)
		if text == "" {
			return 1
		}

		if text == title {
			exact = hwnd
			exactTitle = text
			return 0
		}

		if partial == 0 && strings.Contains(text, title) {
			partial = hwnd
			partialTitle = text
		}
		return 1
	})

	procEnumWindows.Call(callback, 0)
	if exact != 0 {
		return exact, exactTitle, nil
	}
	if partial != 0 {
		return partial, partialTitle, nil
	}

	return 0, "", fmt.Errorf("target window not found: %s", title)
}

func windowTitle(hwnd uintptr) string {
	length, _, _ := procGetWindowTextLenW.Call(hwnd)
	if length == 0 {
		return ""
	}

	buffer := make([]uint16, length+1)
	procGetWindowTextW.Call(hwnd, uintptr(unsafe.Pointer(&buffer[0])), length+1)
	return syscall.UTF16ToString(buffer)
}

func restoreAndFocus(hwnd uintptr) {
	minimized, _, _ := procIsIconic.Call(hwnd)
	if minimized != 0 {
		procShowWindow.Call(hwnd, swRestore)
	}
	procSetForeground.Call(hwnd)
}

func getBounds(hwnd uintptr, area string) (bounds, error) {
	if area == "client" {
		var client rect
		ok, _, _ := procGetClientRect.Call(hwnd, uintptr(unsafe.Pointer(&client)))
		if ok != 0 {
			origin := point{}
			ok, _, _ = procClientToScreen.Call(hwnd, uintptr(unsafe.Pointer(&origin)))
			if ok != 0 {
				width := client.Right - client.Left
				height := client.Bottom - client.Top
				if width > 0 && height > 0 {
					return bounds{
						Left:   origin.X,
						Top:    origin.Y,
						Right:  origin.X + width,
						Bottom: origin.Y + height,
					}, nil
				}
			}
		}
	}

	var window rect
	ok, _, _ := procGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&window)))
	if ok == 0 {
		return bounds{}, fmt.Errorf("failed to read target window bounds")
	}
	if window.Right <= window.Left || window.Bottom <= window.Top {
		return bounds{}, fmt.Errorf("target window bounds are empty")
	}
	return bounds(window), nil
}

func resolvePoint(x int, y int, refW int, refH int, scale bool, area bounds) (int, int) {
	width := int(area.Right - area.Left)
	height := int(area.Bottom - area.Top)

	offsetX := x
	offsetY := y
	if scale {
		offsetX = int(math.Round(float64(x) * float64(width) / float64(refW)))
		offsetY = int(math.Round(float64(y) * float64(height) / float64(refH)))
	}

	screenX := clamp(int(area.Left)+offsetX, int(area.Left), int(area.Right)-1)
	screenY := clamp(int(area.Top)+offsetY, int(area.Top), int(area.Bottom)-1)
	return screenX, screenY
}

func cursorClick(x int, y int) {
	procSetCursorPos.Call(uintptr(x), uintptr(y))
	procMouseEvent.Call(mouseLeftDown, 0, 0, 0, 0)
	time.Sleep(20 * time.Millisecond)
	procMouseEvent.Call(mouseLeftUp, 0, 0, 0, 0)
}

func messageClick(hwnd uintptr, clientX int, clientY int) error {
	lparam := mouseLParam(clientX, clientY)
	downOK, _, _ := procPostMessageW.Call(hwnd, wmLButtonDown, mkLButton, lparam)
	if downOK == 0 {
		return fmt.Errorf("failed to post WM_LBUTTONDOWN")
	}

	time.Sleep(20 * time.Millisecond)
	upOK, _, _ := procPostMessageW.Call(hwnd, wmLButtonUp, 0, lparam)
	if upOK == 0 {
		return fmt.Errorf("failed to post WM_LBUTTONUP")
	}
	return nil
}

func screenToClient(hwnd uintptr, screenX int, screenY int) (int, int, error) {
	p := point{X: int32(screenX), Y: int32(screenY)}
	ok, _, _ := procScreenToClient.Call(hwnd, uintptr(unsafe.Pointer(&p)))
	if ok == 0 {
		return 0, 0, fmt.Errorf("failed to convert screen point to client point")
	}
	return int(p.X), int(p.Y), nil
}

func mouseLParam(clientX int, clientY int) uintptr {
	return uintptr((clientY&0xffff)<<16 | (clientX & 0xffff))
}

func isClickMode(mode string) bool {
	return mode == "cursor" || mode == "message"
}

func clamp(value int, low int, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

func exitf(message string) {
	fmt.Fprintln(os.Stderr, message)
	os.Exit(1)
}
