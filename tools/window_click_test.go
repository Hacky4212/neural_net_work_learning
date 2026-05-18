package main

import "testing"

func TestMouseLParamPacksClientCoordinates(t *testing.T) {
	got := mouseLParam(640, 360)
	want := uintptr(360<<16 | 640)

	if got != want {
		t.Fatalf("mouseLParam(640, 360) = %d, want %d", got, want)
	}
}

func TestValidClickModes(t *testing.T) {
	for _, mode := range []string{"cursor", "message", "sendinput"} {
		if !isClickMode(mode) {
			t.Fatalf("isClickMode(%q) = false, want true", mode)
		}
	}

	if isClickMode("driver") {
		t.Fatal("isClickMode(\"driver\") = true, want false")
	}
}
