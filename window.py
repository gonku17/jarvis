"""Jarvis's own Mac window: a native AppKit window with a WebKit view showing the orb.

No browser needed. launch.sh runs it with `exec`, so it stays the Jarvis.app process
(same Dock icon, and macOS asks for the microphone on behalf of Jarvis).

Usage:  .venv/bin/python window.py http://127.0.0.1:8765/?app=1
"""

from __future__ import annotations

import sys
from pathlib import Path

import objc
from AppKit import (
    NSApp, NSApplication, NSApplicationActivationPolicyRegular, NSBackingStoreBuffered, NSColor,
    NSImage, NSMakeRect, NSMenu, NSMenuItem, NSObject, NSScreen, NSWindow,
    NSWindowStyleMaskClosable, NSWindowStyleMaskFullSizeContentView, NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable, NSWindowStyleMaskTitled, NSWindowTitleHidden,
)
from Foundation import NSURL, NSProcessInfo, NSURLRequest
from WebKit import WKWebView, WKWebViewConfiguration

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/?app=1"
HERE = Path(__file__).resolve().parent
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


class UIDelegate(NSObject):
    # Let the orb page use the microphone and camera without asking twice: macOS itself still
    # asks once for Jarvis. Only ever for Jarvis's own page on this Mac.
    def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
            self, webview, origin, frame, kind, handler):
        handler(1 if origin.host() in LOCAL_HOSTS else 0)  # 1 = grant, 0 = prompt

    # <a target=_blank> and window.open: open in the normal browser instead of nowhere.
    def webView_createWebViewWithConfiguration_forNavigationAction_windowFeatures_(
            self, webview, config, action, features):
        url = action.request().URL()
        if url is not None:
            from AppKit import NSWorkspace
            NSWorkspace.sharedWorkspace().openURL_(url)
        return None


class AppDelegate(NSObject):
    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return True

    def applicationDidFinishLaunching_(self, note):
        frame = NSMakeRect(0, 0, 980, 860)
        screen = NSScreen.mainScreen()
        if screen is not None:
            vis = screen.visibleFrame()
            frame.size.width = min(980, vis.size.width - 40)
            frame.size.height = min(860, vis.size.height - 40)
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
                 | NSWindowStyleMaskResizable | NSWindowStyleMaskFullSizeContentView)
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False)
        win.setTitle_("Jarvis")
        win.setTitleVisibility_(NSWindowTitleHidden)
        win.setTitlebarAppearsTransparent_(True)
        win.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.02, 0.03, 0.06, 1))
        win.setMinSize_((420, 520))
        win.center()
        win.setFrameAutosaveName_("JarvisMain")  # remembers where you put it

        config = WKWebViewConfiguration.alloc().init()
        config.setMediaTypesRequiringUserActionForPlayback_(0)  # Jarvis can talk without a click
        prefs = config.preferences()
        for setter, value in (("_setDeveloperExtrasEnabled_", True),   # right-click > Inspect
                              ("_setMediaDevicesEnabled_", True),      # navigator.mediaDevices
                              ("_setMediaCaptureRequiresSecureConnection_", False)):
            try:
                getattr(prefs, setter)(value)
            except Exception:
                pass
        web = WKWebView.alloc().initWithFrame_configuration_(win.contentView().bounds(), config)
        web.setAutoresizingMask_(18)  # width + height follow the window
        web.setValue_forKey_(False, "drawsBackground")
        self.ui = UIDelegate.alloc().init()
        web.setUIDelegate_(self.ui)
        web.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(URL)))
        win.setContentView_(web)
        win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        self.win, self.web = win, web

    def reload_(self, sender):
        self.web.reload()


def item(title: str, action: str | None, key: str = "", mask: int | None = None) -> NSMenuItem:
    it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
    if mask is not None:
        it.setKeyEquivalentModifierMask_(mask)
    return it


def build_menu() -> NSMenu:
    bar = NSMenu.alloc().init()

    def sub(title: str, items: list) -> None:
        top = NSMenuItem.alloc().init()
        menu = NSMenu.alloc().initWithTitle_(title)
        for it in items:
            menu.addItem_(it if it is not None else NSMenuItem.separatorItem())
        top.setSubmenu_(menu)
        bar.addItem_(top)

    sub("Jarvis", [item("About Jarvis", "orderFrontStandardAboutPanel:"), None,
                   item("Hide Jarvis", "hide:", "h"), None, item("Quit Jarvis", "terminate:", "q")])
    sub("Edit", [item("Undo", "undo:", "z"), item("Redo", "redo:", "Z"), None,
                 item("Cut", "cut:", "x"), item("Copy", "copy:", "c"), item("Paste", "paste:", "v"),
                 item("Select All", "selectAll:", "a")])
    sub("View", [item("Reload", "reload:", "r"), item("Enter Full Screen", "toggleFullScreen:", "f", 1 << 18 | 1 << 20)])
    sub("Window", [item("Minimise", "performMiniaturize:", "m"), item("Close", "performClose:", "w")])
    return bar


def adopt_jarvis_identity() -> None:
    """We run as Python, so WebKit reads Python's Info.plist and hides the microphone unless it
    finds a usage description. Give the running process Jarvis.app's details (rumps does the same)."""
    from Foundation import NSBundle
    try:
        info = NSBundle.mainBundle().infoDictionary()
        info["CFBundleName"] = "Jarvis"
        info["CFBundleIdentifier"] = "local.jarvis.assistant"
        info["NSMicrophoneUsageDescription"] = 'Jarvis listens for "Hey Jarvis" and to what you say.'
        info["NSCameraUsageDescription"] = 'Jarvis looks through the camera only when you ask.'
    except Exception as e:
        print("couldn't set bundle info:", e, file=sys.stderr)


def main() -> None:
    adopt_jarvis_identity()
    NSProcessInfo.processInfo().setProcessName_("Jarvis")
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    icns = HERE / "Jarvis.app/Contents/Resources/Jarvis.icns"
    if not icns.exists():
        icns = Path.home() / "Applications/Jarvis.app/Contents/Resources/Jarvis.icns"
    if icns.exists():
        app.setApplicationIconImage_(NSImage.alloc().initWithContentsOfFile_(str(icns)))
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.setMainMenu_(build_menu())
    app.run()


if __name__ == "__main__":
    main()
