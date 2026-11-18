"""Always-on Jarvis.

Two parts:
  * LocalAudio runs inside the server: it feeds the Mac's microphone into a normal Session and
    plays the replies through the speakers, so "Hey Jarvis" works with no window open. It pauses
    while the orb window is open (the window has the mic then).
  * `python background.py menubar` is a small menu bar app (rumps) that talks to the server over
    http://127.0.0.1:8765/api/... It quits when the server goes away.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import queue
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.background")
FRAME = 1280


class FakeSocket:
    """Stands in for the browser's WebSocket so a Session can use the Mac's own speakers."""

    def __init__(self, local: "LocalAudio") -> None:
        self.local = local

    async def send_text(self, text: str) -> None:
        msg = json.loads(text)
        kind = msg.get("type")
        if kind == "state":
            self.local.status["state"] = msg["state"]
        elif kind == "brain":
            self.local.status["brain"] = msg["brain"]
        elif kind == "stop_audio":
            self.local.player.stop()
        elif kind == "speech_end":
            self.local.player.end_of_speech()

    async def send_bytes(self, data: bytes) -> None:
        self.local.player.add(data)


class Player:
    """Plays Jarvis's WAV sentences one after another through the default speaker."""

    def __init__(self, on_done) -> None:
        import sounddevice as sd

        self.sd = sd
        self.q: queue.Queue = queue.Queue()
        self.on_done = on_done
        self.speech_ended = False
        self.playing = False
        threading.Thread(target=self._run, daemon=True).start()

    def add(self, wav_bytes: bytes) -> None:
        self.q.put(wav_bytes)

    def end_of_speech(self) -> None:
        self.speech_ended = True
        if self.q.empty() and not self.playing:
            self._finish()

    def stop(self) -> None:
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.speech_ended = False
        self.sd.stop()

    def _finish(self) -> None:
        self.speech_ended = False
        self.on_done()

    def _run(self) -> None:
        while True:
            data = self.q.get()
            self.playing = True
            try:
                with wave.open(io.BytesIO(data)) as w:
                    sr = w.getframerate()
                    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                self.sd.play(audio, sr)
                self.sd.wait()
            except Exception as e:
                log.warning("playback failed: %s", e)
            self.playing = False
            if self.q.empty() and self.speech_ended:
                self._finish()


class LocalAudio:
    """Mac microphone -> Session -> Mac speakers. Create and start it inside the server's loop."""

    def __init__(self, make_session, window_open) -> None:
        self.loop = asyncio.get_running_loop()
        self.window_open = window_open
        self.status = {"state": "loading", "brain": "gemma"}
        self.frames: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.player = Player(self._done_speaking)
        self.session = make_session(FakeSocket(self))
        self.session.voice_barge_in = False  # no echo cancellation: only "Hey Jarvis" interrupts
        self.stream = None
        self.task: asyncio.Task | None = None

    def _done_speaking(self) -> None:
        asyncio.run_coroutine_threadsafe(self.session.on_json({"type": "playback_done"}), self.loop)

    def _callback(self, indata, n, t, status) -> None:
        frame = indata[:, 0].copy()

        def put() -> None:
            if not self.frames.full():
                self.frames.put_nowait(frame)

        self.loop.call_soon_threadsafe(put)

    async def start(self) -> None:
        import sounddevice as sd

        await self.session.send_brain()
        await self.session.set_state("idle")
        def open_mic():
            # Opening the mic can block while macOS asks for permission, so do it off the event loop.
            st = sd.InputStream(samplerate=16000, channels=1, dtype="int16",
                                blocksize=FRAME, callback=self._callback)
            st.start()
            return st

        self.stream = await self.loop.run_in_executor(None, open_mic)
        self.task = asyncio.create_task(self._pump())
        log.info("Always on: listening for 'Hey Jarvis' with no window")

    async def _pump(self) -> None:
        while True:
            frame = await self.frames.get()
            if self.window_open():
                if self.session.state != "idle":
                    self.player.stop()
                    if self.session.task:
                        self.session.task.cancel()
                    self.session.state = "idle"
                self.status["state"] = "window"
                continue
            if self.status["state"] == "window":
                self.status["state"] = self.session.state
            try:
                await self.session.on_frame(frame)
            except Exception:
                log.exception("local audio frame failed")

    def stop(self) -> None:
        if self.task:
            self.task.cancel()
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.player.stop()
        if self.session.task:
            self.session.task.cancel()


# ---------------------------------------------------------------- the Mac's mic for the orb window

def input_devices() -> list[str]:
    import sounddevice as sd
    return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]


def pick_input(name: str | None) -> int | None:
    """The device index for `name`, else the default input, unless that is a poor Bluetooth one
    (8 kHz headsets and speakers make Whisper struggle): then the Mac's own microphone."""
    import sounddevice as sd
    devs = list(sd.query_devices())
    ins = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    if name:
        for i, d in ins:
            if d["name"] == name:
                return i
    try:
        default = sd.default.device[0]
        d = devs[default] if default is not None and default >= 0 else None
    except Exception:
        d = None
    if d is None or d["default_samplerate"] < 16000:
        for i, dev in ins:
            if any(w in dev["name"] for w in ("MacBook", "Built-in", "Internal")):
                return i
    return None  # PortAudio's default


class ServerMic:
    """Feeds the Mac's microphone straight into one window's Session. Used when the native window's
    web view can't open the mic itself; the window still shows the orb and plays Jarvis's voice."""

    def __init__(self, session, device: str | None) -> None:
        self.session = session
        self.device = device
        self.loop = asyncio.get_running_loop()
        self.frames: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.stream = None
        self.task: asyncio.Task | None = None
        self.name = ""

    def _callback(self, indata, n, t, status) -> None:
        frame = indata[:, 0].copy()

        def put() -> None:
            if not self.frames.full():
                self.frames.put_nowait(frame)

        self.loop.call_soon_threadsafe(put)

    async def start(self) -> str:
        import sounddevice as sd

        idx = pick_input(self.device)
        self.name = sd.query_devices(idx if idx is not None else sd.default.device[0])["name"]

        def open_mic():
            st = sd.InputStream(device=idx, samplerate=16000, channels=1, dtype="int16",
                                blocksize=FRAME, callback=self._callback)
            st.start()
            return st

        self.stream = await asyncio.wait_for(self.loop.run_in_executor(None, open_mic), 60)
        self.task = asyncio.create_task(self._pump())
        log.info("Window is using the Mac's microphone directly: %s", self.name)
        return self.name

    async def _pump(self) -> None:
        n = 0
        while True:
            frame = await self.frames.get()
            n += 1
            if n % 3 == 0:
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) / 32768
                try:
                    await self.session.send(type="mic_level", level=round(min(1.0, rms * 10), 3))
                except Exception:
                    return
            try:
                await self.session.on_frame(frame)
            except Exception:
                log.exception("server mic frame failed")

    def stop(self) -> None:
        if self.task:
            self.task.cancel()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None


# ---------------------------------------------------------------- start at login

AGENT = Path.home() / "Library/LaunchAgents/local.jarvis.assistant.plist"


def app_path() -> str:
    for p in (Path.home() / "Applications/Jarvis.app", Path("/Applications/Jarvis.app")):
        if p.exists():
            return str(p)
    return str(Path.home() / "Applications/Jarvis.app")


def set_start_at_login(on: bool) -> None:
    if on:
        AGENT.parent.mkdir(parents=True, exist_ok=True)
        AGENT.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.jarvis.assistant</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/open</string><string>-g</string><string>-a</string><string>{app_path()}</string>
    <string>--args</string><string>--background</string>
  </array>
  <key>RunAtLoad</key><true/>
</dict></plist>
""")
    elif AGENT.exists():
        AGENT.unlink()


def start_menu_bar(root: Path):
    """Launch the menu bar icon as its own process (it needs a main thread of its own)."""
    py = root / ".venv/bin/python"
    exe = str(py) if py.exists() else sys.executable
    return subprocess.Popen([exe, str(root / "background.py"), "menubar"], cwd=root,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------------------------------------------------------------- menu bar process

API = "http://127.0.0.1:8765/api"
ICONS = {"idle": "◉", "listening": "◎", "thinking": "◌", "talking": "◉", "loading": "○", "window": "◉"}
LABELS = {"idle": "Say “Hey Jarvis”", "listening": "Listening…", "thinking": "Thinking…",
          "talking": "Speaking…", "loading": "Waking up…", "window": "Orb window is open"}


def run_menu_bar() -> None:
    import httpx
    import rumps

    def call(path: str, body: dict | None = None) -> dict | None:
        try:
            if body is None:
                return httpx.get(f"{API}/{path}", timeout=1).json()
            return httpx.post(f"{API}/{path}", json=body, timeout=2).json()
        except Exception:
            return None

    class JarvisBar(rumps.App):
        def __init__(self) -> None:
            icon = Path(__file__).with_name("menubar-icon.png")
            if icon.exists():
                super().__init__("Jarvis", icon=str(icon), template=False, quit_button=None)
            else:
                super().__init__("Jarvis", title="○", quit_button=None)
            self.has_icon = icon.exists()
            self.misses = 0
            self.status_item = rumps.MenuItem("Waking up…")
            self.gemma = rumps.MenuItem("Brain: Gemma (on this Mac)", callback=lambda _: self.cmd("set_brain", brain="gemma"))
            self.claude = rumps.MenuItem("Brain: Claude", callback=lambda _: self.cmd("set_brain", brain="claude"))
            self.menu = [
                self.status_item,
                None,
                rumps.MenuItem("Open the orb", callback=lambda _: subprocess.Popen(["open", "-a", app_path()])),
                rumps.MenuItem("Talk now", callback=lambda _: self.cmd("ptt")),
                None,
                self.gemma,
                self.claude,
                None,
                rumps.MenuItem("Turn off Always on", callback=self.turn_off),
                rumps.MenuItem("Quit Jarvis", callback=self.quit_all),
            ]

        def cmd(self, kind: str, **extra) -> None:
            call("cmd", {"type": kind, **extra})

        def turn_off(self, _) -> None:
            call("always_on", {"on": False})
            rumps.quit_application()

        def quit_all(self, _) -> None:
            call("quit", {})
            rumps.quit_application()

        @rumps.timer(0.5)
        def refresh(self, _) -> None:
            st = call("status")
            if not st or not st.get("always_on"):
                self.misses += 1
                if self.misses > 10:  # server gone or Always on switched off
                    rumps.quit_application()
                return
            self.misses = 0
            state = st.get("state", "idle")
            if self.has_icon:
                self.title = {"listening": "●", "thinking": "…", "talking": "♪"}.get(state, None)
            else:
                self.title = ICONS.get(state, "◉")
            self.status_item.title = LABELS.get(state, state)
            self.gemma.state = st.get("brain") == "gemma"
            self.claude.state = st.get("brain") == "claude"

    JarvisBar().run()


if __name__ == "__main__" and sys.argv[1:] == ["menubar"]:
    run_menu_bar()
