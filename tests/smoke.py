"""Jarvis smoke test: drives a running Jarvis over its WebSocket with typed text.

Usage:  .venv/bin/python tests/smoke.py   (Jarvis must be running on 127.0.0.1:8765)
Avoids side effects: it doesn't change volume, open apps, create reminders or clear history,
and it removes the test fact it remembers.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request

import websockets

URL = "ws://127.0.0.1:8765/ws"
results: list[tuple[str, bool, str, float]] = []


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.brain = {}
        self.history = None
        self.state = None

    async def recv(self, timeout=60):
        m = await asyncio.wait_for(self.ws.recv(), timeout)
        if isinstance(m, bytes):
            return {"type": "_audio", "bytes": len(m)}
        m = json.loads(m)
        if m["type"] == "brain":
            self.brain = m
        elif m["type"] == "history":
            self.history = m["turns"]
        elif m["type"] == "state":
            self.state = m["state"]
        elif m["type"] == "speech_end":
            # act like the window: tell Jarvis the audio finished playing
            await self.ws.send(json.dumps({"type": "playback_done"}))
        return m

    async def until_idle(self, timeout=150):
        end = time.monotonic() + timeout
        while self.state not in ("idle", "listening"):
            await self.recv(max(1, end - time.monotonic()))

    async def ask(self, text, timeout=120):
        """Send text; return (reply, seconds to first words, seconds to end, audio bytes, events)."""
        await self.until_idle()
        t0 = time.monotonic()
        await self.ws.send(json.dumps({"type": "text", "text": text}))
        reply, first, audio, events = "", None, 0, []
        while True:
            m = await self.recv(timeout)
            events.append(m["type"])
            if m["type"] == "assistant_delta":
                reply += m["text"]
                first = first or time.monotonic() - t0
            elif m["type"] == "_audio":
                audio += m["bytes"]
            elif m["type"] == "error":
                reply += f" [error: {m.get('message')}]"
            elif m["type"] == "speech_end":
                end = time.monotonic() - t0
                while self.state != "listening":
                    await self.recv(10)
                return reply.strip(), first or end, end, audio, events
            elif m["type"] == "state" and m["state"] == "idle" and "assistant_start" not in events and len(events) > 2:
                return reply.strip(), 0, time.monotonic() - t0, audio, events


def check(name, ok, detail="", secs=0.0):
    results.append((name, bool(ok), detail, secs))
    print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {secs:5.1f}s  {detail[:160]}", flush=True)


async def main():
    try:
        st = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=3).read())
        check("server /api/status", True, json.dumps(st))
    except Exception as e:
        check("server /api/status", False, str(e))
        return

    async with websockets.connect(URL, max_size=None) as ws:
        c = Client(ws)
        await c.until_idle(180)
        check("hello + brain info", bool(c.brain), f"brain={c.brain.get('brain')} voice={c.brain.get('voice')} key={c.brain.get('has_key')}")
        check("history sent on connect", c.history is not None, f"{len(c.history or [])} saved messages")

        async def q(name, text, want=None, reject=("error", "sorry, that didn't work"), timeout=120):
            try:
                reply, first, end, audio, _ = await c.ask(text, timeout)
            except Exception as e:
                check(name, False, f"{type(e).__name__}: {e}")
                return ""
            low = reply.lower()
            ok = bool(reply) and not any(r in low for r in reject)
            if want:
                ok = ok and any(w in low for w in want)
            check(name, ok, f"[{first:.1f}s first] {reply}  (audio {audio // 1024} KB)", end)
            return reply

        await q("time", "what time is it", ["it's"])
        await q("date", "what's the date today", ["it's"])
        await q("weather", "what's the weather", ["degrees"])
        await q("weather tomorrow", "what's the weather tomorrow", ["tomorrow"])
        await q("news", "what's the news", ["headlines"])
        await q("web search (LLM)", "search the web for the tallest building in the world")
        await q("memory: remember", "remember that my test colour is turquoise", ["remember"])
        await q("memory: recall (LLM)", "what is my test colour", ["turquoise"])
        await q("memory: forget", "forget test colour", ["forgotten"])
        await q("timer", "set a timer for 3 seconds", ["timer", "seconds"])
        # the timer announcement arrives on its own a few seconds later
        try:
            got = False
            end = time.monotonic() + 15
            while time.monotonic() < end and not got:
                m = await c.recv(max(0.5, end - time.monotonic()))
                if m["type"] == "assistant_delta" and "timer" in m["text"].lower():
                    got = True
            if got:  # let the announcement finish before the next question
                while c.state != "listening":
                    await c.recv(20)
            check("timer announcement", got, "spoke when the timer finished" if got else "no announcement within 15 s")
        except Exception as e:
            check("timer announcement", False, str(e))
        await q("calendar", "what's on my calendar today",
                reject=("error", "sorry, that didn't work", "not authori"))
        await q("mail", "do I have any new emails", reject=("error", "sorry, that didn't work", "not authori"))
        await q("briefing", "good morning", ["good"])
        await q("file finder", "where is the file called requirements", ["found", "couldn't find"])
        await q("screen", "what's on my screen")
        await q("LLM chat (brain)", "tell me a one line joke")

        # personality round trip (restores the original)
        orig = c.brain.get("personality", "butler")
        await c.until_idle()
        await ws.send(json.dumps({"type": "set_personality", "personality": "brief"}))
        t0 = time.monotonic()
        while c.brain.get("personality") != "brief" and time.monotonic() - t0 < 10:
            await c.recv(10)
        check("personality switch", c.brain.get("personality") == "brief", "brief")
        await c.until_idle()
        await ws.send(json.dumps({"type": "set_personality", "personality": orig}))
        while c.brain.get("personality") != orig:
            await c.recv(10)
        await c.until_idle()

        # brain switch: Claude without a key must fall back cleanly
        if not c.brain.get("has_key"):
            await ws.send(json.dumps({"type": "set_brain", "brain": "claude"}))
            saw = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5:
                try:
                    m = await c.recv(1)
                except asyncio.TimeoutError:
                    break
                if m["type"] == "need_key":
                    saw = "need_key"
                    break
            check("Claude without key", saw == "need_key" and c.brain.get("brain") == "gemma", f"asked for a key, stayed on {c.brain.get('brain')}")
        else:
            await q("voice switch to Claude", "switch to Claude", ["claude"])
            await q("Claude answers", "in one sentence, what are you?")
            await q("voice switch to Gemma", "switch to Gemma", ["gemma"])

        # interrupting: ask for something long, then press to stop while talking
        await c.until_idle()
        await ws.send(json.dumps({"type": "text", "text": "tell me a long story about a dragon"}))
        stopped, t0 = False, time.monotonic()
        while time.monotonic() - t0 < 90:
            m = await c.recv(90)
            if m["type"] == "state" and m["state"] == "talking":
                await asyncio.sleep(1.0)
                await ws.send(json.dumps({"type": "ptt"}))
            if m["type"] == "stop_audio":
                stopped = True
                break
        check("interrupt while talking", stopped, "stopped speaking when interrupted" if stopped else "didn't stop", time.monotonic() - t0)

    ok = sum(1 for r in results if r[1])
    print(f"\n{ok}/{len(results)} passed")
    sys.exit(0 if ok == len(results) else 1)


asyncio.run(main())
