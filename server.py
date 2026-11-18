"""Jarvis: a local voice assistant.

Browser (mic + orb) <-> WebSocket <-> this server:
  wake word (openWakeWord "hey jarvis") -> speech-to-text (faster-whisper)
  -> LLM (Ollama, streamed) -> text-to-speech (Piper, sentence by sentence) -> browser.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

OLLAMA_URL = os.environ.get("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("JARVIS_LLM", "gemma4:e4b")
# Small, quick model for everyday chat; the big one handles anything that needs real thought.
FAST_MODEL = os.environ.get("JARVIS_FAST_LLM", "gemma3:1b")
FAST_AVAILABLE = False  # set once Ollama confirms the small model is installed
# Only chit-chat goes to the small model; it gets facts wrong (it said Sydney is Australia's capital).
SMALLTALK_RE = re.compile(r"^(?:hi|hello|hey|yo|good (?:morning|afternoon|evening|night)|thanks?(?: you)?|cheers|"
                          r"how are you|how's it going|what's up|tell me (?:a |another )?(?:[a-z-]+ ){0,3}(?:joke|fun fact)|joke|"
                          r"who are you|what(?:'s| is) your name|i'm (?:bored|tired|back|home)|"
                          r"you're (?:funny|great|awesome|the best)|good (?:job|work)|nice|cool|ok(?:ay)?)\b", re.I)
HARD_RE = re.compile(r"\b(why|explain|how (?:does|do|would|can|could)|compare|difference|analy[sz]e|summari[sz]e|"
                     r"write|draft|plan|calculate|translate|code|step[s ]?by[ -]step|in detail|pros and cons)\b", re.I)


def pick_model(messages: list[dict]) -> str:
    """Quick small model for short everyday questions, Gemma for anything harder or with extra facts."""
    if not FAST_AVAILABLE or load_settings().get("fast_answers", True) is False:
        return LLM_MODEL
    if messages and messages[-1].get("images"):
        return LLM_MODEL  # the small model can't see
    last = (messages[-1]["content"] if messages else "").strip()
    if len(last) > 120 or "\n" in last or HARD_RE.search(last) or not SMALLTALK_RE.match(last):
        return LLM_MODEL
    return FAST_MODEL
WHISPER_MODEL = os.environ.get("JARVIS_WHISPER", "base.en")
PIPER_VOICE = os.environ.get("JARVIS_VOICE", "en_GB-alan-medium")
WAKE_THRESHOLD = float(os.environ.get("JARVIS_WAKE_THRESHOLD", "0.5"))
USER_NAME = os.environ.get("JARVIS_USER_NAME", "sir")
CLAUDE_MODEL = os.environ.get("JARVIS_CLAUDE_MODEL", "claude-opus-5-5")
KEYCHAIN_SERVICE = "Jarvis (Anthropic API key)"
SETTINGS_FILE = ROOT / "settings.json"

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms, the frame size openWakeWord expects
FOLLOW_UP_SECONDS = 6.0  # after Jarvis speaks, keep listening this long without the wake word
MAX_UTTERANCE_SECONDS = 15.0
END_SILENCE_SECONDS = 0.8
HISTORY_TURNS = 6
# Interrupting Jarvis while it talks: "wake" = only "Hey Jarvis", "voice" = also sustained loud speech.
BARGE_IN = os.environ.get("JARVIS_BARGE_IN", "voice")
# Must match between warm-up and chat, or Ollama reloads the model on the first question.
LLM_CTX = int(os.environ.get("JARVIS_LLM_CTX", "2048"))

PERSONALITIES = {
    "butler": ("Classic butler", "You are J.A.R.V.I.S., a calm, witty and very capable personal assistant with dry British humour."),
    "friendly": ("Friendly", "You are Jarvis, a warm, upbeat and encouraging personal assistant who talks like a good friend."),
    "brief": ("Straight to the point", "You are Jarvis, a no-nonsense assistant. Give the shortest useful answer, with no small talk."),
    "sarcastic": ("Cheeky", "You are J.A.R.V.I.S., a brilliant assistant with a cheeky, sarcastic streak who still always helps."),
}


def system_prompt(brain: str) -> str:
    where = (
        "Your voice and ears run on the user's own computer; your thinking is powered by Claude."
        if brain == "claude"
        else "You run entirely on the user's own computer, powered by Gemma."
    )
    return (
        f"{PERSONALITIES.get(load_settings().get('personality', 'butler'), PERSONALITIES['butler'])[1]} {where} "
        f"Address the user as '{skills.preferred_name() or USER_NAME}'. Your replies are spoken aloud, so answer in one to three "
        "short conversational sentences. Never use markdown, lists, emoji or code blocks. "
        "If you don't know something, say so briefly. The user can switch your brain by saying "
        "'switch to Claude' or 'switch to Gemma'."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("jarvis")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------- engines

class Engines:
    def __init__(self) -> None:
        self.pipers: dict = {}
        self.wake = None
        self.stt = None
        self.piper = None
        self.kokoro = None
        self.voice = "piper"  # "piper" (fast) or a Kokoro voice like "bm_george"

    def load_kokoro(self) -> None:
        """Kokoro: more natural British voices. ~120 MB, downloaded the first time it's chosen."""
        import urllib.request

        from kokoro_onnx import Kokoro

        folder = MODELS / "kokoro"
        folder.mkdir(parents=True, exist_ok=True)
        base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        for name in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"):
            if not (folder / name).exists():
                log.info("Downloading %s", name)
                urllib.request.urlretrieve(base + name, folder / (name + ".part"))
                (folder / (name + ".part")).rename(folder / name)
        self.kokoro = Kokoro(str(folder / "kokoro-v1.0.int8.onnx"), str(folder / "voices-v1.0.bin"))

    def set_voice(self, voice: str) -> None:
        if voice.startswith("piper:"):
            name = voice.split(":", 1)[1]
            if name not in self.pipers:
                from piper import PiperVoice
                self.pipers[name] = PiperVoice.load(str(download_piper_voice(name)))
        elif voice != "piper" and self.kokoro is None:
            self.load_kokoro()
        self.voice = voice

    def load(self) -> None:
        MODELS.mkdir(exist_ok=True)

        log.info("Loading wake word model (hey jarvis)…")
        import openwakeword
        from openwakeword.model import Model as WakeModel

        try:
            openwakeword.utils.download_models(model_names=["hey_jarvis"])
        except Exception as e:  # already downloaded, or offline
            log.debug("wake model download skipped: %s", e)
        self.wake = WakeModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")

        log.info("Loading speech-to-text (faster-whisper %s)…", WHISPER_MODEL)
        from faster_whisper import WhisperModel

        self.stt = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8", download_root=str(MODELS / "whisper")
        )

        log.info("Loading voice (Piper %s)…", PIPER_VOICE)
        try:
            from piper import PiperVoice

            onnx = download_piper_voice(PIPER_VOICE)
            self.piper = PiperVoice.load(str(onnx))
        except Exception as e:
            log.warning("Piper unavailable (%s); falling back to macOS 'say'.", e)
            self.piper = None
        voice = load_settings().get("voice", "piper")
        if voice != "piper" and voice in VOICES:
            try:
                self.set_voice(voice)
            except Exception as e:
                log.warning("Kokoro voice unavailable (%s); using Piper.", e)
        log.info("All engines ready.")

    # -- wake word
    def wake_score(self, frame: np.ndarray) -> float:
        scores = self.wake.predict(frame)
        return max(scores.values()) if scores else 0.0

    def wake_reset(self) -> None:
        self.wake.reset()

    # -- speech to text
    def transcribe(self, audio: np.ndarray) -> str:
        pcm = audio.astype(np.float32) / 32768.0
        segments, _ = self.stt.transcribe(
            pcm, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False
        )
        return " ".join(s.text.strip() for s in segments).strip()

    # -- text to speech, returns WAV bytes
    def speak(self, text: str) -> bytes:
        extra = self.pipers.get(self.voice.split(":", 1)[1]) if self.voice.startswith("piper:") else None
        if extra is not None:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                extra.synthesize_wav(text, wav)
            return buf.getvalue()
        if not self.voice.startswith("piper") and self.kokoro is not None:
            samples, sr = self.kokoro.create(text, voice=self.voice, speed=1.0,
                                                lang="en-us" if self.voice.startswith("a") else "en-gb")
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
            return buf.getvalue()
        if self.piper is not None:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                self.piper.synthesize_wav(text, wav)
            return buf.getvalue()
        return self._say(text)

    @staticmethod
    def _say(text: str) -> bytes:
        if not shutil.which("say"):
            return b""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "s.wav"
            subprocess.run(
                ["say", "-o", str(out), "--data-format=LEI16@22050", text], check=True
            )
            return out.read_bytes()


def download_piper_voice(name: str) -> Path:
    """Fetch a Piper voice (e.g. en_GB-alan-medium) from Hugging Face once."""
    import urllib.request

    folder = MODELS / "piper"
    folder.mkdir(parents=True, exist_ok=True)
    onnx = folder / f"{name}.onnx"
    lang, speaker, quality = name.split("-", 2)
    base = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang.split('_')[0]}/{lang}/{speaker}/{quality}/{name}"
    )
    for suffix in (".onnx", ".onnx.json"):
        target = folder / f"{name}{suffix}"
        if not target.exists():
            log.info("Downloading %s", target.name)
            urllib.request.urlretrieve(base + suffix, target)
    return onnx


engines = Engines()
SESSIONS: set = set()


async def announce(text: str) -> bool:
    """Speak something unprompted (e.g. a timer) in every open Jarvis window. True if anyone heard it."""
    spoke = False
    for s in list(SESSIONS):
        if s.state in ("idle", "listening"):
            await s.speak_only(text)
            spoke = True
    return spoke


from skills import Skills  # noqa: E402

skills = Skills(ROOT, announce, allowed=lambda name: bool(load_settings().get(f"allow_{name}")))
skills.settings = lambda: load_settings()


# ---------------------------------------------------------------- helpers

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def clean_for_speech(text: str) -> str:
    text = re.sub(r"[*_#`>~]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


async def ollama_stream(messages: list[dict]):
    model = pick_model(messages)
    log.info("Thinking with %s", model)
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "2h",
        "options": {"temperature": 0.6, "num_ctx": LLM_CTX},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5)) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="ignore")
                raise RuntimeError(f"Ollama error {r.status_code}: {body[:300]}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break


# ---------------------------------------------------------------- brains: Gemma (local) or Claude (API)

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_settings(**changes) -> None:
    data = load_settings()
    data.update(changes)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2))


def get_api_key() -> str | None:
    """The Anthropic API key lives in the macOS Keychain (or ANTHROPIC_API_KEY), never in a file."""
    try:
        import keyring

        key = keyring.get_password(KEYCHAIN_SERVICE, "default")
        if key:
            return key
    except Exception as e:
        log.debug("keychain read failed: %s", e)
    return os.environ.get("ANTHROPIC_API_KEY") or None


def set_api_key(key: str | None) -> None:
    import keyring

    if key:
        keyring.set_password(KEYCHAIN_SERVICE, "default", key)
    else:
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, "default")
        except Exception:
            pass


class NoClaudeKey(Exception):
    pass


async def claude_stream(messages: list[dict]):
    """Stream a reply from Claude. Raises NoClaudeKey / anthropic errors for the caller to fall back."""
    key = get_api_key()
    if not key:
        raise NoClaudeKey()
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=key, max_retries=1, timeout=60)
    system = messages[0]["content"]
    convo = []
    for m in messages[1:]:
        if m["role"] not in ("user", "assistant"):
            continue
        if m.get("images"):
            convo.append({"role": m["role"], "content": [
                *({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}} for img in m["images"]),
                {"type": "text", "text": m["content"]}]})
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    async with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,  # replies are spoken, so they are deliberately short
        system=system,
        messages=convo,
        output_config={"effort": "low"},  # low effort keeps voice replies fast
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "I'm afraid I can't help with that one."


# Voice id -> label. "piper…" voices are fast on any Mac; Kokoro ones sound more natural but are slow on Intel.
# Kokoro ids: first letter b = British, a = American; second f = female, m = male.
VOICES = {
    "piper": "Alan · British ⚡",
    "piper:en_GB-northern_english_male-medium": "Northern · British ⚡",
    "piper:en_US-ryan-medium": "Ryan · American ⚡",
    "piper:en_GB-jenny_dioco-medium": "Jenny · British ⚡",
    "piper:en_GB-alba-medium": "Alba · Scottish ⚡",
    "piper:en_US-amy-medium": "Amy · American ⚡",
    "piper:en_US-lessac-medium": "Lessac · American ⚡",
    "bm_george": "George · British", "bm_daniel": "Daniel · British",
    "bm_fable": "Fable · British", "bm_lewis": "Lewis · British",
    "am_michael": "Michael · American", "am_adam": "Adam · American",
    "am_fenrir": "Fenrir · American", "am_onyx": "Onyx · American",
    "bf_emma": "Emma · British", "bf_isabella": "Isabella · British",
    "bf_alice": "Alice · British", "bf_lily": "Lily · British",
    "af_heart": "Heart · American", "af_bella": "Bella · American",
    "af_nicole": "Nicole · American", "af_sarah": "Sarah · American",
}

VOICE_GENDER = {"piper": "m", "piper:en_GB-northern_english_male-medium": "m", "piper:en_US-ryan-medium": "m",
                "piper:en_GB-jenny_dioco-medium": "f", "piper:en_GB-alba-medium": "f",
                "piper:en_US-amy-medium": "f", "piper:en_US-lessac-medium": "f"}
VOICE_GENDER.update({k: k[1] for k in VOICES if not k.startswith("piper")})

SWITCH_RE = re.compile(
    r"\b(switch|change|swap|go|use|talk)\b.{0,25}?\b(claude|cloud|gemma|gemini|local|offline)\b", re.I
)


def parse_switch(text: str) -> str | None:
    m = SWITCH_RE.search(text)
    if not m:
        return None
    return "claude" if m.group(2).lower() in ("claude", "cloud") else "gemma"


# ---------------------------------------------------------------- session

# Things Whisper "hears" in near-silence: ignore them rather than answering.
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "i'm so glad that you're here",
    "you", "bye", "okay", "so", "um", "uh", "hmm", "the end", "please subscribe",
    "subtitles by the amara.org community", "i'm going to go ahead and do that",
}


def real_speech(text: str, audio: np.ndarray) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return False  # only dots or punctuation
    phrase = " ".join(words)
    if phrase in HALLUCINATIONS:
        return False
    # The same short phrase repeated ("I'm so glad... I'm so glad...")
    sentences = [x.strip() for x in re.split(r"[.!?]+", text.lower()) if x.strip()]
    if len(sentences) >= 2 and len(set(sentences)) == 1:
        return False
    # Too quiet overall to be someone talking to Jarvis
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))) if audio.size else 0.0
    if rms < 80:
        return False
    return True


class Session:
    """One browser connection: a small state machine driven by mic frames."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.state = "idle"
        self.utterance: list[np.ndarray] = []
        self.heard_speech = False
        self.silence = 0.0
        self.listen_started = 0.0
        self.follow_up_until = 0.0
        self.noise_floor = 300.0
        self.loud_frames = 0
        self.interrupted = False
        self.recent: collections.deque = collections.deque(maxlen=8)
        self.task: asyncio.Task | None = None
        self.loop = asyncio.get_running_loop()
        self.brain = load_settings().get("brain", "gemma")
        self.history = skills.memory.recent(4)  # carry on from last time
        self.voice_barge_in = BARGE_IN == "voice"

    async def send_brain(self) -> None:
        await self.send(
            type="brain",
            brain=self.brain,
            has_key=bool(get_api_key()),
            local_model=LLM_MODEL,
            claude_model=CLAUDE_MODEL,
            voice=engines.voice,
            allow_calendar=bool(load_settings().get("allow_calendar")),
            allow_mail=bool(load_settings().get("allow_mail")),
            always_on=bool(load_settings().get("always_on")),
            fast_available=FAST_AVAILABLE, fast_model=FAST_MODEL,
            fast_answers=load_settings().get("fast_answers", True) is not False,
            personality=load_settings().get("personality", "butler"),
            personalities={k: v[0] for k, v in PERSONALITIES.items()},
            voices=VOICES, voice_gender=VOICE_GENDER,
        )

    async def switch_brain(self, brain: str, announce: bool) -> None:
        if brain == "claude" and not get_api_key():
            await self.send_brain()
            await self.send(type="need_key")
            if announce:
                await self.speak_only(
                    "I'd need an Anthropic API key for that. Add one in settings and I'll switch over."
                )
            return
        save_settings(brain=brain)
        for s in list(SESSIONS) + [self]:
            s.brain = brain
            if s is not self:
                await s.send_brain()
        await self.send_brain()
        if announce:
            name = "Claude" if brain == "claude" else "Gemma, running locally"
            await self.speak_only(f"Switching my brain to {name}.")

    async def wait_until_spoken(self, limit: float = 8.0) -> None:
        """Wait for the window to finish playing what was just said (it sends playback_done)."""
        self.spoken = asyncio.Event()
        try:
            await asyncio.wait_for(self.spoken.wait(), limit)
        except asyncio.TimeoutError:
            pass
        self.no_follow_up = False

    async def camera_frame(self) -> str | None:
        """Ask the window for one webcam photo (base64 JPEG)."""
        self.frame_wait = self.loop.create_future()
        await self.send(type="camera_snap")
        try:
            data = await asyncio.wait_for(self.frame_wait, 10)
        except asyncio.TimeoutError:
            data = None
        return data or None

    async def use_server_mic(self, device: str | None) -> None:
        """The window can't reach the mic itself (native window): listen through the Mac directly."""
        import background

        old = getattr(self, "server_mic", None)
        if old:
            old.stop()
        if device:
            save_settings(server_mic_device=device)
        mic = background.ServerMic(self, device or load_settings().get("server_mic_device"))
        try:
            name = await mic.start()
        except Exception as e:
            log.warning("server mic failed: %s", e)
            await self.send(type="server_mic", ok=False, error=str(e))
            return
        self.server_mic = mic
        self.voice_barge_in = False  # no echo cancellation here: only "Hey Jarvis" interrupts
        await self.send(type="server_mic", ok=True, device=name)

    async def speak_only(self, text: str) -> None:
        self.interrupted = False
        await self.send(type="assistant_start")
        await self.send(type="assistant_delta", text=text)
        await self.set_state("talking")
        await self.say(text)
        await self.send(type="speech_end")

    async def send(self, **msg) -> None:
        await self.ws.send_text(json.dumps(msg))

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.send(type="state", state=state)

    def start_listening(self, follow_up: bool = False) -> None:
        self.utterance = []
        self.heard_speech = False
        self.silence = 0.0
        self.listen_started = time.monotonic()
        self.follow_up_until = self.listen_started + FOLLOW_UP_SECONDS if follow_up else 0.0

    async def on_frame(self, frame: np.ndarray) -> None:
        if self.state == "thinking":
            return
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

        if self.state == "talking":
            # Barge-in: the browser's echo cancellation removes most of Jarvis's own voice,
            # so "Hey Jarvis" or someone clearly talking over it stops the reply.
            self.recent.append(frame)
            score = await self.loop.run_in_executor(None, engines.wake_score, frame)
            loud = rms > max(self.noise_floor * 4, 1200)
            self.loud_frames = self.loud_frames + 1 if loud else 0
            if score >= WAKE_THRESHOLD or (self.voice_barge_in and self.loud_frames >= 5):
                log.info("Interrupted (wake %.2f, loud frames %d)", score, self.loud_frames)
                await self.interrupt()
            return

        if self.state == "idle":
            # Track background noise so the end-of-speech detector adapts to the room.
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * min(rms, 3000)
            score = await self.loop.run_in_executor(None, engines.wake_score, frame)
            if score >= WAKE_THRESHOLD:
                log.info("Wake word (%.2f)", score)
                engines.wake_reset()
                self.start_listening()
                await self.set_state("listening")
            return

        # listening
        self.utterance.append(frame)
        speech_level = max(self.noise_floor * 2.5, 500)
        now = time.monotonic()
        if rms > speech_level:
            self.heard_speech = True
            self.silence = 0.0
        else:
            self.silence += FRAME / SAMPLE_RATE

        too_long = now - self.listen_started > MAX_UTTERANCE_SECONDS
        if self.heard_speech and (self.silence >= END_SILENCE_SECONDS or too_long):
            audio = np.concatenate(self.utterance)
            self.state = "thinking"  # stop collecting frames right away
            self.task = asyncio.create_task(self.handle_audio(audio))
        elif not self.heard_speech:
            deadline = self.follow_up_until or (self.listen_started + 5.0)
            if now > deadline:
                await self.set_state("idle")

    async def handle_audio(self, audio: np.ndarray) -> None:
        await self.set_state("thinking")
        t0 = time.monotonic()
        text = await self.loop.run_in_executor(None, engines.transcribe, audio)
        log.info("Heard (%.2fs): %r", time.monotonic() - t0, text)
        if not real_speech(text, audio):
            log.info("Ignored (noise or a Whisper hallucination)")
            await self.set_state("idle")
            return
        await self.respond(text)

    async def respond(self, text: str) -> None:
        self.interrupted = False
        await self.send(type="user", text=text)
        await self.set_state("thinking")
        target = parse_switch(text) if len(text.split()) <= 10 else None
        if target:
            await self.switch_brain(target, announce=True)
            return

        result = await skills.handle(text)
        if result and result.reply is not None:
            self.no_follow_up = not result.follow_up
            if result.reply:
                await self.speak_only(result.reply)
            else:  # silent actions (mute, pause music)
                await self.set_state("talking")
                await self.send(type="speech_end")
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": result.reply or "Done."}]
            skills.memory.add_turns(text, result.reply or "Done.")
            return

        image = None
        if result and result.camera:
            if self.brain != "claude":
                await self.speak_only("Let me have a look. Give me a moment.")
                await self.wait_until_spoken()
            image = await self.camera_frame()
            if not image:
                msg = "I can't see through the camera right now. Check that Jarvis is allowed to use it."
                await self.speak_only(msg)
                return
        system = system_prompt(self.brain) + " " + skills.system_notes()
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": system}] + self.history[-HISTORY_TURNS * 2 :]
        if image:
            messages[-1] = {"role": "user", "images": [image], "content": (
                "This is a photo just taken by the user's webcam, showing them and what's in front of them. "
                "Answer briefly and naturally, as if you can see it yourself: " + text)}
        if result and result.context:
            # Put fresh facts right next to the question (small local models ignore them in the system prompt).
            messages[-1] = {"role": "user", "content": (
                f"{result.context}\n\nUsing the information above, answer this directly and specifically "
                f"(give the actual names, numbers or facts; don't say you'll check): {text}")}

        reply, pending = "", ""
        speaking = False
        t0 = time.monotonic()
        await self.send(type="assistant_start")
        try:
            async for chunk in self.brain_stream(messages):
                reply += chunk
                pending += chunk
                await self.send(type="assistant_delta", text=chunk)
                parts = SENTENCE_END.split(pending)
                for sentence in parts[:-1]:
                    if not speaking:
                        log.info("First sentence after %.2fs", time.monotonic() - t0)
                        speaking = True
                        await self.set_state("talking")
                    await self.say(sentence)
                pending = parts[-1]
            if pending.strip():
                if not speaking:
                    await self.set_state("talking")
                await self.say(pending)
        except httpx.ConnectError:
            msg = "I can't reach Ollama. Is the Ollama app running?"
            await self.send(type="error", message=msg)
            await self.say(msg)
        except Exception as e:
            log.exception("LLM failed")
            await self.send(type="error", message=str(e))
        finally:
            if reply:
                self.history.append({"role": "assistant", "content": reply})
                skills.memory.add_turns(text, reply)
            if not self.interrupted:
                if self.state != "talking":
                    await self.set_state("talking")
                await self.send(type="speech_end")

    async def interrupt(self) -> None:
        """Stop talking right away and listen to the user."""
        self.interrupted = True
        spoke = self.loud_frames >= 5
        self.loud_frames = 0
        if self.task and not self.task.done():
            self.task.cancel()
        await self.send(type="stop_audio")
        engines.wake_reset()
        self.start_listening(follow_up=True)
        if spoke:  # keep the start of what they said
            self.utterance = list(self.recent)
            self.heard_speech = True
        self.recent.clear()
        await self.set_state("listening")

    async def brain_stream(self, messages: list[dict]):
        """Yield reply text from the chosen brain; fall back to Gemma if Claude isn't available."""
        if self.brain == "claude":
            import anthropic

            started = False
            try:
                async for chunk in claude_stream(messages):
                    started = True
                    yield chunk
                return
            except NoClaudeKey:
                note = "No Claude API key is set, so Gemma is answering."
            except anthropic.AuthenticationError:
                note = "Claude rejected the API key, so Gemma is answering. Check the key in settings."
            except (anthropic.APIConnectionError, anthropic.APITimeoutError):
                note = "Can't reach Claude (offline?), so Gemma is answering."
            except anthropic.RateLimitError:
                note = "Claude is rate-limited right now, so Gemma is answering."
            except anthropic.APIStatusError as e:
                note = f"Claude returned an error ({e.status_code}), so Gemma is answering."
            if started:
                return  # part of the reply was already spoken; don't restart it
            log.warning(note)
            await self.send(type="error", message=note)
        async for chunk in ollama_stream(messages):
            yield chunk

    async def say(self, sentence: str) -> None:
        sentence = clean_for_speech(sentence)
        if not sentence:
            return
        wav = await self.loop.run_in_executor(None, engines.speak, sentence)
        if wav:
            await self.ws.send_bytes(wav)

    async def on_json(self, msg: dict) -> None:
        kind = msg.get("type")
        spoken = getattr(self, "spoken", None)
        if kind == "playback_done" and spoken is not None and not spoken.is_set():
            # a filler line ("Let me have a look") finished; the answer is still coming
            spoken.set()
            await self.set_state("thinking")
        elif kind == "playback_done" and getattr(self, "no_follow_up", False):
            # e.g. after "Alexa, lights off": don't treat the speaker's reply as a question for Jarvis
            self.no_follow_up = False
            engines.wake_reset()
            await self.set_state("idle")
        elif kind == "playback_done":
            # Jarvis finished speaking: allow a quick follow-up without the wake word.
            engines.wake_reset()
            self.start_listening(follow_up=True)
            await self.set_state("listening")
        elif kind == "window_ready":
            asyncio.create_task(greet(self))
        elif kind == "camera_frame":
            fut = getattr(self, "frame_wait", None)
            if fut and not fut.done():
                if msg.get("error"):
                    log.warning("camera: %s", msg["error"])
                fut.set_result(msg.get("data"))
        elif kind == "server_mic":
            asyncio.create_task(self.use_server_mic(msg.get("device") or None))
        elif kind == "ptt" and self.state == "talking":
            await self.interrupt()
        elif kind == "ptt" and self.state == "idle":
            self.start_listening(follow_up=True)
            await self.set_state("listening")
        elif kind == "set_allow" and msg.get("what") in ("calendar", "mail"):
            save_settings(**{f"allow_{msg['what']}": bool(msg.get("on"))})
            await self.send_brain()
        elif kind == "set_personality" and msg.get("personality") in PERSONALITIES:
            save_settings(personality=msg["personality"])
            await self.send_brain()
            await self.speak_only({"butler": "Very good, sir. Back to my usual self.",
                                   "friendly": "Hey! This is me being friendly. How's it going?",
                                   "brief": "Understood. Short answers.",
                                   "sarcastic": "Oh wonderful, a personality change. I'm thrilled."}[msg["personality"]])
        elif kind == "hud_action":
            say = await skills.hud_action(str(msg.get("action")))
            if say and self.state in ("idle", "listening"):
                await self.speak_only(say)
        elif kind == "clear_history":
            skills.memory.data["history"] = []
            skills.memory.save()
            self.history = []
            await self.send(type="history", turns=[])
        elif kind == "set_fast_answers":
            save_settings(fast_answers=bool(msg.get("on")))
            await self.send_brain()
        elif kind == "set_always_on":
            asyncio.create_task(always_on_then_refresh(bool(msg.get("on"))))
        elif kind == "set_voice":
            voice = str(msg.get("voice"))
            if voice in VOICES:
                await self.send(type="voice_loading")
                try:
                    await self.loop.run_in_executor(None, engines.set_voice, voice)
                    save_settings(voice=voice)
                except Exception as e:
                    log.warning("voice %s failed: %s", voice, e)
                    await self.send(type="error", message=f"Couldn't load that voice: {e}")
                    await self.send_brain()
                    await self.speak_only("Sorry, I couldn't download that voice. Check the internet connection and try again.")
                    return
                await self.send_brain()
                name = VOICES[voice].split(" ·")[0].replace("Northern", "your northern voice")
                await self.speak_only(f"Hello, I'm {name}. This is how I'll sound from now on.")
        elif kind == "set_brain":
            await self.switch_brain(str(msg.get("brain")), announce=False)
        elif kind == "set_key":
            key = str(msg.get("key", "")).strip()
            try:
                await self.loop.run_in_executor(None, set_api_key, key or None)
                if not key and self.brain == "claude":
                    self.brain = "gemma"
                    save_settings(brain="gemma")
                await self.send(type="key_saved", ok=True, removed=not key)
            except Exception as e:
                await self.send(type="key_saved", ok=False, error=str(e))
            await self.send_brain()
        elif kind == "text" and self.state in ("idle", "listening"):
            text = str(msg.get("text", "")).strip()
            if text:
                self.task = asyncio.create_task(self.respond(text))


# ---------------------------------------------------------------- app

app = FastAPI()
WEB = ROOT / "web"


READY = asyncio.Event()
EXIT_WHEN_IDLE = os.environ.get("JARVIS_EXIT_WHEN_IDLE") == "1"
CLIENTS = 0


LOCAL = None        # background.LocalAudio when Always on is enabled
MENU_BAR = None     # the menu bar icon's process


def notify_mac(text: str) -> None:
    try:
        subprocess.Popen(["osascript", "-e", f'display notification "{text}" with title "Jarvis"'])
    except OSError:
        pass


async def set_always_on(on: bool) -> None:
    """Always on: "Hey Jarvis" works with no window, there's a menu bar icon, and Jarvis starts at login."""
    global LOCAL, MENU_BAR
    import background

    save_settings(always_on=on)
    try:
        background.set_start_at_login(on)
    except Exception as e:
        log.warning("Couldn't change start at login: %s", e)
    if on and LOCAL is None and sys.platform == "darwin":
        await READY.wait()
        try:
            LOCAL = background.LocalAudio(Session, lambda: CLIENTS > 0)
            SESSIONS.add(LOCAL.session)
            await asyncio.wait_for(LOCAL.start(), 30)
        except Exception as e:
            # e.g. macOS is showing "allow microphone?" and hasn't been answered
            log.warning("Couldn't start the Mac microphone for Always on: %r", e)
            if LOCAL is not None:
                SESSIONS.discard(LOCAL.session)
                try:
                    LOCAL.stop()
                except Exception:
                    pass
            LOCAL = None
            notify_mac("Jarvis couldn't use the microphone for Always on. Allow it in System Settings → Privacy & Security → Microphone, then reopen Jarvis.")
        if MENU_BAR is None or MENU_BAR.poll() is not None:
            try:
                MENU_BAR = background.start_menu_bar(ROOT)
            except Exception:
                log.exception("Couldn't start the menu bar icon")
    elif not on:
        if LOCAL is not None:
            SESSIONS.discard(LOCAL.session)
            LOCAL.stop()
            LOCAL = None
        if MENU_BAR is not None and MENU_BAR.poll() is None:
            MENU_BAR.terminate()
        MENU_BAR = None
        if EXIT_WHEN_IDLE and CLIENTS == 0:
            asyncio.create_task(exit_if_still_idle())


async def load_everything() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, engines.load)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=3)) as c:
            tags = (await c.get(f"{OLLAMA_URL}/api/tags")).json()
            names = [m["name"] for m in tags.get("models", [])]
            global FAST_AVAILABLE
            FAST_AVAILABLE = any(n == FAST_MODEL or n == FAST_MODEL + ":latest" for n in names)
            if not FAST_AVAILABLE:
                log.info("Quick-answer model %s not installed (ollama pull %s); using %s for everything.",
                         FAST_MODEL, FAST_MODEL, LLM_MODEL)
            if LLM_MODEL not in names:
                log.warning("Model %s not found in Ollama. Run: ollama pull %s", LLM_MODEL, LLM_MODEL)
            global BRAIN_OK
            BRAIN_OK = LLM_MODEL in names
            for m in ([FAST_MODEL] if FAST_AVAILABLE else []) + ([LLM_MODEL] if LLM_MODEL in names else []):
                log.info("Warming up %s…", m)
                await c.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": m, "prompt": "hi", "think": False, "stream": False,
                          "keep_alive": "2h", "options": {"num_predict": 1, "num_ctx": LLM_CTX}},
                )
    except Exception:
        log.warning("Ollama is not reachable at %s. Start the Ollama app.", OLLAMA_URL)
    SYSTEMS["brain"] = BRAIN_OK
    READY.set()
    log.info("Jarvis is ready: open http://127.0.0.1:8765")
    asyncio.create_task(proactive_loop())


SYSTEMS: dict = {}
PROACTIVE_EVERY = 60
BRAIN_OK = False
GREETED = False


def systems_check() -> list[dict]:
    """The start-up checklist the window shows, film style."""
    st = load_settings()
    voice = "Kokoro" if engines.kokoro is not None and not engines.voice.startswith("piper") else "Piper"
    return [
        {"name": "Wake word", "ok": engines.wake is not None, "detail": "Hey Jarvis"},
        {"name": "Speech recognition", "ok": engines.stt is not None, "detail": f"Whisper {WHISPER_MODEL}"},
        {"name": "Voice", "ok": True, "detail": voice},
        {"name": "Brain", "ok": SYSTEMS.get("brain", False) or st.get("brain") == "claude",
         "detail": "Claude" if st.get("brain") == "claude" else LLM_MODEL},
        {"name": "Calendar", "ok": bool(st.get("allow_calendar")), "detail": "linked" if st.get("allow_calendar") else "off"},
        {"name": "Mail", "ok": bool(st.get("allow_mail")), "detail": "linked" if st.get("allow_mail") else "off"},
        {"name": "Smart home", "ok": st.get("smart_home", "alexa") != "off", "detail": "via Alexa" if st.get("smart_home", "alexa") != "off" else "off"},
    ]


async def greet(session: "Session") -> None:
    """First window of this launch: show the systems check and say hello."""
    global GREETED
    await READY.wait()
    checks = systems_check()
    await session.send(type="systems", checks=checks)
    if GREETED or load_settings().get("startup_greeting", True) is False:
        return
    GREETED = True
    core = {c["name"]: c["ok"] for c in checks if c["name"] in ("Wake word", "Speech recognition", "Voice", "Brain")}
    text = await skills.startup_greeting(core)
    await asyncio.sleep(1.2)  # let the boot animation play first
    if session.state == "idle":
        await session.speak_only(text)


async def proactive_loop() -> None:
    """Jarvis speaks first: a meeting about to start, new email, low battery."""
    said: set = set()
    last_unread = None
    battery_warned = False
    while True:
        await asyncio.sleep(PROACTIVE_EVERY)
        try:
            if not SESSIONS or load_settings().get("proactive", True) is False:
                continue
            info = await skills.hud()
            ev = info.get("event") or {}
            if ev.get("title") and ev.get("start"):
                mins = (ev["start"] - time.time()) / 60
                key = (ev["title"], ev["start"])
                if 0 < mins <= 10 and key not in said and await announce(f"Heads up, {ev['title']} starts in {max(1, round(mins))} minute{'s' if round(mins) != 1 else ''}."):
                    said.add(key)
            mail = info.get("mail") or {}
            unread = mail.get("unread")
            if isinstance(unread, int):
                if last_unread is not None and unread > last_unread and mail.get("latest") and not skills.quiet():
                    m = mail["latest"][0]
                    who = re.sub(r"\s*<.*?>", "", m.get("from", "")).strip('" ') or "someone"
                    subject = (m.get("subject") or "no subject").rstrip(".!? ")
                    if not await announce(f"New email from {who}: {subject}."):
                        continue  # busy talking: try again next round
                last_unread = unread
            b = info.get("battery")
            if b:
                if b["charging"]:
                    battery_warned = False
                elif b["percent"] <= 15 and not battery_warned:
                    battery_warned = await announce(f"Battery is at {b['percent']} percent. You might want to plug in.")
        except Exception:
            log.exception("proactive check failed")


@app.on_event("startup")
async def startup() -> None:
    # Serve the page right away; models load in the background while the orb shows "waking up".
    asyncio.create_task(load_everything())
    if load_settings().get("always_on"):
        asyncio.create_task(set_always_on(True))


async def exit_if_still_idle() -> None:
    await asyncio.sleep(20)
    if CLIENTS == 0 and not load_settings().get("always_on"):
        log.info("Window closed; shutting down.")
        os._exit(0)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global CLIENTS
    await ws.accept()
    CLIENTS += 1
    session = Session(ws)
    SESSIONS.add(session)
    await session.send(type="hello", model=LLM_MODEL)
    await session.send_brain()
    await session.send(type="history", turns=skills.memory.recent(15))

    async def announce_ready() -> None:
        if not READY.is_set():
            await session.set_state("loading")
            await READY.wait()
        await session.set_state("idle")

    ready_task = asyncio.create_task(announce_ready())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if not READY.is_set():
                continue
            if msg.get("bytes") is not None:
                frame = np.frombuffer(msg["bytes"], dtype=np.int16)
                if frame.size == FRAME:
                    await session.on_frame(frame)
            elif msg.get("text"):
                await session.on_json(json.loads(msg["text"]))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        CLIENTS -= 1
        SESSIONS.discard(session)
        ready_task.cancel()
        if getattr(session, "server_mic", None):
            session.server_mic.stop()
        if session.task:
            session.task.cancel()
        if EXIT_WHEN_IDLE and CLIENTS == 0:
            asyncio.create_task(exit_if_still_idle())


@app.get("/api/status")
async def api_status() -> dict:
    st = LOCAL.status if LOCAL else {}
    return {"always_on": LOCAL is not None, "state": st.get("state", "idle"),
            "brain": st.get("brain", load_settings().get("brain", "gemma"))}


MP_VERSION = "0.10.22-rc.20250304"  # must match @mediapipe/tasks-vision in web-src/package.json
MP_FILES = {
    "vision_wasm_internal.js": f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MP_VERSION}/wasm/vision_wasm_internal.js",
    "vision_wasm_internal.wasm": f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MP_VERSION}/wasm/vision_wasm_internal.wasm",
    "vision_wasm_nosimd_internal.js": f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MP_VERSION}/wasm/vision_wasm_nosimd_internal.js",
    "vision_wasm_nosimd_internal.wasm": f"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@{MP_VERSION}/wasm/vision_wasm_nosimd_internal.wasm",
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
}


@app.get("/mp/{name}")
async def mediapipe_file(name: str):
    """Hand-gesture model files: downloaded once (about 18 MB), then served from the Jarvis folder."""
    from fastapi import HTTPException
    if name not in MP_FILES:
        raise HTTPException(404)
    path = MODELS / "mediapipe" / f"{MP_VERSION}-{name}" if name != "gesture_recognizer.task" else MODELS / "mediapipe" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s for hand gestures", name)
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
            r = await c.get(MP_FILES[name])
            r.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.rename(path)
    media = "application/wasm" if name.endswith(".wasm") else "text/javascript" if name.endswith(".js") else "application/octet-stream"
    return FileResponse(path, media_type=media)


@app.get("/api/mics")
async def api_mics() -> dict:
    import background
    try:
        return {"devices": await asyncio.get_running_loop().run_in_executor(None, background.input_devices)}
    except Exception as e:
        return {"devices": [], "error": str(e)}


@app.post("/api/clientlog")
async def api_clientlog(msg: dict) -> dict:
    """Errors from the orb window, so problems show up in jarvis.log."""
    log.warning("window: %s", str(msg.get("message", ""))[:500])
    return {"ok": True}


@app.get("/api/hud")
async def api_hud() -> dict:
    return await skills.hud()


@app.post("/api/cmd")
async def api_cmd(msg: dict) -> dict:
    if LOCAL is None or not READY.is_set():
        return {"ok": False}
    if msg.get("type") in ("ptt", "set_brain"):
        await LOCAL.session.on_json(msg)
        if msg.get("type") == "set_brain":
            for s in list(SESSIONS):
                s.brain = LOCAL.session.brain
                await s.send_brain()
    return {"ok": True}


async def always_on_then_refresh(on: bool) -> None:
    try:
        await set_always_on(on)
    finally:
        for s in list(SESSIONS):
            try:
                await s.send_brain()
            except Exception:
                pass


@app.post("/api/always_on")
async def api_always_on(msg: dict) -> dict:
    asyncio.create_task(always_on_then_refresh(bool(msg.get("on"))))
    return {"ok": True}


@app.post("/api/quit")
async def api_quit() -> dict:
    asyncio.get_running_loop().call_later(0.3, os._exit, 0)
    return {"ok": True}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")


if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
    finally:
        # e.g. port already taken by another Jarvis: don't linger with model threads running
        logging.shutdown()
        os._exit(0)
