# Jarvis (local voice assistant)

Everything runs on your Mac. Nothing is sent to the cloud.

```
Browser (mic + Siri-style orb)  <->  server.py
  "Hey Jarvis"  -> openWakeWord
  your speech   -> faster-whisper (base.en)
  the reply     -> Gemma 4 E4B via Ollama (streamed)
  the voice     -> Piper (en_GB-alan-medium), sentence by sentence
```

## Setup (once)
1. Install Ollama from https://ollama.com/download and open it once.
2. In Terminal: `cd` into this folder and run `./setup.sh`. It downloads the model (several GB), creates a private Python environment in `.venv`, and fetches the speech models.

## Jarvis.app
Copy `Jarvis.app` into Applications (or ~/Applications) and drag it to the Dock. Double-clicking it starts Ollama
and the Jarvis server, then opens the orb in its own window (Chrome/Edge/Brave app mode, or Safari as a fallback).
Close the window and the server shuts itself down about 20 seconds later. Logs go to `jarvis.log`.

## Run from Terminal
`./start.sh` opens http://127.0.0.1:8765. Click **Start Jarvis** and allow the microphone.

- Say **"Hey Jarvis"**, then your question. After it answers you have about 6 seconds to follow up without the wake word.
- Click the orb to talk without the wake word, or type in the box.

## Things to say
- **Memory:** "Remember that I take my coffee black", "What do you remember about me?", "Forget about coffee", "Forget everything". It also picks up "my name is…" and "I live in…". Stored in `memory.json`.
- **Mac:** "Open Safari", "Quit Spotify", "Set a timer for 10 minutes", "Remind me to call mum at 6pm", "Volume to 40", "Mute", "Pause the music", "Next song", "Run shortcut Focus".
- **Live info:** "What's the weather?", "Will it rain tomorrow in Porto?", "What's the news?", "News about Formula 1", "Search for the best pizza in Lisbon", "Who won the match last night?".
- **Time:** "What time is it?", "What's the date?"
- **Local brains:** "What models do you have?", "Switch to Qwen", "Switch to Gemma 12 billion", "Use your smallest model", "Switch to the default model". Any model you've installed with `ollama pull` works. The old one is unloaded so they don't both sit in memory.
- **Voices:** "Use George's voice", "Switch to a female voice", "Try an American accent", "Try a different voice", "Go back to your normal voice", "What voices do you have?".
- **Second brain:** "What do my notes say about the launch?", "Search my files for the lease", "Pull up my free school community", then "Read it out" or "Open it". Searches `~/Documents` by default; say "Use my Obsidian folder as your second brain" or set it in settings. Uses Spotlight, so it also looks inside PDFs and Word files. The first time, macOS may ask to let Jarvis access Documents: click OK.
- **Offline mode:** "Go offline" / "Go online", "Are you offline?".

The first reminder or app action may trigger a macOS permission prompt ("Jarvis wants to control Reminders"): click OK.

## Claude brain (optional)
Jarvis thinks with Gemma (free, offline) by default. Use the Gemma | Claude switch at the top of the window, or say
"switch to Claude" / "switch to Gemma". Claude needs an Anthropic API key (console.anthropic.com, billed per use):
click the gear and paste it. It's stored in the macOS Keychain, not in any file. If the key is missing or Claude
can't be reached, Jarvis answers with Gemma instead. Model: `claude-opus-5-5` (change with `JARVIS_CLAUDE_MODEL`).

## Offline mode
The listening, thinking and speaking always run on your Mac. A few skills go online when you use them: weather
(open-meteo), news (BBC), web search (DuckDuckGo and Wikipedia), Claude if selected, and one-time downloads of new
voices or the hand-gesture model. Say **"go offline"** (or tick Offline mode in settings) and Jarvis stops all of
that: Claude is locked, and weather, news and search politely say they're unavailable. The top bar shows
"offline mode". Say **"go online"** to undo it.

## Updates
If `repo.txt` contains a GitHub repo (e.g. `gonku17/jarvis`), Jarvis.app checks it each time it opens and installs
any new version in place. Your memory, settings, Keychain key, models and Python environment are kept.
Run `bash update.sh` to update by hand. Log: `update.log`.

## Tweaks (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_LLM` | `gemma4:e4b` | Any Ollama model, e.g. `gemma4:e2b` for faster replies |
| `JARVIS_WHISPER` | `base.en` | `tiny.en` faster, `small.en` more accurate |
| `JARVIS_VOICE` | `en_GB-alan-medium` | Any Piper voice name |
| `JARVIS_WAKE_THRESHOLD` | `0.5` | Raise if it wakes by accident, lower if it misses you |
| `JARVIS_USER_NAME` | `sir` | What Jarvis calls you |
| `JARVIS_SECOND_BRAIN` | `~/Documents` | Folder for the second brain (settings or voice override it) |

Example: `JARVIS_LLM=gemma4:e2b JARVIS_USER_NAME=Gon ./start.sh`

## Tests
`.venv/bin/python tests/test_commands.py` checks the spoken commands and second-brain search (no Ollama needed).
`.venv/bin/python tests/smoke.py` drives a running Jarvis end to end.

## Changing the UI
The orb UI source is in `web-src/` (React + Vite). `npm install && npm run build` rebuilds it into `web/`.
The orb is the Orb component from elevenlabs/ui (MIT License, Copyright (c) 2025 Eleven Labs Inc.).

## Always on

In the orb's settings (the gear), turn on **Always on**. Jarvis then listens for "Hey Jarvis" even
with the window closed, puts an icon in the menu bar (status, Talk now, brain switch, Open the orb,
Quit), and starts by itself when you log in. Turn it off from the same switch or from the menu bar.
