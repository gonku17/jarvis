#!/bin/bash
# Jarvis launcher (lives in the Jarvis folder so updates can replace it).
JARVIS_DIR="${JARVIS_DIR:-$HOME/Jarvis}"
URL="http://127.0.0.1:8765/?app=1"
LOG="$JARVIS_DIR/jarvis.log"
PROFILE="$HOME/Library/Application Support/Jarvis/window"
export PATH="/usr/local/bin:/opt/homebrew/bin:$HOME/.local/bin:$PATH"

notify() { osascript -e "display notification \"$1\" with title \"Jarvis\"" >/dev/null 2>&1; }
fail() { osascript -e "display alert \"Jarvis\" message \"$1\" as critical" >/dev/null 2>&1; exit 1; }

[ -x "$JARVIS_DIR/.venv/bin/python" ] || fail "Jarvis isn't set up in $JARVIS_DIR yet. Run setup.sh there first."
# Already open (maybe hidden with the red button)? Ask that copy to show itself.
if [ "${1:-}" != "--background" ] && pgrep -f "python window.py" >/dev/null 2>&1; then
  touch "$JARVIS_DIR/.show_window"
  exit 0
fi

# 1. Ollama (the brain)
if ! curl -s -m 1 http://127.0.0.1:11434/api/tags >/dev/null; then
  open -g -a Ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 30); do curl -s -m 1 http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
fi

# 2. Jarvis server (exits by itself ~20s after the window closes, unless Always on is enabled)
# (a port check, not an HTTP probe: the server can be too busy loading to answer quickly)
if ! lsof -ti tcp:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  cd "$JARVIS_DIR" || fail "Can't open $JARVIS_DIR"
  JARVIS_EXIT_WHEN_IDLE=1 nohup .venv/bin/python server.py >>"$LOG" 2>&1 &
  for _ in $(seq 40); do curl -s -m 1 -o /dev/null http://127.0.0.1:8765/ && break; sleep 0.5; done
  curl -s -m 1 -o /dev/null http://127.0.0.1:8765/ || fail "Jarvis didn't start. See $LOG"
fi

# Started at login for Always on: the server keeps running with the menu bar icon; no window.
[ "${1:-}" = "--background" ] && exit 0

# 3. The orb window: Jarvis's own native window. `exec` keeps this the Jarvis.app process.
if "$JARVIS_DIR/.venv/bin/python" -c "import WebKit, AppKit" 2>/dev/null; then
  cd "$JARVIS_DIR" && exec .venv/bin/python window.py "$URL" >>"$JARVIS_DIR/window.log" 2>&1
fi
# Fallback if the window packages aren't installed: a browser in app mode, else Safari.
for app in "Google Chrome" "Microsoft Edge" "Brave Browser" "Chromium"; do
  if open -Ra "$app" 2>/dev/null; then
    mkdir -p "$PROFILE"
    open -n -a "$app" --args --app="$URL" --user-data-dir="$PROFILE" \
      --autoplay-policy=no-user-gesture-required --window-size=980,860 \
      --no-first-run --no-default-browser-check
    exit 0
  fi
done
open -a Safari "$URL"
