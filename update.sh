#!/bin/bash
# Pulls the latest Jarvis from GitHub (if repo.txt names one) and installs it in place.
# Keeps memory.json, settings.json, jarvis.log, .venv and models/. Safe to run any time.
set -u
DIR="${JARVIS_DIR:-$HOME/Documents/Jarvis}"
cd "$DIR" || exit 0
[ -f repo.txt ] || exit 0
REPO="$(tr -d ' \n' < repo.txt)"          # e.g. gonku17/jarvis
BRANCH="${JARVIS_BRANCH:-main}"
AUTH=()
[ -n "${GITHUB_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN")
echo "$(date) checking $REPO"
SHA="$(curl -fsS -m 5 "${AUTH[@]}" "https://api.github.com/repos/$REPO/commits/$BRANCH" | sed -n 's/^  "sha": "\([0-9a-f]*\)".*/\1/p' | head -1)"
[ -n "$SHA" ] || { echo "no network or repo not found"; exit 0; }
[ "$SHA" = "$(cat .version 2>/dev/null)" ] && { echo "up to date"; exit 0; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL -m 60 "${AUTH[@]}" "https://api.github.com/repos/$REPO/tarball/$SHA" -o "$TMP/j.tgz" || exit 0
tar -xzf "$TMP/j.tgz" -C "$TMP" || exit 0
SRC="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)"
[ -f "$SRC/server.py" ] || { echo "unexpected archive"; exit 0; }

OLD_REQ="$(cat requirements.txt 2>/dev/null)"
rsync -a --exclude memory.json --exclude settings.json --exclude '*.log' --exclude .venv \
  --exclude models --exclude repo.txt --exclude .version --exclude Jarvis.app "$SRC"/ ./
# Refresh the installed app bundle too (its stub only calls update.sh + launch.sh).
# First run: install it into ~/Applications and add it to the Dock.
if [ ! -d "$HOME/Applications/Jarvis.app" ] && [ ! -d "/Applications/Jarvis.app" ]; then
  mkdir -p "$HOME/Applications/Jarvis.app"
  NEW_APP=1
fi
for APP in "$HOME/Applications/Jarvis.app" "/Applications/Jarvis.app"; do
  [ -d "$APP" ] && rsync -a "$SRC/Jarvis.app/" "$APP/" && chmod +x "$APP/Contents/MacOS/Jarvis" \
    && xattr -dr com.apple.quarantine "$APP" 2>/dev/null
done
if [ "${NEW_APP:-0}" = 1 ]; then
  defaults write com.apple.dock persistent-apps -array-add "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$HOME/Applications/Jarvis.app</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>" && killall Dock
fi
chmod +x setup.sh start.sh update.sh launch.sh 2>/dev/null
if [ "$OLD_REQ" != "$(cat requirements.txt)" ] && [ -x .venv/bin/python ]; then
  echo "installing new Python packages"
  uv pip install --python .venv/bin/python -r requirements.txt
fi
echo "$SHA" > .version
echo "updated to $SHA"
osascript -e 'display notification "Jarvis was updated." with title "Jarvis"' >/dev/null 2>&1
