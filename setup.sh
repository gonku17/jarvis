#!/usr/bin/env bash
# One-time setup for Jarvis on macOS. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${JARVIS_LLM:-gemma4:e4b}"

echo "==> Checking Ollama"
if ! command -v ollama >/dev/null 2>&1 && [ ! -d /Applications/Ollama.app ]; then
  echo "Ollama is not installed. Download it from https://ollama.com/download, open it once, then re-run ./setup.sh"
  exit 1
fi
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
  open -a Ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 20); do curl -s http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
fi
echo "==> Downloading the language model ($MODEL). This is several GB and only happens once."
ollama pull "$MODEL"
ollama pull gemma3:1b   # small model for quick everyday answers

echo "==> Setting up Python (in ./.venv, nothing system-wide)"
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"
  else curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
fi
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "==> Downloading the wake word, speech and voice models"
.venv/bin/python -c "import server; server.engines.load()"

echo
echo "All set. Start Jarvis with:  ./start.sh"
