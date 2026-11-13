#!/usr/bin/env bash
# Starts Jarvis and opens it in your browser. Press Ctrl+C to stop.
set -euo pipefail
cd "$(dirname "$0")"
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
  open -a Ollama 2>/dev/null || (nohup ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 20); do curl -s http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
fi
( for _ in $(seq 120); do curl -s http://127.0.0.1:8765/ >/dev/null && open http://127.0.0.1:8765 && exit; sleep 1; done ) &
exec .venv/bin/python server.py
