"""Unit tests for spoken commands and the second brain. No Ollama, mic or Mac needed.

Run:  .venv/bin/python tests/test_commands.py    (or: python -m pytest tests/test_commands.py)
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import commands  # noqa: E402
import secondbrain  # noqa: E402

MODELS = [
    {"name": "gemma4:e4b", "size": 9_600_000_000},
    {"name": "gemma3:1b", "size": 815_000_000},
    {"name": "gemma3:12b", "size": 8_100_000_000},
    {"name": "qwen3.5:9b", "size": 6_600_000_000},
    {"name": "qwen3.5:27b", "size": 17_000_000_000},
    {"name": "gpt-oss:20b", "size": 13_000_000_000},
    {"name": "llama3.2:latest", "size": 2_000_000_000},
]

VOICES = {
    "piper": "Alan · British ⚡",
    "piper:en_GB-northern_english_male-medium": "Northern · British ⚡",
    "piper:en_US-ryan-medium": "Ryan · American ⚡",
    "piper:en_GB-jenny_dioco-medium": "Jenny · British ⚡",
    "piper:en_GB-alba-medium": "Alba · Scottish ⚡",
    "piper:en_US-amy-medium": "Amy · American ⚡",
    "bm_george": "George · British", "bf_emma": "Emma · British", "af_heart": "Heart · American",
}
GENDER = {"piper": "m", "piper:en_GB-northern_english_male-medium": "m", "piper:en_US-ryan-medium": "m",
          "piper:en_GB-jenny_dioco-medium": "f", "piper:en_GB-alba-medium": "f", "piper:en_US-amy-medium": "f",
          "bm_george": "m", "bf_emma": "f", "af_heart": "f"}


def best(phrase: str) -> str | None:
    found = commands.match_model(phrase, MODELS)
    return found[0]["name"] if found else None


def test_model_switch_parsing():
    p = commands.parse_model_switch
    assert p("Switch to Qwen") == "qwen"
    assert p("Jarvis, switch to Quen 3.5 9 billion parameters") == "qwen 3.5 9b"
    assert p("switch your brain to gemma 12 billion") == "gemma 12b"
    assert p("use the gpt oss model") == "gpt-oss"
    assert p("change to llama 3.2") == "llama 3.2"
    assert p("switch to your smallest model") == "smallest"
    assert p("go back to the default model") is None or p("go back to the default model") == "default"
    assert p("switch to the default model") == "default"
    assert p("what models do you have") == "list"
    assert p("which model are you using?") == "list"
    # left to the existing Gemma/Claude switch, or not about models at all
    assert p("switch to Claude") is None
    assert p("switch to Gemma") is None
    assert p("switch to your local brain") is None
    assert p("set volume to 40") is None
    assert p("set a timer for 10 minutes") is None
    assert p("go to sleep") is None
    assert p("switch to the kitchen speaker") is None


def test_model_matching():
    assert best("qwen") == "qwen3.5:9b"            # smallest of the two when only the family is said
    assert best("qwen 27b") == "qwen3.5:27b"
    assert best("qwen 3.5 9b") == "qwen3.5:9b"
    assert best("gemma 12b") == "gemma3:12b"
    assert best("gemma 4") == "gemma4:e4b"
    assert best("gpt-oss") == "gpt-oss:20b"
    assert best("gpt-oss 20b") == "gpt-oss:20b"
    assert best("llama") == "llama3.2:latest"
    assert best("mistral") is None
    assert commands.pick_special("smallest", MODELS, "gemma4:e4b")["name"] == "gemma3:1b"
    assert commands.pick_special("biggest", MODELS, "gemma4:e4b")["name"] == "qwen3.5:27b"
    assert commands.pick_special("default", MODELS, "gemma4:e4b")["name"] == "gemma4:e4b"


def test_model_names():
    assert commands.pretty_model("qwen3.5:9b") == "Qwen 3.5 9B"
    assert commands.pretty_model("gemma4:e4b") == "Gemma 4 E4B"
    assert commands.pretty_model("llama3.2:latest") == "Llama 3.2"
    assert commands.pretty_model("gpt-oss:20b") == "GPT-OSS 20B"
    assert commands.say_model("qwen3.5:27b") == "Qwen 3.5, 27 billion"


def test_voice_switch():
    v = lambda text, cur="piper": commands.parse_voice_switch(text, VOICES, GENDER, cur)  # noqa: E731
    assert v("switch to George's voice") == "bm_george"
    assert v("use Emma's voice") == "bf_emma"
    assert v("change your voice to Emma") == "bf_emma"
    assert v("use a female voice") == "piper:en_GB-jenny_dioco-medium"
    assert v("switch to an American voice") == "piper:en_US-ryan-medium"
    assert v("give me a female American voice") == "piper:en_US-amy-medium"
    assert v("use a Scottish accent") == "piper:en_GB-alba-medium"
    assert v("switch to the kokoro voice") == "bm_george"
    assert v("go back to your normal voice", "bf_emma") == "piper"
    assert v("try a different voice") == "piper:en_GB-northern_english_male-medium"
    assert v("switch to your local voice") == "local"
    assert v("what voices do you have") == "list"
    assert v("turn the sound up") is None
    assert v("what's the weather") is None
    assert v("give us a female voice") == "piper:en_GB-jenny_dioco-medium"


def test_offline():
    o = commands.parse_offline
    assert o("go offline") == "on"
    assert o("Jarvis, turn on offline mode") == "on"
    assert o("switch to offline mode") == "on"
    assert o("offline mode") == "on"
    assert o("go back online") == "off"
    assert o("turn off offline mode") == "off"
    assert o("are you offline?") == "status"
    assert o("switch to your offline brain") is None   # still the plain brain switch
    assert o("what's the weather") is None


def test_second_brain():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "Free School Community").mkdir()
        (root / "Free School Community" / "launch plan.md").write_text(
            "# Launch plan\nThe Skool community opens on 1 March. Price is free for the first 100 members.")
        (root / "Recipes").mkdir()
        (root / "Recipes" / "pancakes.txt").write_text("Pancakes: 2 eggs, 200 g flour, 300 ml milk.")
        (root / ".hidden").mkdir()
        (root / ".hidden" / "pancakes secret.txt").write_text("pancakes pancakes")
        with zipfile.ZipFile(root / "contract.docx", "w") as z:
            z.writestr("word/document.xml", "<w:document><w:p><w:t>The rent is 950 pounds a month.</w:t></w:p></w:document>")

        hits = secondbrain._walk(root, ["pancakes"])
        assert all(".hidden" not in str(h) for h in hits)

        hits = secondbrain.search(root, "free school community")
        assert hits and hits[0].path.name == "Free School Community" and hits[0].is_dir
        assert "launch plan.md" in hits[0].children

        hits = secondbrain.search(root, "how much flour for pancakes")
        assert hits[0].path.name == "pancakes.txt" and "200 g flour" in hits[0].excerpt

        hits = secondbrain.search(root, "when does the skool community open")
        assert any(h.path.name == "launch plan.md" for h in hits)

        hits = secondbrain.search(root, "rent")
        assert hits[0].path.name == "contract.docx" and "950" in hits[0].excerpt

        assert secondbrain.search(root, "quantum chromodynamics") == []
        text = secondbrain.describe(root, secondbrain.search(root, "pancakes"))
        assert "Recipes/pancakes.txt" in text
        assert secondbrain.readable_start(root / "Free School Community" / "launch plan.md").startswith("Launch plan")
        assert secondbrain.resolve_folder("my recipes folder", home=root) == root / "Recipes"
        assert secondbrain.resolve_folder("nothing here", home=root) is None


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failed += 1
                import traceback
                print(f"FAIL  {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
