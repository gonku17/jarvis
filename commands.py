"""Spoken commands that change how Jarvis runs: which local model, which voice, offline mode.

Pure functions (no I/O) so they can be tested without Ollama or a microphone.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- shared helpers

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fourteen": "14", "twenty": "20",
    "twenty seven": "27", "thirty": "30", "thirty two": "32", "seventy": "70",
}

# What Whisper tends to write when you say a model's name.
SPOKEN_FAMILIES = [
    (r"\b(?:qwen|quen|quinn|qwin|kwen|gwen|queen|quin)\b", "qwen"),
    (r"\b(?:gemma|jemma|gemma's|gema|gemmer)\b", "gemma"),
    (r"\b(?:llama|lama|llamma|lamma)\b", "llama"),
    (r"\bdeep ?seek\b", "deepseek"),
    (r"\bg\.? ?p\.? ?t\.?[ -]?o\.? ?s\.? ?s\.?\b|\bgpt[ -]?os\b|\bgpt[ -]?oss\b", "gpt-oss"),
    (r"\b(?:mistral|mistrial)\b", "mistral"),
    (r"\b(?:phi|fi|fie)\b", "phi"),
    (r"\bsmall ?lm\b", "smollm"),
]
FAMILIES = ("qwen", "gemma", "llama", "deepseek", "gpt-oss", "mistral", "mixtral", "phi", "smollm",
            "granite", "olmo", "command-r", "nemotron", "hermes", "codellama", "tinyllama", "magistral")

SWITCH_VERB = r"(?:switch|change|swap|go|move|flip|jump|set)"


def _clean(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"^(?:hey |ok |okay )?jarvis[,.]? ", "", t)
    return t.strip().rstrip(".!?").strip()


def tokens(name: str) -> list[str]:
    """'qwen3.5:9b' -> ['qwen', '3.5', '9b'];  'gpt-oss:20b' -> ['gpt', 'oss', '20b']."""
    name = name.lower().removesuffix(":latest")
    return re.findall(r"[a-z]+|\d+(?:\.\d+)?b?", name)


def normalise_spoken_model(phrase: str) -> str:
    p = " " + phrase.lower() + " "
    for pat, fam in SPOKEN_FAMILIES:
        p = re.sub(pat, fam, p)
    for word in sorted(NUMBER_WORDS, key=len, reverse=True):
        p = re.sub(rf"\b{word}\b", NUMBER_WORDS[word], p)
    p = re.sub(r"(\d) ?point ?(\d)", r"\1.\2", p)
    p = re.sub(r"(\d+(?:\.\d+)?) ?(?:billion|bn|b)\b(?: parameters?)?", r"\1b", p)
    p = re.sub(r"\b(?:the|a|an|model|brain|version|parameters?|local|your|my|please|one|llm)\b", " ", p)
    return re.sub(r"\s+", " ", p).strip()


def pretty_model(name: str) -> str:
    """'qwen3.5:9b' -> 'Qwen 3.5 9B';  'gemma4:e4b' -> 'Gemma 4 E4B';  'llama3.2:latest' -> 'Llama 3.2'."""
    base, _, tag = name.partition(":")
    base = base.split("/")[-1]
    family = re.sub(r"(?<=[a-z])(?=\d)", " ", base.replace("-", " ")).split()
    words = [w.upper() if w in ("gpt", "oss", "r") else w.capitalize() for w in family]
    label = " ".join(words).replace("GPT OSS", "GPT-OSS")
    if tag and tag != "latest":
        label += " " + tag.upper()
    return label


def say_model(name: str) -> str:
    """How to say a model name out loud: 'Qwen 3.5, 9 billion'."""
    label = re.sub(r" E\d+B$", "", pretty_model(name))  # "Gemma 4 E4B" -> "Gemma 4"
    return re.sub(r" (\d+(?:\.\d+)?)B\b", r", \1 billion", label)


# ---------------------------------------------------------------- local model switching

LIST_MODELS_RE = re.compile(
    r"\b(?:what|which) (?:local )?(?:models|brains)\b|\blist (?:your |the |my )?(?:local )?(?:models|brains)\b|"
    r"\bwhat (?:local )?(?:models|brains) (?:do you have|are installed|can you (?:use|run|switch to))\b|"
    r"\bwhich (?:model|brain) are you (?:using|on|running)\b|\bwhat (?:model|brain) are you (?:using|on|running)\b"
)
MODEL_SWITCH_RE = re.compile(
    rf"^(?:please |can you |could you )?(?:{SWITCH_VERB})(?: over)?(?: your| my| the)?(?: local)?(?: brain| model)?"
    rf"(?: over)? (?:to|into) (?:the |your |my )?(?P<x>.+?)(?: model| brain)?(?: please)?$"
    rf"|^(?:please )?(?:use|try|load|run) (?:the |your |my )?(?P<y>.+?) (?:model|brain)(?: please)?$"
    rf"|^(?:please )?(?:use|try|load|run) (?:the |your |my )?(?P<z>(?:{'|'.join(FAMILIES)})\b.*)$"
)
SPECIAL_MODELS = {
    "smallest": "smallest", "small": "smallest", "fastest": "smallest", "quickest": "smallest", "tiny": "smallest",
    "biggest": "biggest", "big": "biggest", "largest": "biggest", "smartest": "biggest", "most powerful": "biggest",
    "default": "default", "usual": "default", "normal": "default", "standard": "default", "original": "default",
}


def parse_model_switch(text: str) -> str | None:
    """Returns 'list', a special ('smallest', 'biggest', 'default'), a normalised model phrase, or None.

    Plain 'switch to Gemma / Claude / local / offline' is left to the brain switch.
    """
    t = _clean(text)
    if LIST_MODELS_RE.search(t):
        return "list"
    m = MODEL_SWITCH_RE.match(t)
    if not m:
        return None
    raw = (m.group("x") or m.group("y") or m.group("z") or "").strip()
    raw = re.sub(r"\b(?:brain|model)\b", " ", raw).strip()
    raw_words = raw.split()
    if raw_words and raw_words[-1] in ("one",) and len(raw_words) > 1:
        raw = " ".join(raw_words[:-1])
    if raw in SPECIAL_MODELS:
        return SPECIAL_MODELS[raw]
    phrase = normalise_spoken_model(raw)
    if not phrase or phrase in ("claude", "cloud", "gemma", "local", "offline", "online"):
        return None
    if not any(re.search(rf"\b{re.escape(f)}", phrase) for f in FAMILIES):
        return None  # "switch to the kitchen speaker" isn't about models
    if "voice" in phrase or "mode" in phrase:
        return None
    return phrase


def match_model(phrase: str, models: list[dict]) -> list[dict]:
    """Installed Ollama models (dicts with 'name' and 'size') that fit a spoken phrase, best first."""
    want = tokens(phrase)
    if not want:
        return []
    compact = re.sub(r"[^a-z0-9]", "", phrase)
    scored = []
    for m in models:
        name = m["name"]
        have = tokens(name)
        joined = re.sub(r"[^a-z0-9]", "", name.lower().removesuffix(":latest"))
        if compact == joined:
            score = 2.0
        elif all(w in have for w in want):
            score = len(want) / len(have)
        elif joined.startswith(compact) and len(compact) >= 4:
            score = 0.5 * len(compact) / len(joined)
        else:
            continue
        scored.append((-score, m.get("size") or 0, name, m))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return [s[3] for s in scored]


def pick_special(kind: str, models: list[dict], default: str) -> dict | None:
    chat = [m for m in models if "embed" not in m["name"]]
    if not chat:
        return None
    if kind == "smallest":
        return min(chat, key=lambda m: m.get("size") or 0)
    if kind == "biggest":
        return max(chat, key=lambda m: m.get("size") or 0)
    for m in chat:
        if m["name"] in (default, default + ":latest"):
            return m
    return None


# ---------------------------------------------------------------- voice switching

VOICE_WORDS = r"\b(?:voice|voices|accent|sound)\b"
LIST_VOICES_RE = re.compile(r"\b(?:what|which) voices\b|\blist (?:your |the )?voices\b|\bwhat voice are you\b")
LOCAL_VOICE_RE = re.compile(r"\b(?:local|offline|on-device|on device) voice\b")
VOICE_CMD_RE = re.compile(
    rf"\b(?:{SWITCH_VERB}|use|try|give (?:me|us)|pick|have|want)\b.*\b(?:voice|accent)\b|\b(?:voice|accent)\b.*\b(?:to|as)\b|"
    rf"\bsound (?:like|more)\b|\b(?:another|different|next|new) voice\b"
)
GENDER_WORDS = {"female": "f", "woman": "f", "women": "f", "girl": "f", "lady": "f", "woman's": "f",
                "male": "m", "man": "m", "man's": "m", "guy": "m", "men": "m", "boy": "m"}
ACCENT_WORDS = {"british": "gb", "english": "gb", "uk": "gb", "posh": "gb", "american": "us", "america": "us",
                "scottish": "scot", "northern": "north"}


def voice_accent(voice_id: str, label: str) -> str:
    low = label.lower()
    if "scottish" in low:
        return "scot"
    if "northern" in low:
        return "north"
    return "us" if "american" in low else "gb"


def parse_voice_switch(text: str, voices: dict[str, str], genders: dict[str, str], current: str) -> str | None:
    """Returns a voice id, 'list', 'local', or None.  voices: id -> 'Alan · British ⚡'."""
    t = _clean(text)
    if LIST_VOICES_RE.search(t):
        return "list"
    if not re.search(VOICE_WORDS, t) or not VOICE_CMD_RE.search(t):
        return None
    if LOCAL_VOICE_RE.search(t):
        return "local"
    ids = list(voices)
    if re.search(r"\b(?:another|different|next|new) voice\b", t):
        i = ids.index(current) if current in ids else -1
        return ids[(i + 1) % len(ids)]
    if re.search(r"\b(?:default|usual|normal|original|standard|fast|piper)\b", t):
        return "piper"
    words = re.findall(r"[a-z']+", t)
    # A voice by name: "use George's voice", "switch to Emma"
    names = {voices[v].split(" ·")[0].lower(): v for v in ids}
    for w in words:
        w = w.removesuffix("'s")
        if w in names and w not in ("northern",):
            return names[w]
    if "kokoro" in words:
        return current if not current.startswith("piper") else "bm_george"
    gender = next((GENDER_WORDS[w] for w in words if w in GENDER_WORDS), None)
    accent = next((ACCENT_WORDS[w] for w in words if w in ACCENT_WORDS), None)
    if not gender and not accent:
        return None
    fits = [v for v in ids
            if (not gender or genders.get(v) == gender) and (not accent or voice_accent(v, voices[v]) == accent)]
    if not fits:
        return None
    others = [v for v in fits if v != current]
    return (others or fits)[0]


# ---------------------------------------------------------------- strict offline mode

OFFLINE_ON_RE = re.compile(
    r"^(?:please )?(?:go|stay|work|be) offline(?: mode)?(?: now| please)?$|"
    r"\b(?:turn on|enable|start|switch on|switch to|go into|enter|activate) (?:the )?offline mode\b|"
    r"^offline mode(?: on| please)?$|\bdisconnect from the internet\b|\bcut (?:off )?the internet\b"
)
OFFLINE_OFF_RE = re.compile(
    r"^(?:please )?(?:go|come|get) (?:back )?online(?: now| please)?$|^back online$|"
    r"\b(?:turn off|disable|stop|switch off|leave|exit|end|deactivate) (?:the )?offline mode\b|"
    r"^offline mode off$|\breconnect (?:to )?the internet\b|^you can go online(?: again)?$"
)
OFFLINE_ASK_RE = re.compile(r"^(?:are you|is) (?:offline|online|in offline mode)\b|^(?:are you|am i) connected")


def parse_offline(text: str) -> str | None:
    """'on', 'off', 'status' or None."""
    t = _clean(text)
    if OFFLINE_OFF_RE.search(t):
        return "off"
    if OFFLINE_ON_RE.search(t):
        return "on"
    if OFFLINE_ASK_RE.search(t):
        return "status"
    return None
