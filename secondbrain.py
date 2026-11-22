"""Second brain: search a folder of your own notes and documents, all on this Mac.

Uses Spotlight (mdfind) when it's available, which also looks inside PDFs and Word files,
and falls back to walking the folder. Nothing leaves the computer unless the Claude brain
is selected, in which case the short excerpts go to Claude along with the question.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXT = {".txt", ".md", ".markdown", ".text", ".csv", ".tsv", ".json", ".yaml", ".yml", ".org", ".rst",
            ".html", ".htm", ".log", ".tex", ".py", ".js", ".ts", ".swift", ".sh", ".ini", ".toml"}
DOC_EXT = {".pdf", ".docx", ".doc", ".rtf", ".rtfd", ".odt", ".pages", ".key", ".numbers", ".pptx", ".xlsx"}
CHEAP_DOC_EXT = {".docx"}  # quick to read without Spotlight; PDFs are only matched by name there
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "Library", ".Trash", "build", "dist"}
STOP = set("""a an the of to in on at for and or is are was were be been my me i you your our about what whats
what's does do did say says said have has had any anything something tell show find search look check pull up
notes note files file documents document docs doc second brain folder from with that this it its please jarvis
can could would will know there where which who how when info information stuff things""".split())

MAX_WALK_ENTRIES = 30000
WALK_SECONDS = 3.0


def terms_of(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9'+#.-]*", query.lower())
    out = []
    for w in words:
        w = w.strip("'.-")
        if len(w) >= 2 and w not in STOP and w not in out:
            out.append(w)
    return out


@dataclass
class Hit:
    path: Path
    score: float
    is_dir: bool = False
    excerpt: str = ""
    children: list[str] = field(default_factory=list)


def read_text(path: Path, limit: int = 400_000) -> str:
    """Best-effort plain text of a file. Empty string if it can't be read."""
    ext = path.suffix.lower()
    try:
        if ext in TEXT_EXT:
            with open(path, "r", errors="ignore") as f:
                text = f.read(limit)
            if ext in (".html", ".htm"):
                text = re.sub(r"<[^>]+>", " ", text)
            return text
        if ext == ".pdf":
            try:
                from pypdf import PdfReader  # optional
            except ImportError:
                return _textutil(path)
            reader = PdfReader(str(path))
            parts, size = [], 0
            for page in reader.pages[:30]:
                t = page.extract_text() or ""
                parts.append(t)
                size += len(t)
                if size > limit:
                    break
            return "\n".join(parts)[:limit]
        if ext == ".docx":
            with zipfile.ZipFile(path) as z:
                xml = z.read("word/document.xml").decode("utf8", "ignore")
            xml = re.sub(r"</w:p>", "\n", xml)
            return re.sub(r"<[^>]+>", "", xml)[:limit]
        if ext in (".doc", ".rtf", ".rtfd", ".odt"):
            return _textutil(path)[:limit]
    except Exception:
        return ""
    return ""


def _textutil(path: Path) -> str:
    """macOS's built-in converter (rtf, doc, docx, odt, html). Not on other systems."""
    if sys.platform != "darwin" or path.suffix.lower() == ".pdf":
        return ""
    try:
        return subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                              capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return ""


def excerpt(text: str, terms: list[str], size: int = 450) -> str:
    """The most relevant slice of text around the search terms."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    if not text:
        return ""
    low = text.lower()
    best, best_n = 0, -1
    for t in terms:
        for m in re.finditer(re.escape(t), low):
            start = max(0, m.start() - size // 3)
            window = low[start:start + size]
            n = sum(window.count(x) for x in terms)
            if n > best_n:
                best, best_n = start, n
            break  # first occurrence of each term is enough
    snippet = text[best:best + size].strip()
    return ("…" if best > 0 else "") + snippet + ("…" if best + size < len(text) else "")


def _spotlight(root: Path, terms: list[str]) -> list[Path]:
    if sys.platform != "darwin" or not terms:
        return []
    found: list[str] = []
    try:
        query = " ".join(terms)
        found += subprocess.run(["mdfind", "-onlyin", str(root), query],
                                capture_output=True, text=True, timeout=6).stdout.splitlines()
        for t in terms[:3]:
            found += subprocess.run(["mdfind", "-onlyin", str(root), "-name", t],
                                    capture_output=True, text=True, timeout=6).stdout.splitlines()
    except Exception:
        return []
    seen, out = set(), []
    for p in found:
        if p not in seen and "/." not in p and "/Library/" not in p:
            seen.add(p)
            out.append(Path(p))
    return out[:400]


def _walk(root: Path, terms: list[str]) -> list[Path]:
    """Every file and folder under root whose name has a search term, plus small text files whose contents do."""
    out: list[Path] = []
    t0, n = time.monotonic(), 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
        for name in dirnames + filenames:
            n += 1
            p = Path(dirpath) / name
            low = name.lower()
            if any(t in low for t in terms):
                out.append(p)
            elif name in filenames and p.suffix.lower() in TEXT_EXT | CHEAP_DOC_EXT:
                try:
                    if p.stat().st_size < 2_000_000:
                        body = read_text(p, limit=200_000).lower()
                        if any(t in body for t in terms):
                            out.append(p)
                except OSError:
                    pass
        if n > MAX_WALK_ENTRIES or time.monotonic() - t0 > WALK_SECONDS:
            break
    return out


def search(root: Path, query: str, limit: int = 3) -> list[Hit]:
    terms = terms_of(query)
    if not terms or not root.is_dir():
        return []
    candidates = _spotlight(root, terms) or _walk(root, terms)
    hits: list[Hit] = []
    for p in candidates:
        try:
            is_dir = p.is_dir() and p.suffix.lower() not in (".rtfd", ".pages", ".key", ".numbers")
        except OSError:
            continue
        name = p.stem.lower() if not is_dir else p.name.lower()
        rel = str(p.relative_to(root)).lower() if p.is_relative_to(root) else name
        score = 0.0
        score += sum(4 for t in terms if t in name)
        score += sum(1 for t in terms if t in rel and t not in name)
        if all(t in name for t in terms):
            score += 4
        text = ""
        if not is_dir:
            if p.suffix.lower() not in TEXT_EXT | DOC_EXT:
                continue
            text = read_text(p)
            low = text.lower()
            score += sum(min(low.count(t), 5) for t in terms)
        if score <= 0:
            continue
        try:
            score += max(0.0, 1.0 - (time.time() - p.stat().st_mtime) / (365 * 86400))  # a little boost for recent
        except OSError:
            pass
        hit = Hit(path=p, score=score, is_dir=is_dir)
        if is_dir:
            try:
                kids = sorted((c for c in p.iterdir() if not c.name.startswith(".")),
                              key=lambda c: c.stat().st_mtime, reverse=True)
                hit.children = [c.name for c in kids[:12]]
            except OSError:
                pass
        else:
            hit.excerpt = excerpt(text, terms)
        hits.append(hit)
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def describe(root: Path, hits: list[Hit], max_chars: int = 1800) -> str:
    """Context for the LLM: where each hit is and the relevant slice of it."""
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        where = h.path.relative_to(root) if h.path.is_relative_to(root) else h.path
        if h.is_dir:
            body = "Folder containing: " + (", ".join(h.children) if h.children else "nothing")
        else:
            body = h.excerpt or "(couldn't read the text of this file)"
        block = f"[{i}] {where}\n{body}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        parts.append(block)
        used += len(block)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


def readable_start(path: Path, chars: int = 900) -> str:
    """The beginning of a file, tidied for reading aloud."""
    text = read_text(path, limit=20_000)
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)  # front matter
    text = re.sub(r"[#*_`>|]+", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= chars:
        return text
    cut = text[:chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > chars // 2 else cut + "…"


def resolve_folder(spoken: str, home: Path | None = None) -> Path | None:
    """'documents' -> ~/Documents;  'obsidian vault' -> ~/Documents/Obsidian Vault, if it exists."""
    home = home or Path.home()
    want = re.sub(r"\b(?:my|the|folder|directory)\b", " ", spoken.lower())
    want = re.sub(r"\s+", " ", want).strip()
    if not want:
        return None
    if want.startswith(("/", "~")):
        p = Path(want).expanduser()
        return p if p.is_dir() else None
    bases = [home, home / "Documents", home / "Desktop", home / "Library/Mobile Documents/com~apple~CloudDocs"]
    for base in bases:
        try:
            for child in base.iterdir():
                if child.is_dir() and not child.name.startswith(".") and child.name.lower() == want:
                    return child
        except OSError:
            continue
    for base in bases:
        try:
            for child in base.iterdir():
                if child.is_dir() and not child.name.startswith(".") and want in child.name.lower():
                    return child
        except OSError:
            continue
    return None
