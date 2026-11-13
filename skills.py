"""Jarvis skills: memory, Mac control and live info.

Each skill looks at what the user said. It either answers directly (fast, no LLM),
or adds context (for example web results) for the LLM to answer from, or does nothing.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import html
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx

log = logging.getLogger("jarvis.skills")


@dataclass
class Result:
    reply: str | None = None  # speak this directly, no LLM
    context: str | None = None  # give this to the LLM as extra information


# ---------------------------------------------------------------- memory

class Memory:
    """Facts about the user plus recent conversation, saved in memory.json next to Jarvis."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = {"facts": [], "history": []}
        try:
            self.data.update(json.loads(path.read_text()))
        except Exception:
            pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

    @property
    def facts(self) -> list[str]:
        return self.data["facts"]

    def remember(self, fact: str) -> None:
        fact = fact.strip().rstrip(".")
        if not fact:
            return
        # Replace an older version of the same kind of fact ("My name is…", "I live in…").
        key = re.sub(r"\W+", " ", fact.lower()).split()[:3]
        self.data["facts"] = [f for f in self.facts if re.sub(r"\W+", " ", f.lower()).split()[:3] != key]
        self.data["facts"].append(fact)
        self.save()

    def forget(self, about: str) -> int:
        about = about.lower().strip()
        before = len(self.facts)
        if about in ("everything", "all", "it all", "all of it"):
            self.data["facts"] = []
            self.data["history"] = []
        else:
            self.data["facts"] = [f for f in self.facts if about not in f.lower()]
        self.save()
        return before - len(self.facts)

    def add_turns(self, user: str, assistant: str, keep: int = 40) -> None:
        self.data["history"] += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        self.data["history"] = self.data["history"][-keep:]
        self.save()

    def recent(self, turns: int) -> list[dict]:
        return list(self.data["history"][-turns * 2 :])

    def fact_value(self, pattern: str) -> str | None:
        for f in reversed(self.facts):
            m = re.search(pattern, f, re.I)
            if m:
                return m.group(1).strip()
        return None


# Things people say about themselves that are worth keeping without being asked.
AUTO_FACTS = [
    (re.compile(r"\bmy name is ([a-z][a-z' -]{1,30})", re.I), "The user's name is {0}"),
    (re.compile(r"\bcall me ([a-z][a-z' -]{1,30})", re.I), "The user wants to be called {0}"),
    (re.compile(r"\bi live in ([a-z][a-z ,'-]{1,40})", re.I), "The user lives in {0}"),
    (re.compile(r"\bi(?:'m| am) from ([a-z][a-z ,'-]{1,40})", re.I), "The user is from {0}"),
    (re.compile(r"\bmy birthday is ([\w ,]{3,30})", re.I), "The user's birthday is {0}"),
    (re.compile(r"\bi (?:really )?(?:love|like|prefer|hate|don't like) ([^.?!]{3,60})", re.I), None),
]


def auto_facts(text: str) -> list[str]:
    found = []
    for rx, template in AUTO_FACTS:
        m = rx.search(text)
        if not m:
            continue
        value = m.group(1).strip(" .,!?")
        value = re.split(r"\b(and|but|so|because)\b", value)[0].strip()
        if template:
            found.append(template.format(value.title() if "name" in template or "called" in template else value))
        else:
            found.append("The user said: " + m.group(0).strip(" .,!?"))
    return found


# ---------------------------------------------------------------- time parsing

NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90,
}
UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400}


def _num(word: str) -> float | None:
    word = word.lower()
    if re.fullmatch(r"\d+(\.\d+)?", word):
        return float(word)
    return NUM_WORDS.get(word)


def parse_duration(text: str) -> int | None:
    """'5 minutes', 'an hour and a half', 'half an hour', '90 seconds' -> seconds."""
    t = text.lower()
    if re.search(r"\bhalf an? hour\b", t):
        return 1800
    total = 0.0
    for m in re.finditer(r"(\d+(?:\.\d+)?|[a-z-]+)\s+(second|sec|minute|min|hour|hr|day)s?\b", t):
        n = _num(m.group(1))
        if n is not None:
            total += n * UNIT_SECONDS[m.group(2)]
    if re.search(r"\band a half\b", t) and total:
        unit = re.search(r"(second|minute|hour)s?\s+and a half", t)
        if unit:
            total += UNIT_SECONDS[unit.group(1)] / 2
    return int(total) or None


def parse_when(text: str, now: dt.datetime | None = None) -> dt.datetime | None:
    """'in 10 minutes', 'at 5pm', 'at 17:30', 'tomorrow at 9', 'tomorrow' -> datetime."""
    now = now or dt.datetime.now()
    t = text.lower()
    m = re.search(r"\bin (.+)", t)
    if m:
        secs = parse_duration(m.group(1))
        if secs:
            return now + dt.timedelta(seconds=secs)
    day = now.date() + dt.timedelta(days=1) if "tomorrow" in t else now.date()
    m = re.search(r"\bat (\d{1,2})(?:[:.](\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", t)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        ampm = (m.group(3) or "").replace(".", "")
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if not ampm and hour < 8 and "morning" not in t:
            hour += 12  # "at 5" most likely means 5pm
        if hour > 23 or minute > 59:
            return None
        when = dt.datetime.combine(day, dt.time(hour, minute))
        if when <= now and "tomorrow" not in t:
            when += dt.timedelta(days=1)
        return when
    if "tomorrow" in t:
        return dt.datetime.combine(day, dt.time(9, 0))
    return None


def say_time(when: dt.datetime) -> str:
    now = dt.datetime.now()
    clock = when.strftime("%-I:%M %p").replace(":00 ", " ")
    if when.date() == now.date():
        return f"at {clock}"
    if when.date() == now.date() + dt.timedelta(days=1):
        return f"tomorrow at {clock}"
    return when.strftime(f"on %A at {clock}")


def say_duration(secs: int) -> str:
    parts = []
    for name, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        n, secs = divmod(secs, size)
        if n:
            parts.append(f"{n} {name}{'s' if n != 1 else ''}")
    return " and ".join(parts[:2]) or "0 seconds"


# ---------------------------------------------------------------- mac helpers

def osascript(script: str, timeout: int = 10) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "AppleScript failed")
    return r.stdout.strip()


def as_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def installed_apps() -> dict[str, str]:
    apps = {}
    for folder in ("/Applications", "/System/Applications", "/System/Applications/Utilities",
                   str(Path.home() / "Applications"), "/Applications/Utilities"):
        p = Path(folder)
        if p.is_dir():
            for app in p.glob("*.app"):
                apps[app.stem.lower()] = app.stem
    return apps


def notify(title: str, text: str) -> None:
    try:
        osascript(f"display notification {as_str(text)} with title {as_str(title)} sound name \"Glass\"")
    except Exception:
        pass


# ---------------------------------------------------------------- live info helpers

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Safari/605.1.15"}

WEATHER_CODES = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


async def geocode(client: httpx.AsyncClient, place: str | None) -> tuple[float, float, str] | None:
    if place:
        r = await client.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": place, "count": 1})
        res = r.json().get("results") or []
        if res:
            return res[0]["latitude"], res[0]["longitude"], res[0]["name"]
    # No city known: estimate from the internet connection.
    r = await client.get("https://ipinfo.io/json", headers=UA)
    data = r.json()
    if "loc" in data:
        lat, lon = map(float, data["loc"].split(","))
        return lat, lon, data.get("city", "your area")
    return None


async def weather(place: str | None, tomorrow: bool) -> str:
    async with httpx.AsyncClient(timeout=8) as c:
        loc = await geocode(c, place)
        if not loc:
            return "I couldn't work out where you are. Tell me your city, for example: I live in London."
        lat, lon, name = loc
        r = await c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
            "current": "temperature_2m,weather_code,apparent_temperature",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
        })
        w = r.json()
    d = w["daily"]
    i = 1 if tomorrow else 0
    desc = WEATHER_CODES.get(d["weather_code"][i], "mixed weather")
    hi, lo, rain = round(d["temperature_2m_max"][i]), round(d["temperature_2m_min"][i]), d["precipitation_probability_max"][i]
    rain_txt = f" Chance of rain {rain} percent." if rain is not None and rain >= 20 else ""
    if tomorrow:
        return f"Tomorrow in {name}: {desc}, between {lo} and {hi} degrees.{rain_txt}"
    cur = w["current"]
    now_desc = WEATHER_CODES.get(cur["weather_code"], desc)
    return (f"It's {round(cur['temperature_2m'])} degrees and {now_desc} in {name}. "
            f"Today's high is {hi}, low {lo}.{rain_txt}")


async def headlines(n: int = 5) -> str:
    async with httpx.AsyncClient(timeout=8, headers=UA) as c:
        r = await c.get("https://feeds.bbci.co.uk/news/rss.xml")
    titles = re.findall(r"<item>.*?<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", r.text, re.S)
    titles = [html.unescape(t).strip() for t in titles][:n]
    if not titles:
        return "I couldn't fetch the news just now."
    return "Here are the top headlines from the BBC. " + " ".join(t.rstrip(".") + "." for t in titles)


async def web_search(query: str, n: int = 5) -> list[dict]:
    """DuckDuckGo's HTML results page: free, no key."""
    async with httpx.AsyncClient(timeout=10, headers=UA, follow_redirects=True) as c:
        r = await c.post("https://html.duckduckgo.com/html/", data={"q": query})
    results = []
    for block in re.findall(r'<div class="result results_links.*?</div>\s*</div>', r.text, re.S):
        title = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.S)
        snippet = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        if title:
            clean = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
            results.append({"title": clean(title.group(1)), "snippet": clean(snippet.group(1)) if snippet else ""})
        if len(results) >= n:
            break
    return results


# ---------------------------------------------------------------- calendar and mail (local apps)

CALENDAR_SCRIPT = """
set startD to (current date) + ({offset} * days)
set time of startD to 0
set endD to startD + ({span} * days)
set out to ""
tell application "Calendar"
  repeat with c in calendars
    try
      set evs to (every event of c whose start date >= startD and start date < endD)
      repeat with e in evs
        set out to out & ((start date of e) as «class isot» as string) & "|" & (allday event of e) & "|" & (summary of e) & linefeed
      end repeat
    end try
  end repeat
end tell
return out
"""


def calendar_events(offset_days: int = 0, span_days: int = 1) -> list[tuple[dt.datetime, bool, str]]:
    raw = osascript(CALENDAR_SCRIPT.format(offset=offset_days, span=span_days), timeout=30)
    events = []
    for line in raw.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            try:
                events.append((dt.datetime.fromisoformat(parts[0]), parts[1] == "true", parts[2].strip()))
            except ValueError:
                continue
    return sorted(events)


def describe_events(events, when: str) -> str:
    if not events:
        return f"You've got nothing on your calendar {when}."
    items = []
    for start, allday, title in events[:8]:
        items.append(f"{title} all day" if allday else f"{title} at {start.strftime('%-I:%M %p').replace(':00 ', ' ')}")
    count = len(events)
    return f"You have {count} thing{'s' if count != 1 else ''} {when}: " + "; ".join(items) + "."


MAIL_SCRIPT = """
tell application "Mail"
  set out to (unread count of inbox) as string & linefeed
  set msgs to messages 1 thru {n} of inbox
  repeat with m in msgs
    set out to out & "<<MSG>>" & (sender of m) & "|" & (subject of m) & "|" & ((date received of m) as «class isot» as string) & "|" & (read status of m) & linefeed
    {body}
  end repeat
end tell
return out
"""


def latest_mail(n: int = 5, with_body: bool = False) -> tuple[int, list[dict]]:
    body = ""
    if with_body:
        body = ('set c to (content of m) as string\n'
                '    if length of c > 1500 then set c to text 1 thru 1500 of c\n'
                '    set out to out & "<<BODY>>" & c & linefeed')
    script = MAIL_SCRIPT.format(n=n, body=body)
    raw = osascript(script, timeout=40)
    lines = raw.split("\n", 1)
    unread = int(lines[0].strip() or 0)
    msgs = []
    for chunk in (lines[1] if len(lines) > 1 else "").split("<<MSG>>")[1:]:
        head, _, body_txt = chunk.partition("<<BODY>>")
        f = head.strip().split("|")
        if len(f) >= 4:
            sender = re.sub(r"\s*<.*?>", "", f[0]).strip().strip('"')
            msgs.append({"from": sender, "subject": f[1], "date": f[2], "unread": f[3] == "false",
                         "body": body_txt.strip()})
    return unread, msgs


# ---------------------------------------------------------------- the router

class Skills:
    def __init__(self, root: Path, announce: Callable[[str], Awaitable[None]],
                 allowed: Callable[[str], bool] = lambda name: False) -> None:
        self.memory = Memory(root / "memory.json")
        self.allowed = allowed  # allowed("calendar") / allowed("mail"), set in settings
        self.announce = announce  # speak something unprompted (timers)
        self.timers: list[asyncio.Task] = []

    def system_notes(self) -> str:
        now = dt.datetime.now()
        notes = [f"Current local date and time: {now.strftime('%A %-d %B %Y, %-I:%M %p')}."]
        if self.memory.facts:
            notes.append("Things you remember about the user: " + "; ".join(self.memory.facts) + ".")
        return " ".join(notes)

    def preferred_name(self) -> str | None:
        return self.memory.fact_value(r"(?:wants to be called|name is) (.+)")

    async def handle(self, text: str) -> Result | None:
        t = text.lower().strip().rstrip(".!?")
        for fn in (self.memory_skill, self.timer_skill, self.reminder_skill, self.app_skill,
                   self.volume_skill, self.music_skill, self.shortcut_skill, self.time_skill,
                   self.calendar_skill, self.mail_skill,
                   self.weather_skill, self.news_skill, self.search_skill):
            try:
                res = await fn(t, text)
            except Exception as e:
                log.exception("skill %s failed", fn.__name__)
                return Result(reply=f"Sorry, that didn't work: {e}")
            if res:
                return res
        for fact in auto_facts(text):
            self.memory.remember(fact)
        return None

    # -- memory
    async def memory_skill(self, t: str, raw: str) -> Result | None:
        m = re.match(r"^(?:please )?remember (?:that )?(.+)", raw.strip(), re.I)
        if m:
            fact = m.group(1).strip()
            fact = re.sub(r"^\bI\b", "The user", fact)
            fact = re.sub(r"^\bmy\b", "The user's", fact, flags=re.I)
            self.memory.remember(fact)
            return Result(reply="Got it. I'll remember that.")
        if re.search(r"what do you (?:remember|know) about me|what have you remembered", t):
            if not self.memory.facts:
                return Result(reply="Nothing yet. Tell me to remember something and I will.")
            return Result(context="List these remembered facts back to the user naturally: " + "; ".join(self.memory.facts))
        m = re.match(r"^forget (?:about )?(.+)", t)
        if m:
            n = self.memory.forget(m.group(1))
            if m.group(1) in ("everything", "all", "it all", "all of it"):
                return Result(reply="Done. I've forgotten everything.")
            return Result(reply=f"Forgotten." if n else "I didn't have anything about that.")
        return None

    # -- timers
    async def timer_skill(self, t: str, raw: str) -> Result | None:
        if not re.search(r"\b(timer|countdown)\b", t):
            return None
        if re.search(r"\b(cancel|stop|clear)\b", t):
            n = sum(1 for task in self.timers if not task.done() and task.cancel())
            return Result(reply="Timer cancelled." if n else "There's no timer running.")
        secs = parse_duration(t)
        if not secs:
            return Result(reply="How long should the timer be?")
        label = say_duration(secs)

        async def ring() -> None:
            await asyncio.sleep(secs)
            notify("Jarvis", f"Your timer for {label} is done.")
            try:
                subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
            except OSError:
                pass
            try:
                await self.announce(f"Your timer for {label} is done.")
            except Exception:
                log.exception("timer announce failed")

        self.timers = [x for x in self.timers if not x.done()] + [asyncio.create_task(ring())]
        return Result(reply=f"Timer set for {label}.")

    # -- reminders (Reminders app)
    async def reminder_skill(self, t: str, raw: str) -> Result | None:
        m = re.search(r"\bremind me (?:to |about |that )?(.+)", raw, re.I)
        if not m:
            return None
        rest = m.group(1).strip().rstrip(".!?")
        when = parse_when(rest)
        what = re.sub(r"\b(in \d+.*|in (?:a|an|half|one|two|three|five|ten|fifteen|twenty|thirty)\b.*|at \d.*|tomorrow.*)$", "", rest, flags=re.I).strip() or rest
        what = what[0].upper() + what[1:]
        props = f"name:{as_str(what)}"
        script = f'tell application "Reminders" to make new reminder with properties {{{props}}}'
        if when:
            delta = int((when - dt.datetime.now()).total_seconds())
            script = (f"set d to (current date) + {delta}\n"
                      f'tell application "Reminders" to make new reminder with properties '
                      f"{{{props}, remind me date:d, due date:d}}")
        await asyncio.get_running_loop().run_in_executor(None, osascript, script)
        return Result(reply=f"I'll remind you to {what[0].lower() + what[1:]} {say_time(when)}." if when
                      else f"Added to your reminders: {what}.")

    # -- apps
    async def app_skill(self, t: str, raw: str) -> Result | None:
        m = re.match(r"^(?:please )?(open|launch|start|quit|close) (?:the )?(?:app )?(.+?)(?: app)?$", t)
        if not m:
            return None
        verb, name = m.group(1), m.group(2)
        apps = installed_apps()
        match = apps.get(name) or next(iter(difflib.get_close_matches(name, apps.keys(), n=1, cutoff=0.6)), None)
        if match and match in apps:
            match = apps[match]
        if not match:
            return None  # not an app; let the LLM handle it
        if verb in ("quit", "close"):
            osascript(f"tell application {as_str(match)} to quit")
            return Result(reply=f"Closed {match}.")
        subprocess.run(["open", "-a", match], check=True)
        return Result(reply=f"Opening {match}.")

    # -- volume
    async def volume_skill(self, t: str, raw: str) -> Result | None:
        if not re.search(r"\b(volume|mute|unmute|louder|quieter|softer)\b", t):
            return None
        if "unmute" in t:
            osascript("set volume without output muted")
            return Result(reply="Sound's back on.")
        if re.search(r"\bmute\b", t):
            osascript("set volume with output muted")
            return Result(reply="")
        cur = int(osascript("output volume of (get volume settings)") or 50)
        m = re.search(r"(\d{1,3})\s*(%|percent)?", t)
        if m and re.search(r"\b(to|at|set)\b", t):
            new = int(m.group(1))
        elif re.search(r"\b(up|louder|increase|raise)\b", t):
            new = cur + 15
        elif re.search(r"\b(down|quieter|softer|lower|decrease)\b", t):
            new = cur - 15
        elif re.search(r"\b(max|maximum|full)\b", t):
            new = 100
        else:
            return Result(reply=f"The volume is at {cur} percent.")
        new = max(0, min(100, new))
        osascript(f"set volume output volume {new} without output muted")
        return Result(reply=f"Volume {new} percent.")

    # -- music
    async def music_skill(self, t: str, raw: str) -> Result | None:
        m = re.match(r"^(play|pause|resume|stop|skip|next|previous)(?: the)?(?: (?:music|song|track|it))?$", t)
        if not m:
            return None
        player = "Music"
        try:
            if osascript('application "Spotify" is running') == "true":
                player = "Spotify"
        except Exception:
            pass
        cmd = {"play": "play", "resume": "play", "pause": "pause", "stop": "pause",
               "skip": "next track", "next": "next track", "previous": "previous track"}[m.group(1)]
        osascript(f"tell application {as_str(player)} to {cmd}")
        return Result(reply="")

    # -- Shortcuts app
    async def shortcut_skill(self, t: str, raw: str) -> Result | None:
        m = re.match(r"^(?:run|start) (?:the |my )?(?:shortcut )?(.+?)(?: shortcut)?$", t)
        if not m or not ("shortcut" in t or t.startswith("run ")):
            return None
        names = subprocess.run(["shortcuts", "list"], capture_output=True, text=True).stdout.splitlines()
        lookup = {n.lower(): n for n in names}
        best = difflib.get_close_matches(m.group(1), lookup.keys(), n=1, cutoff=0.6)
        if not best:
            return Result(reply=f"I couldn't find a shortcut called {m.group(1)}.")
        name = lookup[best[0]]
        subprocess.Popen(["shortcuts", "run", name])
        return Result(reply=f"Running {name}.")

    # -- time and date (instant)
    async def time_skill(self, t: str, raw: str) -> Result | None:
        now = dt.datetime.now()
        if re.fullmatch(r"(what(?:'s| is) the time|what time is it)( (right )?now)?", t):
            return Result(reply=f"It's {now.strftime('%-I:%M %p')}.")
        if re.fullmatch(r"(what(?:'s| is) (?:the date|today(?:'s date)?)|what day is it)( today)?", t):
            return Result(reply=f"It's {now.strftime('%A, %-d %B')}.")
        return None

    # -- calendar
    async def calendar_skill(self, t: str, raw: str) -> Result | None:
        if not re.search(r"\b(calendar|schedule|agenda|meetings?|events?|what(?:'s| is| have i got) on|am i free|do i have anything)\b", t):
            return None
        if not self.allowed("calendar"):
            return Result(reply="I don't have access to your calendar. You can turn it on in my settings.")
        offset, span, when = 0, 1, "today"
        if "tomorrow" in t:
            offset, when = 1, "tomorrow"
        elif re.search(r"\bthis week\b|\bweek\b", t):
            span, when = 7, "in the next seven days"
        loop = asyncio.get_running_loop()
        events = await loop.run_in_executor(None, calendar_events, offset, span)
        if offset == 0 and span == 1:
            now = dt.datetime.now()
            events = [e for e in events if e[1] or e[0] >= now - dt.timedelta(minutes=30)]
            when = "for the rest of today"
        return Result(reply=describe_events(events, when))

    # -- mail
    async def mail_skill(self, t: str, raw: str) -> Result | None:
        if not re.search(r"\b(e-?mails?|inbox|mail)\b", t):
            return None
        if not self.allowed("mail"):
            return Result(reply="I don't have access to your email. You can turn it on in my settings.")
        loop = asyncio.get_running_loop()
        read_it = bool(re.search(r"\b(read|what does|what did|open|summar)", t))
        unread, msgs = await loop.run_in_executor(None, latest_mail, 1 if read_it else 5, read_it)
        if not msgs:
            return Result(reply="Your inbox is empty.")
        if read_it:
            m = msgs[0]
            return Result(context=(f"The user's latest email. From: {m['from']}. Subject: {m['subject']}. "
                                   f"Body (may be cut off):\n{m['body']}\n\nSummarise it in two or three spoken sentences."))
        who = "; ".join(f"{m['from']}: {m['subject']}" for m in msgs if m["unread"]) or \
            "; ".join(f"{m['from']}: {m['subject']}" for m in msgs[:3])
        lead = f"You have {unread} unread email{'s' if unread != 1 else ''}." if unread else "No unread emails."
        return Result(reply=f"{lead} Latest: {who}.")

    # -- weather
    async def weather_skill(self, t: str, raw: str) -> Result | None:
        if not re.search(r"\b(weather|forecast|temperature outside|rain|raining|snow|umbrella|how (?:hot|cold) is it)\b", t):
            return None
        m = re.search(r"\b(?:in|for|at) ([a-z][a-z .'-]+?)(?: today| tomorrow| now| this week)?$", t)
        place = m.group(1).strip() if m and m.group(1) not in ("the morning", "the evening") else None
        place = place or self.memory.fact_value(r"lives in (.+)")
        return Result(reply=await weather(place, tomorrow="tomorrow" in t))

    # -- news
    async def news_skill(self, t: str, raw: str) -> Result | None:
        if not re.search(r"\b(news|headlines)\b", t):
            return None
        m = re.search(r"\b(?:news|headlines) (?:about|on|regarding) (.+)", t)
        if m:
            results = await web_search(m.group(1) + " news")
            return Result(context=self.format_results(m.group(1), results))
        return Result(reply=await headlines())

    # -- web search: give the LLM fresh results to answer from
    async def search_skill(self, t: str, raw: str) -> Result | None:
        m = re.match(r"^(?:search(?: the web)?(?: for)?|google|look up|find out|search online for) (.+)", t)
        query = m.group(1) if m else None
        if not query and re.search(
            r"\b(latest|today's|tonight|this week|right now|current(?:ly)?|score|who won|price of|stock|"
            r"release date|recently|newest|in 20\d\d)\b", t):
            query = raw
        if not query:
            return None
        results = await web_search(query)
        if not results:
            return None
        return Result(context=self.format_results(query, results))

    @staticmethod
    def format_results(query: str, results: list[dict]) -> str:
        lines = [f"- {r['title']}: {r['snippet']}" for r in results]
        return (f"Fresh web search results for '{query}' (use them to answer; they may be partial):\n"
                + "\n".join(lines))
