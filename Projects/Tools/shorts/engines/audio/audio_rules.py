"""
Kestrel Shorts - Audio rules
============================

The checks and arithmetic for an audio plan. No LLM. The Voice Director uses them while it builds the plan
and the Reviewer can reuse them to report, so a rule lives in one place.

Plan format (version 1)
  {"version": 1, "script_hash", "theme", "arc": [...], "moments": [...], "lines": [...],
   "music": [{asset, from_beat, to_beat}], "ambience": [...], "sfx": [...], "dynamics": [...],
   "mix": {...}, "wishlist": [...], "voice_wishes": [...], "metadata": {...}}
  arc      per beat: beat_id, role, energy, tempo, music, ambience, note
  moments  sub-beat shifts: beat_id, from_cue, to_cue, energy, tempo, why, unit_ids
  lines    one entry per UNIT of speech, in script order. A unit is a sentence of a script line (split_units), so
           a long line can be spoken with a different pace, tone and pause per sentence: unit_id (e.g.
           "beat_1a.2"), section_id, beat_id, index, start, end (character span inside the script line),
           speaker, energy, pace, tone, pause_after, emphasis[], why, speed, pause_seconds, music_db,
           tone_supported
  sfx      asset, unit_id, anchor (start | end | words), words, level, db, why
  dynamics type (dip | drop_out | swell), unit_id, anchor, words, seconds, db

The script itself is never part of the plan: only section ids, and a hash that proves it was not changed.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence

from core import config
from creative.catalogs import asset_catalog as ac
from creative.catalogs import audio_catalog as au

HOLD_WORDS = 150            # one held silence per this many words, plus one
RUSHED_WORDS = 100          # one "rushed" unit per this many words, plus one
MIN_UNIT_WORDS = 3          # a sentence shorter than this is spoken together with its neighbour
_BREAK = re.compile(r"[.!?][\"')\]]*(\s+)(?=[A-Z0-9\"'(])")


def script_hash(sections: Sequence[Any]) -> str:
    digest = hashlib.sha1()
    for s in sections:
        digest.update(f"{s.section_id}|{s.text}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def contains(text: str, phrase: str) -> bool:
    phrase = str(phrase).strip().lower()
    return bool(phrase) and phrase in str(text).lower()


def split_units(sections: Sequence[Any]) -> List[Dict[str, Any]]:
    """The units of speech: every script line split into sentences (very short ones join their neighbour).
    [{unit_id, section_id, beat_id, index, start, end, text}], where start/end are character offsets in the line."""
    units: List[Dict[str, Any]] = []
    for s in sections:
        text = s.text
        cuts = [0] + [m.end(1) for m in _BREAK.finditer(text)]
        spans = [[a, cuts[i + 1] if i + 1 < len(cuts) else len(text)] for i, a in enumerate(cuts)]
        merged: List[List[int]] = []
        for span in spans:
            if merged and len(text[span[0]:span[1]].split()) < MIN_UNIT_WORDS:
                merged[-1][1] = span[1]
            else:
                merged.append(span)
        if len(merged) > 1 and len(text[merged[0][0]:merged[0][1]].split()) < MIN_UNIT_WORDS:
            merged[1][0] = merged[0][0]
            del merged[0]
        for n, (a, b) in enumerate(merged, 1):
            end = a + len(text[a:b].rstrip())
            units.append({"unit_id": f"{s.section_id}.{n}", "section_id": s.section_id, "beat_id": s.beat_id or "",
                          "index": n, "start": a, "end": end, "text": text[a:end]})
    return units


def max_rushed(total_words: int) -> int:
    return 1 + total_words // RUSHED_WORDS


def line_seconds(words: int, pace: str, pause: str) -> float:
    return words / config.WORDS_PER_SECOND / au.PACE_SPEED.get(pace, 1.0) + au.PAUSE_SECONDS.get(pause, 0.0)


def estimate_seconds(lines: List[Dict[str, Any]], words_of: Dict[str, int]) -> float:
    return sum(line_seconds(words_of.get(l["unit_id"], 0), l["pace"], l["pause_after"]) for l in lines)


def sfx_cap(seconds: float, density: str) -> int:
    interval = au.SFX_INTERVAL_SECONDS.get(density, au.SFX_INTERVAL_SECONDS["medium"])
    return max(1, int(round(seconds / interval)))


def max_holds(total_words: int) -> int:
    return 1 + total_words // HOLD_WORDS


def segments(arc: List[Dict[str, Any]], key: str) -> List[Dict[str, str]]:
    """Runs of consecutive beats that use the same asset (music or ambience): one continuing bed."""
    runs: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for entry in arc:
        asset = entry.get(key) or ""
        if current and current["asset"] == asset:
            current["to_beat"] = entry["beat_id"]
            continue
        current = {"asset": asset, "from_beat": entry["beat_id"], "to_beat": entry["beat_id"]}
        runs.append(current)
    return [r for r in runs if r["asset"]]


def check_plan(plan: Dict[str, Any], sections: Sequence[Any], density: str = "medium") -> List[Dict[str, str]]:
    """Every problem an audio plan has: [{severity, check, beat_id, unit_id, problem, suggestion}]."""
    issues: List[Dict[str, str]] = []

    def add(severity: str, check: str, problem: str, unit_id: str = "", beat_id: str = "",
            suggestion: str = "") -> None:
        issues.append({"severity": severity, "check": check, "beat_id": beat_id, "unit_id": unit_id,
                       "problem": problem, "suggestion": suggestion})

    units = split_units(sections)
    text_of = {u["unit_id"]: u["text"] for u in units}
    beat_of = {u["unit_id"]: u["beat_id"] for u in units}
    if plan.get("script_hash") != script_hash(sections):
        add("error", "script", "The script changed after the audio plan was made.",
            suggestion="Run the Voice Director again.")
    lines = plan.get("lines", [])
    if [l.get("unit_id") for l in lines] != [u["unit_id"] for u in units]:
        add("error", "lines", "The plan does not cover every sentence of the script exactly once, in order.")
    holds = rushed = 0
    for line in lines:
        uid = line.get("unit_id", "")
        text, beat = text_of.get(uid, ""), beat_of.get(uid, "")
        if line.get("pace") not in au.PACES:
            add("error", "line", f"Unknown pace {line.get('pace')!r}.", uid, beat)
        if line.get("tone") not in au.TONES:
            add("error", "line", f"Unknown tone {line.get('tone')!r}.", uid, beat)
        if line.get("pause_after") not in au.PAUSES:
            add("error", "line", f"Unknown pause {line.get('pause_after')!r}.", uid, beat)
        for word in line.get("emphasis") or []:
            if not contains(text, word):
                add("warning", "emphasis", f"Emphasis {word!r} is not in the sentence.", uid, beat)
        holds += line.get("pause_after") == "hold"
        rushed += line.get("pace") == "rushed"
    run = longest = 0
    previous = None
    for line in lines:
        key = (line.get("tone"), line.get("pace"))
        run = run + 1 if key == previous else 1
        previous = key
        longest = max(longest, run)
    if longest >= 4:
        add("warning", "delivery", f"{longest} sentences in a row are spoken the same way; vary the delivery.")
    total_words = sum(len(t.split()) for t in text_of.values())
    if holds > max_holds(total_words):
        add("warning", "pause",
            f"{holds} held silences in {total_words} words; at most {max_holds(total_words)} keep their weight.")
    if rushed > max_rushed(total_words):
        add("warning", "pace", f"{rushed} rushed sentences in {total_words} words; use it for a real burst only.")
    if lines and lines[-1].get("pause_after") not in (None, "none"):
        add("warning", "pause", "The last sentence ends with a pause; the video should end on the voice.",
            lines[-1].get("unit_id", ""))
    for kind in ("music", "ambience"):
        for run in plan.get(kind, []):
            if not ac.exists(kind, run.get("asset", "")):
                add("error", kind, f"{kind} {run.get('asset')!r} is not in the library.")
    for event in plan.get("sfx", []):
        uid = event.get("unit_id", "")
        if not ac.exists("sfx", event.get("asset", "")):
            add("error", "sfx", f"Sound effect {event.get('asset')!r} is not in the library.", uid, beat_of.get(uid, ""))
        if event.get("anchor") == "words" and not contains(text_of.get(uid, ""), event.get("words", "")):
            add("error", "sfx", f"Sound anchor {event.get('words')!r} is not in {uid}.", uid, beat_of.get(uid, ""))
    for event in plan.get("dynamics", []):
        uid = event.get("unit_id", "")
        if event.get("type") not in au.DYNAMICS:
            add("error", "dynamics", f"Unknown dynamic {event.get('type')!r}.", uid, beat_of.get(uid, ""))
        if event.get("anchor") == "words" and not contains(text_of.get(uid, ""), event.get("words", "")):
            add("error", "dynamics", f"Dynamic anchor {event.get('words')!r} is not in {uid}.", uid, beat_of.get(uid, ""))
    seconds = float(plan.get("metadata", {}).get("estimated_seconds", 0))
    if len(plan.get("sfx", [])) > sfx_cap(seconds or 1, density) + len(set(beat_of.values())):
        add("warning", "sfx", "More sound effects than the theme's density allows.")
    if seconds > config.MAX_VIDEO_SECONDS:
        add("error", "length", f"With this delivery the video runs about {seconds:.0f} s; "
                               f"the limit is {config.MAX_VIDEO_SECONDS:.0f} s.")
    return issues