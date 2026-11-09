"""
Kestrel Shorts - Voice Director
===============================

Decides what the viewer HEARS, from the final approved script: how every line is spoken, the music and
background sound under it, sound effects, and the dynamics of the soundtrack (rising, dipping, going silent
before a reveal). It never changes a word of the script: the plan holds only section ids, and a hash proves
the script was untouched. Voice decides how long the video really is, so the Frame Supervisor later fits the
visuals to this audio, the way a film is cut to its music.

Two kinds of LLM call, small each:
  1. ONE arc call that sees the whole video (beats, roles, intents, what is on screen): per beat an energy
     (hush, calm, steady, rising, intense, peak) and a tempo, a music bed and an ambience bed from the library,
     and up to three "moments": stretches INSIDE a beat that need a different energy or tempo (a sub-climax,
     a sudden quiet).
  2. ONE call per beat that sees its sentences, the arc around it and the shots on screen: per sentence a pace,
     a tone, a pause after it and the words to lean on; sound effects from the library anchored to a sentence;
     dynamics. (A script line can hold several sentences, so a long line is not forced into one delivery.)

Code owns every number (speed, seconds, decibels, how many effects the theme allows) and every repair, see
creative/catalogs/audio_catalog.py and engines/audio/audio_rules.py. A long pause is part of the expression:
it is kept only after a line whose tone, emphasis or energy earns it. Tone needs a reference clip per tone in
the character's voice profile (a `styles` map); a tone without a clip is recorded as a voice wish, and the
engine speaks the plain voice until the clip exists. Sounds the library lacks go to the asset wishlist.

    python -m agents.voice_director --from-run <path to shared_context.json>
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from agents._common import read_brief
from core import config, llm
from core.models.script import Script, ScriptBeat
from core.registry_loader import RegistryLoader
from creative.catalogs import asset_catalog as ac
from creative.catalogs import audio_catalog as au
from engines.audio import audio_rules as ar

logger = logging.getLogger(__name__)

STAGE = "voice_director"
PAUSE_ORDER = list(au.PAUSES)           # none, short, medium, long, hold: increasing
MENU_LIMIT = 60


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _beat_text(beat: ScriptBeat) -> str:
    return " ".join(s.text for s in beat.sections)


def _short(text: str, words: int = 12) -> str:
    parts = str(text).split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def _menu(kind: str) -> Tuple[List[str], str]:
    found = ac.scan(kind)
    names = list(found)[:MENU_LIMIT]
    text = "\n".join(f"  {n}" + (f": {found[n]['hint']}" if found[n].get("hint") else "") for n in names)
    return names, text or "  (none)"


def _enum(values: Any) -> Dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _unit_spans(beat: ScriptBeat, units: List[Dict[str, Any]]) -> List[Tuple[str, int, int]]:
    """Each unit's character span inside the beat's narration (its lines joined by one space)."""
    offsets, offset = {}, 0
    for s in beat.sections:
        offsets[s.section_id] = offset
        offset += len(s.text) + 1
    return [(u["unit_id"], offsets[u["section_id"]] + u["start"], offsets[u["section_id"]] + u["end"]) for u in units]


def _clean_wish(text: Any) -> str:
    """A wish is a few words about a sound the library lacks; models also write 'none needed' here."""
    text = " ".join(str(text or "").split()).strip(" .\"'")
    low = text.lower()
    if not text or low.startswith(("none", "no ", "n/a", "nothing", "not ")) or len(text.split()) > 10:
        return ""
    return text


def _unquote(cue: Any) -> str:
    return str(cue or "").strip().strip("\"'\u201c\u201d\u2018\u2019").strip()


# ----------------------------------------------------------------------
# Stage 1: the arc of the whole video
# ----------------------------------------------------------------------

def _arc_schema(beat_ids: List[str], music: List[str], ambience: List[str]) -> Dict[str, Any]:
    text = {"type": "string"}
    entry = {"type": "object", "properties": {
        "beat_id": _enum(beat_ids), "energy": _enum(au.ENERGIES), "tempo": _enum(au.PACES),
        "music": _enum([""] + music), "ambience": _enum([""] + ambience),
        "music_wish": text, "note": text},
        "required": ["beat_id", "energy", "tempo", "note"]}
    moment = {"type": "object", "properties": {
        "beat_id": _enum(beat_ids), "from_cue": text, "to_cue": text,
        "energy": _enum(au.ENERGIES), "tempo": _enum(au.PACES), "why": text},
        "required": ["beat_id", "from_cue", "energy", "why"]}
    return {"type": "object", "properties": {
        "arc": {"type": "array", "items": entry, "minItems": len(beat_ids), "maxItems": len(beat_ids)},
        "moments": {"type": "array", "items": moment, "maxItems": 3}},
        "required": ["arc"]}


def _arc_prompt(info: Dict[str, Any], theme: Dict[str, Any], beats: List[ScriptBeat],
                shots: Dict[str, List[Dict[str, Any]]], music_menu: str, ambience_menu: str) -> str:
    audio = theme.get("audio") or {}
    moods = ", ".join(audio.get("music_mood", [])) or "not specified"
    hints = " ".join(str(theme.get("format_hints", "")).split()) or "not specified"
    outline = []
    for b in beats:
        first = b.sections[0]
        on_screen = "; ".join(_short(s.get("intent", ""), 8) for s in shots.get(b.beat_id, [])[:3])
        outline.append(f"  {b.beat_id} [{b.role}] \"{b.title}\" ({sum(len(s.text.split()) for s in b.sections)} words)"
                       f" - {first.narrative_purpose or b.title}" + (f" | on screen: {on_screen}" if on_screen else ""))
    return f"""Task: plan the SOUND ARC of a short video from its final script: how the energy of the voice and music rises and falls from beat to beat. You choose words; code handles every number.

Topic: {info['topic']}
Audience: {info['audience']}
Video format: {hints}
Music mood of this channel: {moods}

The video, beat by beat:
{chr(10).join(outline)}

Music in the library (use only these names):
{music_menu}
Ambience in the library (use only these names):
{ambience_menu}

Energies: {au.help_text(au.ENERGIES)}.
Tempos: {au.help_text(au.PACES)}.

Rules:
- Give EVERY beat an energy and a tempo.
- Energy follows the CONTENT, not a formula. Shape the video like a film score: use contrast, and pull the sound DOWN (hush) just before a reveal, the way a horror film goes quiet before the ghost appears. An explainer may stay steady with a gentle rise; a conclusion usually settles (calm or steady) unless it is itself the reveal. Not every video needs a peak, and a peak only works after quieter beats. Tempo rushed is for a real burst of a few words, never a whole beat.
- moments are OPTIONAL and rare: leave the list empty unless a stretch inside a beat clearly differs (a sub-climax, a sudden quiet). from_cue and to_cue are short phrases copied exactly from that beat's NARRATION (never its title); from_cue "" = its start, to_cue "" = its end.
- music: ONE bed per run of beats: the same name on consecutive beats means the music continues. Change it only when the mood changes. "" means no music. ambience: the same idea for a background sound.
- If nothing in a library fits, leave "" and say what you wanted in music_wish (a few words).
- note: up to 10 words on why.
Return JSON only."""


def _validate_arc(data: Dict[str, Any], beats: List[ScriptBeat], units_by_beat: Dict[str, List[Dict[str, Any]]],
                  music: List[str], ambience: List[str]) -> Dict[str, Any]:
    entries: Dict[str, Dict[str, Any]] = {}
    for e in data.get("arc", []):
        if isinstance(e, dict) and e.get("beat_id") in {b.beat_id for b in beats} and e["beat_id"] not in entries:
            entries[e["beat_id"]] = e
    if not entries:
        raise ValueError("arc must contain one entry per beat.")
    notes: List[str] = []
    arc: List[Dict[str, Any]] = []
    wishes: List[str] = []
    for b in beats:
        e = entries.get(b.beat_id)
        energy = (e or {}).get("energy")
        if energy not in au.ENERGIES:
            energy = au.ROLE_ENERGY.get(b.role, "steady")
            notes.append(f"{b.beat_id}: no energy given, using {energy} for a {b.role} beat")
        tempo = (e or {}).get("tempo") if (e or {}).get("tempo") in au.PACES else au.ENERGY_TEMPO[energy]
        arc.append({"beat_id": b.beat_id, "role": b.role, "energy": energy, "tempo": tempo,
                    "music": (e or {}).get("music") if (e or {}).get("music") in music else "",
                    "ambience": (e or {}).get("ambience") if (e or {}).get("ambience") in ambience else "",
                    "note": str((e or {}).get("note", "")).strip()})
        wish = _clean_wish((e or {}).get("music_wish", ""))
        if wish and not arc[-1]["music"]:
            wishes.append(wish)

    by_id = {b.beat_id: b for b in beats}
    moments: List[Dict[str, Any]] = []
    for m in data.get("moments", []) or []:
        if not isinstance(m, dict) or m.get("beat_id") not in by_id or m.get("energy") not in au.ENERGIES:
            continue
        beat = by_id[m["beat_id"]]
        lowered = _beat_text(beat).lower()
        cue_from, cue_to = _unquote(m.get("from_cue")), _unquote(m.get("to_cue"))
        if beat.title and beat.title.strip().lower() in (cue_from.lower(), cue_to.lower()):
            notes.append(f"{beat.beat_id}: dropped a moment whose cue is the beat title, not the narration")
            continue
        start = lowered.find(cue_from.lower()) if cue_from else 0
        end = (lowered.find(cue_to.lower()) + len(cue_to)) if cue_to and cue_to.lower() in lowered else (
            len(lowered) if not cue_to else -1)
        if start < 0 or end <= start:
            notes.append(f"{beat.beat_id}: dropped a moment whose cues are not in the narration "
                         f"({cue_from!r} .. {cue_to!r})")
            continue
        tempo = m.get("tempo") if m.get("tempo") in au.PACES else au.ENERGY_TEMPO[m["energy"]]
        parent = next(a for a in arc if a["beat_id"] == beat.beat_id)
        if m["energy"] == parent["energy"] and tempo == parent["tempo"]:
            continue                                    # no change: not a moment
        covered = [uid for uid, a, b in _unit_spans(beat, units_by_beat[beat.beat_id]) if a < end and b > start]
        moments.append({"beat_id": beat.beat_id, "from_cue": cue_from, "to_cue": cue_to, "energy": m["energy"],
                        "tempo": tempo, "why": str(m.get("why", "")).strip(), "unit_ids": covered})
    return {"arc": arc, "moments": moments[:3], "notes": notes, "wishes": wishes}


# ----------------------------------------------------------------------
# Stage 2: one beat, sentence by sentence
# ----------------------------------------------------------------------

def _beat_schema(unit_ids: List[str], sfx_names: List[str], sfx_cap: int) -> Dict[str, Any]:
    text = {"type": "string"}
    line = {"type": "object", "properties": {
        "unit_id": _enum(unit_ids), "pace": _enum(au.PACES), "tone": _enum(au.TONES),
        "pause_after": _enum(au.PAUSES), "emphasis": {"type": "array", "items": text, "maxItems": 2}, "why": text},
        "required": ["unit_id", "pace", "tone", "pause_after"]}
    anchor = {"anchor": _enum(au.ANCHORS), "words": text}
    properties: Dict[str, Any] = {
        "lines": {"type": "array", "items": line, "minItems": len(unit_ids), "maxItems": len(unit_ids)},
        "dynamics": {"type": "array", "maxItems": au.MAX_DYNAMICS_PER_BEAT, "items": {"type": "object", "properties": {
            "type": _enum(au.DYNAMICS), "unit_id": _enum(unit_ids), **anchor,
            "duration": _enum(au.DYNAMIC_SECONDS)}, "required": ["type", "unit_id", "anchor"]}},
        "sfx_wish": text}
    if sfx_names and sfx_cap > 0:
        properties["sfx"] = {"type": "array", "maxItems": sfx_cap, "items": {"type": "object", "properties": {
            "asset": _enum(sfx_names), "unit_id": _enum(unit_ids), **anchor, "level": _enum(au.LEVELS), "why": text},
            "required": ["asset", "unit_id", "anchor"]}}
    return {"type": "object", "properties": properties, "required": ["lines"]}


def _beat_prompt(info, theme, beat, prev_beat, next_beat, arc, defaults, units, shots, characters, sfx_menu,
                 sfx_cap) -> str:
    here = next(a for a in arc if a["beat_id"] == beat.beat_id)
    curve = " -> ".join(f"{a['beat_id']}:{a['energy']}" + ("*" if a["beat_id"] == beat.beat_id else "") for a in arc)
    speaker = {s.section_id: s.speaker_character_id or "narrator" for s in beat.sections}
    bed = ", ".join(x for x in (here["music"], here["ambience"]) if x) or "none playing"
    lines = "\n".join(
        f"  {u['unit_id']} [{speaker[u['section_id']]} | default {defaults[u['unit_id']][0]}, "
        f"{defaults[u['unit_id']][1]}] \"{u['text']}\"" for u in units)
    on_screen = "\n".join(f"  from \"{_short(s.get('cue', ''), 6) or 'the start'}\": {s.get('intent', '')}"
                          for s in shots) or "  (not planned)"
    tones = "\n".join(f"  {cid}: real recordings for {', '.join(sorted(ts))}" for cid, ts in characters.items()) or "  (none)"
    sfx_rule = (f"- sfx: AT MOST {sfx_cap}, and none is often right: use a sound only when the words call for it, and never "
                f"the same sound twice in a row. anchor: start or end of a sentence, or words (exact words of that sentence "
                f"where the sound lands). level: quiet, low or medium. Reactions (a chuckle, a gasp) go at the start or end "
                f"of a sentence. Silence is also a choice."
                if sfx_cap > 0 else "- sfx: none for this beat.")
    return f"""Task: direct the VOICE and SOUND of ONE beat of a short video: how each sentence is spoken, and which sound effects and dynamics support it. The words are fixed: you never change them.

Topic: {info['topic']}
Sound arc of the whole video (energy per beat, * marks this one): {curve}
This beat: {beat.beat_id} [{beat.role}] "{beat.title}" - energy {here['energy']}, tempo {here['tempo']}. {here['note']}
Previous beat: {prev_beat or '(this is the first)'}
Next beat: {next_beat or '(this is the last)'}
Sentences (id [speaker | default energy, pace] "text"):
{lines}
What is on screen in this beat:
{on_screen}

Characters and the tones they can really speak (other tones sound plain until a clip exists, so use them only when a sentence truly needs them):
{tones}
Sound effects in the library (use only these):
{sfx_menu}

Paces: {au.help_text(au.PACES)}.
Tones: {au.help_text(au.TONES)}.
Pauses: {au.help_text(au.PAUSES)}.
Dynamics: {au.help_text(au.DYNAMICS)}.

Rules:
- Every sentence gets a pace, a tone and a pause_after. Judge each sentence on its own: a line often holds several, so vary them instead of one delivery for the whole line. Start from the default pace; deviate when the words ask for it. Most sentences are neutral or warm, with pause none or short: expression works by contrast.
- Real questions rise (curious). A discovery or surprise is excited. A heavy fact is serious. Before a reveal, use hushed or dramatic with a long pause or a hold. Pace rushed is for one exciting sentence, rarely.
- A long pause or a hold is PART of the expression: use one only after a sentence that earns it (a question that should land, a reveal, a heavy fact). Never at the end of the video.
- emphasis: up to two exact words from the sentence the speaker leans on, [] for none. why: up to 8 words.
{sfx_rule}
- dynamics: at most {au.MAX_DYNAMICS_PER_BEAT} for this beat, and usually none. They act on the music and ambience, so skip them when none plays here (playing here: {bed}). Use one only where the story needs it (tension before a reveal, a payoff). anchor works as for sfx.
- sfx_wish: only if a sound would really help and the library has none: a few words describing it. Otherwise "".
Return JSON only."""


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def run_voice_director(
    script: Any = None,
    context: Any = None,
    visual_plan: Optional[Dict[str, Any]] = None,
    num_predict: Optional[int] = None,
    *,
    registry: Optional[RegistryLoader] = None,
    **ignored: Any,
) -> Dict[str, Any]:
    script = Script(**script) if isinstance(script, dict) else script
    if not isinstance(script, Script) or not script.sections:
        raise ValueError("A script with sections is required.")

    info = read_brief(context)
    ctx = info["ctx"]
    registry = registry or RegistryLoader()
    channel_id = info["channel_id"]
    run_id = ctx.run_id if ctx else ""
    beats = script.beats
    units = ar.split_units(script.sections)
    units_by_beat: Dict[str, List[Dict[str, Any]]] = {b.beat_id: [] for b in beats}
    for u in units:
        units_by_beat[u["beat_id"]].append(u)

    theme: Dict[str, Any] = {}
    try:
        theme_id = registry.get_channel(channel_id).get("theme_id") if channel_id else ""
        theme = registry.get_theme(theme_id) if theme_id else {}
    except Exception as exc:
        print(f"[Voice Director] WARNING: no theme ({exc}); using plain defaults.")
    density = (theme.get("audio") or {}).get("sfx_density", "medium")

    plan_in = visual_plan if visual_plan is not None else (ctx.read_stage_output("visual_plan") if ctx else None)
    shots: Dict[str, List[Dict[str, Any]]] = {}
    for shot in (plan_in or {}).get("shots", []):
        shots.setdefault(shot.get("beat_id", ""), []).append(shot)

    # Which tones each character can really speak: a `styles` map of tone -> reference clip in the voice profile.
    characters: Dict[str, set] = {}
    for member in (registry.character_menu(channel_id) if channel_id else []):
        real = {"neutral"}
        try:
            voice_id = registry.get_character(member["id"]).get("voice_profile_id")
            real |= set((registry.get_voice(voice_id).get("styles") or {}) if voice_id else {})
        except Exception:
            pass
        characters[member["id"]] = real

    music_names, music_menu = _menu("music")
    ambience_names, ambience_menu = _menu("ambience")
    sfx_names, sfx_menu = _menu("sfx")

    # ---- stage 1: the arc ------------------------------------------------
    arc_data = llm.generate_json(
        STAGE, _arc_prompt(info, theme, beats, shots, music_menu, ambience_menu),
        schema=_arc_schema([b.beat_id for b in beats], music_names, ambience_names),
        validate=lambda d: _validate_arc(d, beats, units_by_beat, music_names, ambience_names),
        retries=1, num_predict=num_predict)
    arc, moments = arc_data["arc"], arc_data["moments"]
    for note in arc_data["notes"]:
        print(f"[Voice Director] {note}")
    for wish in arc_data["wishes"]:
        ac.request("music", au.slug(wish), wish, STAGE, run_id)

    # ---- stage 2: the sentences of each beat -----------------------------
    total_words = sum(len(s.text.split()) for s in script.sections)
    rough_seconds = total_words / config.WORDS_PER_SECOND * 1.1
    sfx_total = ar.sfx_cap(rough_seconds, density) if sfx_names else 0
    sfx_left = sfx_total
    raw_lines: Dict[str, Dict[str, Any]] = {}
    sfx_events: List[Dict[str, Any]] = []
    dynamics: List[Dict[str, Any]] = []
    for index, beat in enumerate(beats):
        entry = next(a for a in arc if a["beat_id"] == beat.beat_id)
        beat_units = units_by_beat[beat.beat_id]
        defaults: Dict[str, Tuple[str, str]] = {}
        for u in beat_units:
            moment = next((m for m in moments if u["unit_id"] in m["unit_ids"]), None)
            defaults[u["unit_id"]] = ((moment["energy"], moment["tempo"]) if moment else (entry["energy"], entry["tempo"]))
        ids = [u["unit_id"] for u in beat_units]
        text_of = {u["unit_id"]: u["text"] for u in beat_units}
        share = -(-sfx_total * sum(len(t.split()) for t in text_of.values()) // max(1, total_words))   # rounded up
        cap = max(0, min(au.SFX_PER_BEAT, sfx_left, max(1, share))) if sfx_names else 0

        def validate(data: Dict[str, Any], beat=beat, defaults=defaults, ids=ids, text_of=text_of, cap=cap) -> Dict[str, Any]:
            given = {l.get("unit_id"): l for l in data.get("lines", []) if isinstance(l, dict)}
            if not given:
                raise ValueError("lines must contain one entry per sentence of the beat.")
            notes: List[str] = []
            lines: Dict[str, Dict[str, Any]] = {}
            for uid in ids:
                l = given.get(uid)
                if l is None:
                    notes.append(f"{uid}: no direction given, plain delivery")
                    l = {}
                lines[uid] = {
                    "pace": l.get("pace") if l.get("pace") in au.PACES else defaults[uid][1],
                    "tone": l.get("tone") if l.get("tone") in au.TONES else "neutral",
                    "pause_after": l.get("pause_after") if l.get("pause_after") in au.PAUSES else "none",
                    "emphasis": [w.strip() for w in (l.get("emphasis") or []) if isinstance(w, str) and ar.contains(text_of[uid], w)][:2],
                    "why": str(l.get("why", "")).strip()}
            events: List[Dict[str, Any]] = []
            used: set = set()
            for e in (data.get("sfx") or [])[:cap]:
                if not isinstance(e, dict) or e.get("asset") not in sfx_names or e.get("unit_id") not in ids:
                    continue
                if e["unit_id"] in used:
                    continue
                anchor, words = e.get("anchor"), str(e.get("words", "")).strip()
                if anchor not in au.ANCHORS or (anchor == "words" and not ar.contains(text_of[e["unit_id"]], words)):
                    notes.append(f"{e['unit_id']}: dropped a sound effect whose anchor is not in the sentence ({words!r})")
                    continue
                used.add(e["unit_id"])
                level = e.get("level") if e.get("level") in au.LEVELS else "low"
                events.append({"asset": e["asset"], "unit_id": e["unit_id"], "anchor": anchor,
                               "words": words if anchor == "words" else "", "level": level,
                               "db": au.LEVEL_DB[level], "why": str(e.get("why", "")).strip()})
            dyn: List[Dict[str, Any]] = []
            for e in (data.get("dynamics") or [])[:au.MAX_DYNAMICS_PER_BEAT]:
                if not isinstance(e, dict) or e.get("type") not in au.DYNAMICS or e.get("unit_id") not in ids:
                    continue
                anchor, words = e.get("anchor"), str(e.get("words", "")).strip()
                if anchor not in au.ANCHORS or (anchor == "words" and not ar.contains(text_of[e["unit_id"]], words)):
                    notes.append(f"{e['unit_id']}: dropped a {e['type']} whose anchor is not in the sentence ({words!r})")
                    continue
                duration = e.get("duration") if e.get("duration") in au.DYNAMIC_SECONDS else "short"
                dyn.append({"type": e["type"], "unit_id": e["unit_id"], "anchor": anchor,
                            "words": words if anchor == "words" else "", "seconds": au.DYNAMIC_SECONDS[duration],
                            "db": au.DYNAMIC_DB[e["type"]]})
            return {"lines": lines, "sfx": events, "dynamics": dyn, "notes": notes,
                    "sfx_wish": _clean_wish(data.get("sfx_wish"))}

        prev_b = f"[{beats[index - 1].role}] {beats[index - 1].title} (energy {arc[index - 1]['energy']})" if index else ""
        next_b = (f"[{beats[index + 1].role}] {beats[index + 1].title} (energy {arc[index + 1]['energy']})"
                  if index + 1 < len(beats) else "")
        data = llm.generate_json(
            STAGE, _beat_prompt(info, theme, beat, prev_b, next_b, arc, defaults, beat_units, shots.get(beat.beat_id, []),
                                characters, sfx_menu if sfx_names else "  (none: leave sfx out)", cap),
            schema=_beat_schema(ids, sfx_names, cap), validate=validate, retries=1, num_predict=num_predict)
        raw_lines.update(data["lines"])
        sfx_events += data["sfx"]
        dynamics += data["dynamics"]
        sfx_left -= len(data["sfx"])
        for note in data["notes"]:
            print(f"[Voice Director] {beat.beat_id}: {note}")
        if data["sfx_wish"]:
            ac.request("sfx", au.slug(data["sfx_wish"]), data["sfx_wish"], STAGE, run_id)
        print(f"[Voice Director] {beat.beat_id} [{beat.role}]: energy {entry['energy']}, tempo {entry['tempo']}, "
              f"{len(data['lines'])} sentence(s), {len(data['sfx'])} effect(s), {len(data['dynamics'])} dynamic(s)")

    plan = _assemble(script, units, beats, arc, moments, raw_lines, sfx_events, dynamics, characters, theme, density)
    issues = ar.check_plan(plan, script.sections, density)
    plan["metadata"]["issues"] = issues
    m = plan["metadata"]
    errors = sum(1 for i in issues if i["severity"] == "error")
    print(f"[Voice Director] {len(plan['lines'])} sentences, about {m['estimated_seconds']:.0f} s of voice, "
          f"{len(plan['music'])} music bed(s), {len(plan['sfx'])} effect(s), {len(plan['dynamics'])} dynamic(s); "
          f"{len(m['repairs'])} repair(s) in code, {errors} error(s) and {len(issues) - errors} warning(s) left")
    return plan


# ----------------------------------------------------------------------
# Assembly: the code layer (defaults, repairs and every number)
# ----------------------------------------------------------------------

def _assemble(script, units, beats, arc, moments, raw_lines, sfx_events, dynamics, characters, theme, density):
    repairs: List[str] = []
    energy_of: Dict[str, str] = {}
    for a in arc:
        for u in units:
            if u["beat_id"] == a["beat_id"]:
                energy_of[u["unit_id"]] = a["energy"]
    for m in moments:
        for uid in m["unit_ids"]:
            energy_of[uid] = m["energy"]
    role_of = {b.beat_id: b.role for b in beats}
    section = {s.section_id: s for s in script.sections}
    last_of_beat = {}
    for u in units:
        last_of_beat[u["beat_id"]] = u["unit_id"]
    last_of_beat = {uid: bid for bid, uid in last_of_beat.items()}
    final_id = units[-1]["unit_id"]
    total_words = sum(len(s.text.split()) for s in script.sections)
    holds_left, rushed_left = ar.max_holds(total_words), ar.max_rushed(total_words)
    story_emphasis_used: set = set()

    lines: List[Dict[str, Any]] = []
    for u in units:
        uid, s = u["unit_id"], section[u["section_id"]]
        raw = raw_lines.get(uid, {})
        speaker = s.speaker_character_id or "narrator"
        energy = energy_of.get(uid, "steady")
        pace = raw.get("pace") or au.ENERGY_TEMPO[energy]
        tone = raw.get("tone") or "neutral"
        pause = raw.get("pause_after") or "none"
        emphasis = list(raw.get("emphasis") or [])
        if (s.emphasis and s.section_id not in story_emphasis_used and ar.contains(u["text"], s.emphasis)
                and not any(s.emphasis.lower() in e.lower() or e.lower() in s.emphasis.lower() for e in emphasis)):
            emphasis = (emphasis + [s.emphasis])[:2]
            story_emphasis_used.add(s.section_id)
        if pace == "rushed":
            if rushed_left <= 0:
                pace = "brisk"
                repairs.append(f"{uid}: too many rushed sentences for this length, now brisk")
            else:
                rushed_left -= 1
        if u["text"].strip().endswith("?") and tone == "neutral":
            tone = "curious"
            repairs.append(f"{uid}: a question, tone set to curious")
        if pause in ("long", "hold"):
            earned = tone in au.EXPRESSIVE_TONES or bool(emphasis) or energy in ("hush", "peak")
            if not earned:
                pause = "medium"
                repairs.append(f"{uid}: a {raw.get('pause_after')} pause without an expressive reason, now medium")
            elif pause == "hold":
                if holds_left <= 0:
                    pause = "long"
                    repairs.append(f"{uid}: too many held silences for this length, now a long pause")
                else:
                    holds_left -= 1
        beat_id = last_of_beat.get(uid)
        if beat_id and uid != final_id:
            wanted = au.BEAT_END_PAUSE.get(role_of.get(beat_id, ""), "short")
            if PAUSE_ORDER.index(pause) < PAUSE_ORDER.index(wanted):
                pause = wanted
        if uid == final_id and pause != "none":
            pause = "none"
            repairs.append(f"{uid}: the video ends on the voice, pause removed")
        supported = tone == "neutral" or tone in characters.get(speaker, {"neutral"})
        lines.append({"unit_id": uid, "section_id": u["section_id"], "beat_id": u["beat_id"], "index": u["index"],
                      "start": u["start"], "end": u["end"], "speaker": speaker, "energy": energy,
                      "pace": pace, "tone": tone, "pause_after": pause, "emphasis": emphasis,
                      "why": raw.get("why", ""), "speed": au.PACE_SPEED[pace],
                      "pause_seconds": au.PAUSE_SECONDS[pause],
                      "music_db": au.LEVEL_DB[au.ENERGY_LEVEL[energy]], "tone_supported": supported})

    kept_sfx: List[Dict[str, Any]] = []
    for e in sfx_events:                                           # the same sound twice in a row is not a choice
        if kept_sfx and kept_sfx[-1]["asset"] == e["asset"]:
            repairs.append(f"{e['unit_id']}: the same sound effect ({e['asset']}) twice in a row, dropped")
            continue
        kept_sfx.append(e)

    words_of = {u["unit_id"]: len(u["text"].split()) for u in units}
    bed_of = {a["beat_id"]: bool(a["music"] or a["ambience"]) for a in arc}
    beat_of_unit = {u["unit_id"]: u["beat_id"] for u in units}
    drop_outs = 0
    candidates: List[Dict[str, Any]] = []
    for d in dynamics:
        if not bed_of.get(beat_of_unit.get(d["unit_id"], ""), False):
            repairs.append(f"{d['unit_id']}: a {d['type']} needs music or ambience under it and none plays here, dropped")
            continue
        if d["type"] == "drop_out":
            drop_outs += 1
            if drop_outs > au.MAX_DROP_OUTS:
                repairs.append(f"{d['unit_id']}: too many drop_outs, one dropped")
                continue
        candidates.append(d)
    allowed = max(1, round(ar.estimate_seconds(lines, words_of) / au.SECONDS_PER_DYNAMIC))
    rank = {"peak": 0, "hush": 0, "intense": 1, "rising": 2, "steady": 3, "calm": 3}
    ranked = sorted(range(len(candidates)), key=lambda i: (rank.get(energy_of.get(candidates[i]["unit_id"], "steady"), 3), i))
    keep = set(ranked[:allowed])
    kept_dynamics = [d for i, d in enumerate(candidates) if i in keep]
    for i, d in enumerate(candidates):
        if i not in keep:
            repairs.append(f"{d['unit_id']}: more dynamics than this length carries, the {d['type']} was dropped")

    words_of = {u["unit_id"]: len(u["text"].split()) for u in units}
    wishes = Counter((l["speaker"], l["tone"]) for l in lines if not l["tone_supported"])
    audio = theme.get("audio") or {}
    return {
        "version": 1, "script_hash": ar.script_hash(script.sections), "theme": theme.get("theme_id", ""),
        "arc": arc, "moments": moments, "lines": lines,
        "music": ar.segments(arc, "music"), "ambience": ar.segments(arc, "ambience"),
        "sfx": kept_sfx, "dynamics": kept_dynamics,
        "mix": {"voice_db": au.VOICE_DB, "duck_db": au.DUCK_DB, "sfx_density": density,
                "music_mood": audio.get("music_mood", [])},
        "voice_wishes": [{"character_id": c, "tone": t, "sentences": n} for (c, t), n in wishes.most_common()],
        "metadata": {"agent": STAGE, "beats": len(beats), "estimated_seconds": round(ar.estimate_seconds(lines, words_of), 1),
                     "repairs": repairs}}


def _main() -> None:
    import argparse

    from core.run_context import RunContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True, help="shared_context.json of a run that has a story_director output")
    args = parser.parse_args()
    ctx = RunContext.load(args.from_run)
    script = Script(**ctx.read_stage_output("story_director"))
    plan = run_voice_director(script, context=ctx)
    ctx.write_stage_output("voice_plan", plan)
    ctx.save()
    print_plan(plan, script)


def print_plan(plan: Dict[str, Any], script: Script) -> None:
    text_of = {u["unit_id"]: u["text"] for u in ar.split_units(script.sections)}
    by_beat: Dict[str, List[Dict[str, Any]]] = {}
    for line in plan["lines"]:
        by_beat.setdefault(line["beat_id"], []).append(line)
    for a in plan["arc"]:
        print(f"\n== {a['beat_id']} [{a['role']}]  energy {a['energy']}, tempo {a['tempo']}"
              f"  music: {a['music'] or '-'}  ambience: {a['ambience'] or '-'}\n   {a['note']}")
        for m in [x for x in plan["moments"] if x["beat_id"] == a["beat_id"]]:
            print(f"   moment {m['from_cue']!r} .. {m['to_cue'] or 'end'}: energy {m['energy']}, tempo {m['tempo']} ({m['why']})")
        for l in by_beat.get(a["beat_id"], []):
            flag = "" if l["tone_supported"] else "  (no clip yet)"
            emph = f"  lean on: {', '.join(l['emphasis'])}" if l["emphasis"] else ""
            print(f"   {l['unit_id']:<11}{l['speaker']:<10}{l['energy']:<8}{l['pace']:<8}{l['tone']:<9}"
                  f"pause {l['pause_after']:<7}{emph}{flag}\n               \"{_short(text_of[l['unit_id']], 14)}\"")
            for e in [x for x in plan["sfx"] if x["unit_id"] == l["unit_id"]]:
                print(f"         sfx {e['asset']} at {e['anchor']}{' ' + repr(e['words']) if e['words'] else ''} ({e['level']})")
            for e in [x for x in plan["dynamics"] if x["unit_id"] == l["unit_id"]]:
                print(f"         dynamic {e['type']} at {e['anchor']}{' ' + repr(e['words']) if e['words'] else ''} ({e['seconds']} s)")
    meta = plan["metadata"]
    print(f"\nEstimated length of the voice: {meta['estimated_seconds']:.0f} s")
    print(f"Music beds: {plan['music'] or 'none'}\nAmbience beds: {plan['ambience'] or 'none'}")
    print(f"\nRepairs made in code ({len(meta['repairs'])}):")
    for line in meta["repairs"]:
        print("  -", line)
    if plan["voice_wishes"]:
        print("\nTones the plan wanted that no recording supports yet:")
        for w in plan["voice_wishes"]:
            print(f"  {w['character_id']}: {w['tone']} ({w['sentences']} sentence(s))")
    print(f"\nProblems left ({len(meta['issues'])}):")
    for i in meta["issues"]:
        print(f"  [{i['severity']:<7}] {i['unit_id'] or 'plan':<11}{i['check']:<10}{i['problem']}")


generate_voice_plan = run_voice_director
execute_voice_director = run_voice_director

if __name__ == "__main__":
    _main()