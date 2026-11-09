"""
Kestrel Shorts - Visual Director (version 3: plan, then fill)
=============================================================

Decides what the viewer SEES, beat by beat: images, screen elements (tables, lists, flows...), text,
backdrops, particles and the avatar. It writes choices and short descriptions; code owns every number and every
mechanical choice (layout, sizes, positions, motions, easing, transitions, overlays), taking them from the
channel's theme and the layout spec (engines/visual/layout_spec.py).

A small model does better with narrow questions than with one big one, so each beat takes two steps:
  1. PLAN   one call: how many shots, where each starts (a cue: a phrase from the narration), and the KIND of
            visual of each: photo, cutout, photo_set, sequence (a quick zoom or reveal), element (and which),
            or text_only, plus a short subject and intent.
  2. FILL   one small call per shot with a schema that fits its kind: a picture shot gets only image fields; an
            element shot gets the typed content schema of that one element; a text shot gets only texts.

Rule breaks are REPAIRED in code, not retried (every repair is printed): an image brief that shows the cast is
dropped, writing words are stripped, an element with invented or uncited data (or naming a character) and a
text that is too long or cut off are dropped; a shot that is too full is kept, marked, and becomes a request to
the Story Director. Afterwards engines/visual/visual_rules.py repairs the whole plan (duplicate images become
reuses, unrelated zoom chains become fades, assets missing from the library are replaced and wishlisted,
kinetic use is capped by the theme) and fills the mechanical fields. Only invalid JSON triggers a retry, and
then it fails loudly.

    python -m agents.visual_director --from-run <path to shared_context.json>
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agents._common import read_brief
from agents.story_director import _feedback_items
from core import config, llm
from core.models.script import Script, ScriptBeat
from core.registry_loader import RegistryLoader
from creative.catalogs import asset_catalog as ac
from creative.catalogs import beat_role_catalog
from creative.catalogs import visual_catalog as vc
from creative.catalogs import visual_style_catalog
from engines.visual import visual_rules as vr

logger = logging.getLogger(__name__)

STAGE = "visual_director"
PHASES = ("draft", "revision", "finalize")
TEXT_WORDS = {"attention": 8, "explanatory": 14}
TEXT_ANCHOR = {"attention": "upper_center", "explanatory": "lower_center"}
CORNERS = ["bottom_right", "bottom_left"]          # an avatar keeps its corner: first character right, second left
VISUALS = ["photo", "cutout", "photo_set", "sequence", "element", "text_only"]
PICTURES = ("photo", "cutout", "photo_set", "sequence")
ELEMENT_LIMITS = {"flow": [("steps", 2, 6)], "list": [("items", 2, 6)], "timeline": [("events", 2, 6)],
                  "ranking": [("items", 3, 6)], "table": [("rows", 2, 8)], "chart": [("labels", 2, 8), ("values", 2, 8)]}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def avatar_plan(beats: List[ScriptBeat], level: str) -> Dict[str, str]:
    """Per beat: required | allowed | not_allowed, from the avatar dial and beat roles."""
    fits = beat_role_catalog.get_beat_role_catalog()
    fit = lambda b: fits.get(b.role, {}).get("avatar_fit", "medium")
    if level == "none":
        return {b.beat_id: "not_allowed" for b in beats}
    if level == "constant":
        return {b.beat_id: "required" for b in beats}
    if level == "frequent":
        return {b.beat_id: ("required" if fit(b) in ("high", "medium") else "allowed") for b in beats}
    plan, cap, count, run = {}, max(1, int(0.4 * len(beats))), 0, 0     # cameo
    for b in beats:
        ok = fit(b) == "high" and count < cap and run < 2
        plan[b.beat_id] = "allowed" if ok else "not_allowed"
        count, run = count + ok, (run + 1 if ok else 0)
    return plan


def _beat_text(beat: ScriptBeat) -> str:
    return " ".join(s.text for s in beat.sections)


def _speaker_at(beat: ScriptBeat, position: int) -> Optional[str]:
    """Who is speaking at a character position of the beat's narration."""
    start = 0
    for section in beat.sections:
        end = start + len(section.text)
        if position <= end:
            return section.speaker_character_id
        start = end + 1
    return beat.sections[-1].speaker_character_id


def _short(text: str, words: int = 9) -> str:
    parts = str(text).split()
    return " ".join(parts[:words]) + ("..." if len(parts) > words else "")


def _example_schema(example: Any) -> Dict[str, Any]:
    """A JSON schema built from a catalog example: strings, numbers, lists, objects, 'a | b' choices."""
    if isinstance(example, bool):
        return {"type": "boolean"}
    if isinstance(example, (int, float)):
        return {"type": "number"}
    if isinstance(example, str):
        return {"type": "string", "enum": [x.strip() for x in example.split("|")]} if " | " in example else {"type": "string"}
    if isinstance(example, list):
        return {"type": "array", "items": _example_schema(example[0]) if example else {"type": "string"}}
    if isinstance(example, dict):
        return {"type": "object", "properties": {k: _example_schema(v) for k, v in example.items()},
                "required": list(example)}
    return {}


def _clean_content(value: Any) -> Any:
    """Remove citation markers like [2] that the model copies from the findings."""
    if isinstance(value, str):
        return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", value).strip()
    if isinstance(value, list):
        return [_clean_content(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean_content(v) for k, v in value.items()}
    return value


def _text_schema(max_items: int, min_items: int = 0) -> Dict[str, Any]:
    item = {"type": "object", "properties": {"kind": {"type": "string", "enum": list(TEXT_WORDS)},
                                             "content": {"type": "string"}}, "required": ["kind", "content"]}
    return {"type": "array", "items": item, "minItems": min_items, "maxItems": max_items}


# ----------------------------------------------------------------------
# Step 1: plan the shots of a beat
# ----------------------------------------------------------------------

def _plan_schema(max_shots: int, kinds: List[str], poses: List[str], particles: List[str]) -> Dict[str, Any]:
    text = {"type": "string"}
    enum = lambda values: {"type": "string", "enum": list(values)}
    shot = {"type": "object", "properties": {
        "cue": text, "intent": text, "visual": enum(VISUALS), "subject": text, "element_kind": enum(kinds),
        "avatar": {"type": "boolean"}, "pose_id": enum(poses or [""]), "particles": enum([""] + particles),
        "kinetic": {"type": "boolean"}},
        "required": ["cue", "intent", "visual", "subject", "element_kind"]}
    return {"type": "object", "properties": {"shots": {"type": "array", "items": shot, "minItems": 1,
                                                       "maxItems": max_shots}}, "required": ["shots"]}


def _plan_prompt(info, theme, world_line, beat, prev_beat, next_beat, policy, level, kinds, numbers,
                 max_shots, current_shots, feedback) -> str:
    style = info["visual_style"]
    style_desc = visual_style_catalog.get_visual_style_catalog().get(style, {}).get("description", "")
    narration = "\n".join(f'  [{s.speaker_character_id or "narrator"}] {s.text}' for s in beat.sections)
    first = beat.sections[0]
    hints = " ".join(str(theme.get("format_hints", "")).split()) or "not specified"
    element_lines = "\n".join(f"    {k}{' [data]' if vc.ELEMENT_KINDS[k].get('data') else ''}: "
                              f"{vc.ELEMENT_KINDS[k]['description']}" for k in kinds)
    avatar_rule = {"required": "an avatar is added to every shot automatically: set avatar true",
                   "allowed": "set avatar true only where a character reacts, asks or hands over",
                   "not_allowed": "no avatar in this beat: set avatar false"}[policy]
    extra = ""
    if current_shots:
        extra += "\nCurrent plan for this beat (revise it):\n" + "\n".join(
            f"  - {s.get('intent', '')}" for s in current_shots) + "\n"
    if feedback:
        extra += "\nFeedback to address:\n" + "\n".join(
            f"  - from {i['from']}: {i['problem']}" + (f" Suggestion: {i['suggestion']}" if i["suggestion"] else "")
            for i in feedback) + "\n"
    return f"""Task: plan the SHOTS of ONE beat of a short video: how many shots, where each starts, and what KIND of visual each is. You do not write the details yet.

Now: {info['time_line']}
Topic: {info['topic']}
Audience: {info['audience']}
Video format: {hints}
Visual style of generated images: {style or 'not specified'}{(' - ' + style_desc) if style_desc else ''}
World mood: {world_line}

This beat: {beat.beat_id} [{beat.role}] "{beat.title}" - intent: {first.narrative_purpose}; about: {first.setting_context}
Narration (in order):
{narration}
Previous beat: {prev_beat or '(this is the first)'}
Next beat: {next_beat or '(this is the last)'}
{extra}
Kinds of visual (choose one per shot):
  photo      - one picture of a real object, place or scene
  cutout     - one object or place standing alone on a coloured backdrop
  photo_set  - two to four DIFFERENT pictures at once: a comparison or a collection (posters, items)
  sequence   - two or three views of one subject, one after another (a quick zoom or reveal; use rarely)
  element    - a screen element instead of a picture; then set element_kind:
{element_lines}
  text_only  - a short text carries the shot (rarely)
Numbers that appear in the research: {numbers or 'none (so no [data] elements)'}

Rules:
- At most {max_shots} shots: a new shot when the idea changes or the narration moves on to a new thing. A shot needs at least {config.MIN_WORDS_PER_SHOT} words of narration. Start every later shot at a cue: a short phrase copied exactly from the narration ("" for the first shot).
- Pictures are the heart of the video: give at least half of the shots a picture (photo, cutout, photo_set or sequence). Use an element when the narration lists, compares, orders, counts, or explains a process step by step.
- A photograph can only show real things: objects, places, scenes, silhouettes or distant figures. Cross-sections, diagrams, rays, arrows, maps, charts and writing are elements, never pictures. Close-ups of faces, eyes or skin only when the topic is about them, and then one plain macro at most.
- subject: what the picture or element is about, up to 10 words, never a character or avatar name. intent: up to 12 words.
- A cutout is a solid object with a clear outline (a camera, a lens, a globe, a tool): never a body part (an eye, a hand, a face), an organ, a cross-section, a scene or anything diffuse. A comparison is a photo_set with ONE subject per picture.
- Pictures are for things a camera could photograph: objects, places, people at work. Body parts, organs, anatomy, mechanisms inside something and abstract processes are NOT photos: show them with an element (flow, labels, comparison, list) and leave the picture out. Never use more than two pictures of the same subject in the whole video.
- avatar: {avatar_rule} (level '{level}').
- kinetic: true only for one punchy line in the whole video, rarely. particles: leave "" unless atmosphere really needs it.
- element_kind is only used when visual is element; otherwise give any value.
Return JSON only."""


def _plan_beat(prompt: str, schema: Dict[str, Any], lowered: str, words: List[str], max_shots: int,
               num_predict: Optional[int]) -> List[Dict[str, Any]]:
    def validate(data: Dict[str, Any]) -> Dict[str, Any]:
        raw = [s for s in data.get("shots", []) if isinstance(s, dict)]
        if not raw:
            raise ValueError("shots must contain at least one shot.")
        out: List[Dict[str, Any]] = []
        prev_pos = -1
        for i, s in enumerate(raw[:max_shots]):
            cue, pos = "", 0
            if i > 0:
                cue = str(s.get("cue", "")).strip()
                pos = lowered.find(cue.lower()) if cue else -1
                if pos <= prev_pos:                      # missing or out of order: spread evenly
                    wi = round(i * len(words) / min(len(raw), max_shots))
                    cue = " ".join(words[wi:wi + 3])
                    pos = lowered.find(cue.lower())
                if not cue or pos <= prev_pos:
                    continue
                prev_pos = pos
            out.append({"cue": cue, "position": max(pos, 0), "intent": str(s.get("intent", "")).strip(),
                        "visual": s.get("visual") if s.get("visual") in VISUALS else "photo",
                        "subject": " ".join(str(s.get("subject", "")).split()),
                        "element_kind": s.get("element_kind"), "avatar": bool(s.get("avatar")),
                        "pose_id": s.get("pose_id"), "particles": s.get("particles") or "",
                        "kinetic": bool(s.get("kinetic"))})
        if not out:
            raise ValueError("no usable shot: every later shot needs a cue that appears in the narration.")
        return {"shots": out}

    return llm.generate_json(STAGE, prompt, schema=schema, validate=validate, retries=1,
                             num_predict=num_predict)["shots"]


# ----------------------------------------------------------------------
# Step 2: fill one shot
# ----------------------------------------------------------------------

def _fill_schema(visual: str, element_kind: str, media_ids: List[str]) -> Dict[str, Any]:
    text = {"type": "string"}
    enum = lambda values: {"type": "string", "enum": list(values)}
    refs = [""] + media_ids + (["previous"] if visual == "sequence" else [])
    media = {"type": "object", "properties": {
        "source": enum(vc.names(vc.MEDIA_SOURCES)), "ref": enum(refs), "change": enum(["none"] + vc.CHANGE_LEVELS),
        "brief": text, "framing": enum(vc.FRAMINGS)},
        "required": ["source", "ref", "change", "brief", "framing"]}
    if visual in PICTURES:
        low, high = {"photo": (1, 1), "cutout": (1, 1), "photo_set": (2, 4), "sequence": (2, 3)}[visual]
        props: Dict[str, Any] = {"media": {"type": "array", "items": media, "minItems": low, "maxItems": high},
                                 "text": _text_schema(2)}
        if visual == "cutout":
            props["backdrop"] = {"type": "object", "properties": {
                "kind": enum(vc.names(vc.BACKDROPS)), "color": enum(vc.BACKDROP_COLORS)}, "required": ["kind", "color"]}
        return {"type": "object", "properties": props, "required": ["media"]}
    if visual == "element":
        content = _example_schema(vc.ELEMENT_KINDS[element_kind]["content"])
        for key, low, high in ELEMENT_LIMITS.get(element_kind, []):
            if key in content.get("properties", {}):
                content["properties"][key].update({"minItems": low, "maxItems": high})
        return {"type": "object", "properties": {
            "content": content, "source_findings": {"type": "array", "items": {"type": "integer"}},
            "text": _text_schema(1)}, "required": ["content", "source_findings"]}
    return {"type": "object", "properties": {"text": _text_schema(2, 1)}, "required": ["text"]}


def _fill_prompt(info, shot, visual, element_kind, slice_text, media_index, findings, numbers) -> str:
    style = info["visual_style"]
    style_desc = visual_style_catalog.get_visual_style_catalog().get(style, {}).get("description", "")
    head = f"""Task: write the details of ONE shot of a short video.

Topic: {info['topic']}
Visual style of generated images: {style or 'not specified'}{(' - ' + style_desc) if style_desc else ''} (do not repeat the style in briefs)
This shot: {shot['intent']} - the visual is a {visual}{(' (' + element_kind + ')') if visual == 'element' else ''} about: {shot['subject']}
The narration while this shot is on screen: "{slice_text}"
"""
    texts = (f"- text: attention text (up to {TEXT_WORDS['attention']} words, a punchy phrase or number) and/or "
             f"explanatory text (up to {TEXT_WORDS['explanatory']} words, a label or definition): complete phrases that "
             "do not repeat the narration. Never write subtitles.\n")
    if visual in PICTURES:
        index = "\n".join(f"  {m['media_id']}: {_short(m['brief'], 8)}" for m in media_index) or "  (none yet)"
        count = {"photo": "exactly one picture", "cutout": "exactly one picture", "photo_set": "two to four different pictures (never repeat one)",
                 "sequence": "two to three views of one subject"}[visual]
        return head + f"""
Earlier images you may reuse or vary (id: what it shows):
{index}

Write {count}.
- brief: 8 to 20 words describing ONLY the subject (never the backdrop or the style): concrete, pictureable, what a photograph could show. No writing, labels, arrows or diagrams. No character names. Each picture shows ONE subject: a comparison is two separate pictures (a lens in one, an eye in the other), never two things in one picture.
- source: generate (a new picture), reuse (ref = the id of an earlier image shown unchanged: write no new description) or variation (ref = an earlier image, change small, medium or large: the same subject from a new angle or at another time).
{'- sequence: each view after the first is a variation (source variation, ref "previous") of the one before, a step of the zoom or reveal. Its brief must say what changes (closer, from another side, a later moment): never repeat the first brief.' + chr(10) if visual == 'sequence' else ''}- framing: how close the picture is.
{'- backdrop: what sits behind the cut-out: kind and colour role.' + chr(10) if visual == 'cutout' else ''}{texts}Return JSON only."""
    if visual == "element":
        facts = "\n".join(f"  {i}. {str(f)[:150]}" for i, f in enumerate(findings[:12], 1)) or "  (none)"
        data_note = ("This element is [data]: use ONLY numbers that appear in the findings and cite the finding "
                     "numbers in source_findings." if vc.ELEMENT_KINDS[element_kind].get("data")
                     else "Use short labels from the narration or findings.")
        return head + f"""
Research findings (numbered):
{facts}
Numbers that appear in the findings: {numbers or 'none'}

Write the content of the {element_kind}, exactly in the shape of the schema. {data_note} Keep every item short.
- source_findings: the numbers of the findings it comes from ([] if none).
- text: at most one short text, or none; it must not repeat the narration.
Return JSON only."""
    return head + f"\nWrite the short text(s) for this shot.\n{texts}Return JSON only."


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def run_visual_director(
    script: Any = None,
    context: Any = None,
    story_plan: Any = None,
    revision_feedback: Any = None,
    previous_plan: Optional[Dict[str, Any]] = None,
    phase: str = "draft",
    num_predict: Optional[int] = None,
    *,
    characters: Any = None,
    registry: Optional[RegistryLoader] = None,
    **ignored: Any,                    # tolerate old keyword arguments (channel, world, registered_assets...)
) -> Dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}.")
    script = Script(**script) if isinstance(script, dict) else script
    if not isinstance(script, Script) or not script.sections:
        raise ValueError("A script with sections is required.")
    if phase != "draft" and not previous_plan:
        raise ValueError(f"phase={phase!r} requires previous_plan.")

    info = read_brief(context)
    ctx = info["ctx"]
    registry = registry or RegistryLoader()
    channel_id = info["channel_id"]
    beats = script.beats
    level = info["avatar_presence"]
    policy = avatar_plan(beats, level)
    run_id = ctx.run_id if ctx else ""

    char_ids = [c if isinstance(c, str) else c.character_id for c in (characters or [])] or (
        [m["id"] for m in registry.character_menu(channel_id)] if channel_id else [])
    poses = {cid: registry.pose_menu(cid) for cid in char_ids}
    default_pose = {cid: next((p["pose_id"] for p in ps if p.get("default")), ps[0]["pose_id"] if ps else "")
                    for cid, ps in poses.items()}
    all_poses = sorted({p["pose_id"] for ps in poses.values() for p in ps})
    banned_set = {"avatar", *(c.lower() for c in char_ids), *(p.lower() for p in all_poses)}
    for member in (registry.character_menu(channel_id) if channel_id else []):
        banned_set.add(str(member["name"]).lower())
    banned = sorted(t for t in banned_set if t)

    theme: Dict[str, Any] = {}
    try:
        theme_id = registry.get_channel(channel_id).get("theme_id") if channel_id else ""
        theme = registry.get_theme(theme_id) if theme_id else {}
    except Exception as exc:                     # a missing theme is not fatal, but the user must know
        print(f"[Visual Director] WARNING: no theme ({exc}); using plain defaults.")
    findings = list((ctx.read_stage_output("researcher") or {}).get("findings", [])) if ctx else []
    narration = {b.beat_id: _beat_text(b) for b in beats}
    numbers = ", ".join(sorted(vr.numbers_in(" ".join(str(f) for f in findings))))[:240]
    kinds = [k for k, spec in vc.ELEMENT_KINDS.items() if numbers or not spec.get("data")]

    world_line = "not specified"
    world_id = script.metadata.get("world_id") or (ctx.world_id if ctx else None)
    if world_id:
        try:
            w = registry.load_world(world_id)
            world_line = f"{w.name}: {w.description}; {w.atmosphere}".strip("; ")
        except Exception as exc:
            logger.warning("Could not load world %r: %s", world_id, exc)

    particle_names = list(ac.scan("particle"))
    texture_names = list(ac.scan("texture"))

    old_shots: Dict[str, List[Dict[str, Any]]] = {}
    for shot in (previous_plan or {}).get("shots", []):
        old_shots.setdefault(shot["beat_id"], []).append(shot)
    items = _feedback_items(revision_feedback)
    redo = {b.beat_id for b in beats}
    if phase != "draft":
        named = {i["beat_id"] for i in items}
        redo = {b.beat_id for b in beats} if ("any" in named or not named) else named & {b.beat_id for b in beats}

    def last_number(prefix: str, key: str, nested: str) -> int:
        found = [int(str(x[key])[len(prefix):]) for s in (previous_plan or {}).get("shots", [])
                 for x in s.get(nested, []) if str(x.get(key, "")).startswith(prefix)
                 and str(x[key])[len(prefix):].isdigit()]
        return max(found or [0])

    counter, element_counter = last_number("m", "media_id", "media"), last_number("e", "element_id", "elements")
    shots_out: List[Dict[str, Any]] = []
    media_index: List[Dict[str, Any]] = []      # every image that really gets made, in order (for reuse)
    all_ids: set = set()
    requests: List[Dict[str, str]] = []

    for index, beat in enumerate(beats):
        keep = phase != "draft" and beat.beat_id not in redo and beat.beat_id in old_shots
        if keep:       # a kept beat must not depend on images that no longer exist
            keep = {m["ref"] for s in old_shots[beat.beat_id] for m in s.get("media", []) if m.get("ref")} <= all_ids
        if keep:
            for s in old_shots[beat.beat_id]:
                shots_out.append(s)
                for m in s.get("media", []):
                    all_ids.add(m["media_id"])
                    if m.get("source") != "reuse":
                        media_index.append({"media_id": m["media_id"], "brief": m.get("brief", ""),
                                            "beat_id": beat.beat_id})
            continue

        btext = _beat_text(beat)
        words, lowered = btext.split(), btext.lower()
        pol = policy[beat.beat_id]
        max_shots = max(1, min(config.MAX_SHOTS_PER_BEAT, len(words) // config.MIN_WORDS_PER_SHOT or 1))
        prev_b = f"[{beats[index - 1].role}] {beats[index - 1].title}" if index else ""
        next_b = f"[{beats[index + 1].role}] {beats[index + 1].title}" if index + 1 < len(beats) else ""

        skeletons = _plan_beat(
            _plan_prompt(info, theme, world_line, beat, prev_b, next_b, pol, level, kinds, numbers, max_shots,
                         old_shots.get(beat.beat_id) if phase != "draft" else None,
                         [i for i in items if i["beat_id"] in (beat.beat_id, "any")]),
            _plan_schema(max_shots, kinds, all_poses, particle_names), lowered, words, max_shots, num_predict)

        beat_shots: List[Dict[str, Any]] = []
        notes: List[str] = []
        ids = [m["media_id"] for m in media_index]
        brief_of = {m["media_id"]: m["brief"] for m in media_index}

        for k, sk in enumerate(skeletons):
            label = f"shot {k + 1}"
            visual = sk["visual"]
            kind = sk["element_kind"] if sk["element_kind"] in kinds else None
            if visual == "element" and not kind:
                notes.append(f"{label}: an element without a valid kind became a text shot")
                visual = "text_only"
            end = skeletons[k + 1]["position"] if k + 1 < len(skeletons) else len(btext)
            slice_text = " ".join(btext[sk["position"]:end].split()) or btext
            filled = llm.generate_json(
                STAGE, _fill_prompt(info, sk, visual, kind, slice_text, media_index, findings, numbers),
                schema=_fill_schema(visual, kind or "", ids), retries=1, num_predict=num_predict)

            media: List[Dict[str, Any]] = []
            if visual in PICTURES:
                for j, m in enumerate(filled.get("media") or []):
                    if not isinstance(m, dict):
                        continue
                    source = m.get("source") if m.get("source") in vc.MEDIA_SOURCES else "generate"
                    ref = m.get("ref") if m.get("ref") in ids else ""
                    chain = visual == "sequence" and j > 0 and m.get("ref") == "previous"
                    if chain:
                        source = "variation"
                    elif source in ("reuse", "variation") and not ref:
                        source = "generate"
                    if source == "generate":
                        ref = ""
                    brief = " ".join(str(m.get("brief", "")).split())[:220] or (brief_of.get(ref, "") if source == "reuse" else "")
                    if not brief:
                        continue
                    if source != "reuse":
                        if vr.looks_like_element(brief):
                            notes.append(f"{label}: dropped an element written as an image ('{_short(brief, 6)}')")
                            continue
                        low = brief.lower()
                        if any(re.search(rf"\b{re.escape(t)}\b", low) for t in banned):
                            notes.append(f"{label}: dropped an image that shows the cast ('{_short(brief, 6)}')")
                            continue
                        cleaned = vr.strip_writing(brief)
                        if cleaned != brief:
                            notes.append(f"{label}: removed writing words from '{_short(brief, 6)}'")
                            brief = cleaned
                        if not brief:
                            continue
                    item = {"source": source, "ref": ref,
                            "change": (m.get("change") if m.get("change") in vc.CHANGE_LEVELS else "medium")
                                      if source == "variation" else "none",
                            "brief": brief,
                            "framing": m.get("framing") if m.get("framing") in vc.FRAMINGS else "medium",
                            "cutout": visual == "cutout"}
                    if chain:
                        item["_previous"] = True
                    media.append(item)

            elements: List[Dict[str, Any]] = []
            if visual == "element":
                content = filled.get("content") if isinstance(filled.get("content"), dict) else {}
                content = _clean_content(content)
                element = {"kind": kind, "content": content,
                           "source_findings": [n for n in filled.get("source_findings") or []
                                               if isinstance(n, int) and not isinstance(n, bool)]}
                problems = vr.check_element(element, findings)
                blob = " ".join(vr._texts(content)).lower()
                if any(re.search(rf"\b{re.escape(t)}\b", blob) for t in banned):
                    problems.append("it names a character")
                if problems:                              # never invented data: the element is dropped
                    notes.append(f"{label}: dropped a {kind}: " + "; ".join(problems))
                else:
                    elements.append(element)

            texts: List[Dict[str, str]] = []
            for t in filled.get("text") or []:
                if not isinstance(t, dict):
                    continue
                tkind, content = t.get("kind"), " ".join(str(t.get("content", "")).split())
                if tkind not in TEXT_WORDS or not content:
                    continue
                if len(content.split()) > TEXT_WORDS[tkind] or vr._CUT_OFF.search(content):
                    notes.append(f"{label}: dropped a {tkind} text that was too long or cut off ('{content[:40]}')")
                    continue
                texts.append({"kind": tkind, "content": content, "anchor": TEXT_ANCHOR[tkind]})

            avatar = None
            if pol == "required" or (pol == "allowed" and sk["avatar"]):
                cid = _speaker_at(beat, sk["position"])
                cid = cid if cid in char_ids else (char_ids[0] if char_ids else None)
                if cid:
                    pose = sk["pose_id"] if sk["pose_id"] in {p["pose_id"] for p in poses[cid]} else default_pose[cid]
                    avatar = {"character_id": cid, "pose_id": pose, "anchor": CORNERS[char_ids.index(cid) % 2],
                              "size": "medium", "motion": "bob"}

            backdrop = None
            b = filled.get("backdrop") if isinstance(filled.get("backdrop"), dict) else {}
            if b.get("kind") in vc.BACKDROPS:
                backdrop = {"kind": b["kind"],
                            "color": b.get("color") if b.get("color") in vc.BACKDROP_COLORS else "background"}
                if b["kind"] == "texture":
                    if texture_names:
                        backdrop["asset"] = texture_names[len(shots_out) % len(texture_names)]
                    else:
                        backdrop["kind"] = "flat"

            fx = {"particles": {"asset": sk["particles"], "motion": "float_up"}} if sk["particles"] in particle_names and sk["particles"] else None

            if not (media or elements or texts or avatar):
                notes.append(f"{label}: nothing usable was left, the shot was dropped")
                continue
            shot = {"cue": sk["cue"], "intent": sk["intent"] or beat.sections[0].narrative_purpose or beat.title,
                    "focus": "image" if visual in PICTURES else "text",
                    "arrangement": "sequence" if visual == "sequence" else "auto",
                    "backdrop": backdrop, "media": media, "elements": elements, "text": texts, "avatar": avatar,
                    "fx": fx, "subtitle_style": "kinetic" if sk["kinetic"] else "regular"}
            if sk["kinetic"]:
                shot["kinetic"] = {"mood": "punchy", "layer": "over"}
            shot["layout"] = vr.derive_layout(shot)
            load = vr.check_load(shot)
            if not load["fits"]:                          # kept, marked, and handed to the Story Director
                shot["overloaded"] = load["notes"]
                requests.append({"beat_id": beat.beat_id,
                                 "problem": f"{label} of this beat is too full for the screen: " + "; ".join(load["notes"]),
                                 "proposal": "Split this beat's narration so each part carries fewer items, one part per shot."})
            beat_shots.append(shot)

        if not beat_shots:
            raise ValueError(f"Beat {beat.beat_id}: no usable shot was left after the checks. " + "; ".join(notes))
        for k, shot in enumerate(beat_shots, 1):
            shot["shot_id"], shot["beat_id"] = f"{beat.beat_id}_s{k}", beat.beat_id
            previous_id = ""
            for m in shot["media"]:
                counter += 1
                m["media_id"], m["beat_id"], m["shot_id"] = f"m{counter}", beat.beat_id, shot["shot_id"]
                if m.pop("_previous", False):
                    m["ref"] = previous_id
                    if not previous_id:
                        m["source"], m["change"] = "generate", "none"
                previous_id = m["media_id"]
                all_ids.add(m["media_id"])
                if m["source"] != "reuse":
                    media_index.append({"media_id": m["media_id"], "brief": m["brief"], "beat_id": beat.beat_id})
            for e in shot["elements"]:
                element_counter += 1
                e["element_id"] = f"e{element_counter}"
            shots_out.append(shot)
        for note in notes:
            print(f"[Visual Director] {beat.beat_id}: {note}")
        print(f"[Visual Director] {beat.beat_id} [{beat.role}]: {len(beat_shots)} shot(s), "
              f"{sum(len(s['media']) for s in beat_shots)} image(s), "
              f"{sum(len(s['elements']) for s in beat_shots)} element(s), avatar={pol}")

    plan: Dict[str, Any] = {
        "version": 2, "phase": phase, "style": info["visual_style"], "theme": theme.get("theme_id", ""),
        "avatar_level": level, "shots": shots_out, "script_requests": requests, "metadata": {"agent": STAGE}}
    repairs = vr.repair_plan(plan, run_id, theme)
    vr.assign_defaults(plan, theme)
    issues = vr.check_plan(plan, findings, banned, theme, narration)

    media_all = [m for s in plan["shots"] for m in s["media"]]
    errors_left = sum(1 for i in issues if i["severity"] == "error")
    plan["metadata"].update({
        "beats": len(beats), "shots": len(plan["shots"]),
        "images_generated": sum(m["source"] == "generate" for m in media_all),
        "images_variation": sum(m["source"] == "variation" for m in media_all),
        "images_reused": sum(m["source"] == "reuse" for m in media_all),
        "cutouts": sum(bool(m.get("cutout")) and m["source"] != "reuse" for m in media_all),
        "elements": sum(len(s["elements"]) for s in plan["shots"]),
        "kinetic_shots": sum(s["subtitle_style"] == "kinetic" for s in plan["shots"]),
        "avatar_beats": sorted({s["beat_id"] for s in plan["shots"] if s.get("avatar")}),
        "feedback_items": len(items), "repairs": repairs, "issues": issues})
    m = plan["metadata"]
    print(f"[Visual Director] {m['shots']} shots, new={m['images_generated']} variations={m['images_variation']} "
          f"reused={m['images_reused']} cutouts={m['cutouts']} elements={m['elements']}; "
          f"{len(repairs)} repair(s) in code, {errors_left} error(s) and {len(issues) - errors_left} warning(s) left")
    return plan


def script_requests_as_feedback(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The director's requests, in the shape the Story Director accepts as revision_feedback."""
    return {"source": "visual_director", "status": "revision_requested", "issues": [
        {"id": f"v{i}", "from": "visual_director", "beat_id": r["beat_id"],
         "problem": r["problem"], "suggestion": r["proposal"]}
        for i, r in enumerate(plan.get("script_requests", []), 1)], "suggestions": []}


generate_visual_plan = run_visual_director
execute_visual_director = run_visual_director


def _main() -> None:
    import argparse

    from core.run_context import RunContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True, help="shared_context.json of a run that has a story_director output")
    args = parser.parse_args()
    ctx = RunContext.load(args.from_run)
    script = Script(**ctx.read_stage_output("story_director"))
    plan = run_visual_director(script, context=ctx)
    ctx.write_stage_output("visual_plan", plan)
    ctx.save()

    for beat in script.beats:
        print(f"\n== {beat.beat_id}: {beat.title} [{beat.role}]")
        for s in [x for x in plan["shots"] if x["beat_id"] == beat.beat_id]:
            av = f" + avatar {s['avatar']['character_id']}:{s['avatar']['pose_id']}" if s.get("avatar") else ""
            cue = f", cue: {s['cue']!r}" if s["cue"] else ""
            print(f"  {s['shot_id']} ({s['layout']}, subtitles {s['subtitle_style']}, "
                  f"in: {s.get('transition_in', 'cut')}{cue}){av}\n    why: {s['intent']}")
            if s.get("backdrop"):
                print(f"    backdrop: {s['backdrop']['kind']} ({s['backdrop']['color']})")
            for m in s["media"]:
                tag = m["source"] + (f" of {m['ref']}" if m["ref"] else "") + (", cut-out" if m.get("cutout") else "")
                print(f"    {m['media_id']} [{tag}] {m['brief']}")
            for e in s["elements"]:
                print(f"    element/{e['kind']} ({e['element_id']}): {json.dumps(e['content'], ensure_ascii=False)[:110]}")
            for t in s["text"]:
                print(f"    text/{t['kind']}: {t['content']}")
            if s.get("fx"):
                print(f"    particles: {s['fx']['particles']}")
            if s.get("overloaded"):
                print(f"    TOO FULL: {'; '.join(s['overloaded'])}")
    meta = plan["metadata"]
    print(f"\nRepairs made in code ({len(meta['repairs'])}):")
    for line in meta["repairs"]:
        print("  -", line)
    print(f"\nProblems left ({len(meta['issues'])}):")
    for i in meta["issues"]:
        print(f"  [{i['severity']:<7}] {i['shot_id'] or 'plan':<12}{i['check']:<8}{i['problem']}")
    if plan["script_requests"]:
        print("\nRequests to the Story Director:", json.dumps(plan["script_requests"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()