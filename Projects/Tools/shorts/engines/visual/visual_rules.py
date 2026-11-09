"""
Kestrel Shorts - Visual rules
=============================

The checks and repairs for a visual plan. No LLM. The Visual Director uses them while it writes (errors go
back to the model, safe fixes are made in code) and the Reviewer reuses them to report, so a rule lives in
one place.

Plan format (version 2)
  {"version": 2, "phase", "style", "theme", "avatar_level", "shots": [shot, ...],
   "script_requests": [...], "metadata": {...}}
  shot:
    shot_id, beat_id, cue, intent, focus
    arrangement    optional hint: sequence | scatter | inset (otherwise code derives the layout)
    layout         derived from the images: single | inset | stack | grid | scatter | sequence | text_only
    backdrop       {"kind", "color": palette role, "asset": for textures} or null
    media[]        {media_id, source: generate|reuse|variation, ref, change, brief, framing, motion, cutout}
    elements[]     {element_id, kind, content, source_findings, motion}
    text[]         {kind: attention|explanatory, content, anchor, motion}
    avatar         {character_id, pose_id, anchor, size, motion} or null (never blocks subtitles: they win)
    fx             {"particles": {"asset", "motion"}} or null
    overlays[], transition_in, easing
    subtitle_style regular | kinetic;  kinetic: {"mood", "emphasis_words", "layer"} (only when kinetic)

Kinetic subtitles are text LAYERS like images: the Frame Supervisor gives each phrase a position, size,
colour, rotation and layer (over or under the media) from the narration's word timings. The director only
picks the style per shot (and hints); the theme caps how often it is used.

    python -m engines.visual.visual_rules --from-run <path to shared_context.json>   # dry run on a saved plan
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from core.geometry import Rect
from creative.catalogs import asset_catalog as ac
from creative.catalogs import visual_catalog as vc
from engines.visual.layout_spec import (GUTTER, content_rect, fit_text, layout_for_images, list_fit,
                                        soft_region, split_items, split_table, subtitle_slot, table_fit)

DUPLICATE_SIMILARITY = 0.75          # two briefs this alike are the same picture
CHANGED_BRIEF_SIMILARITY = 0.6       # a "reuse" whose brief is less alike than this is really a new view
DEFAULT_KINETIC_SHARE, DEFAULT_KINETIC_RUN = 0.2, 1
VARIATION_SAME_SIMILARITY = 0.85     # a "variation" this alike to its source is really a reuse
OVERLOAD_SLACK = 0.10                # height estimates are rough (the Frame Supervisor measures exactly): "too full" only beyond this

_STOP = {"a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for", "from", "by", "its", "is", "are",
         "as", "into", "that", "this", "it", "their", "his", "her"}
_WRITING = re.compile(r"\b(labell?ed|labels?|captions?|texts?|letters|lettering|typography|logos?|arrows?)\b", re.I)
_NOT_A_PHOTO = re.compile(r"\b(cross[- ]sections?|diagrams?|schematics?|light rays?|rays|charts?|graphs?|maps?|"
                          r"infographics?|blueprints?|x-rays?|brains?|neurons?|neural|nerves?|cortex|organs?|anatom\w*|retina|cornea|optic|skeleton|skull|muscles?|veins?|arteries|lungs?)\b", re.I)
_FACE = re.compile(r"\b(?:close[- ]?ups?|macro|portrait)\b.{0,40}\b(?:faces?|eyes?|skin|lips|mouth|iris|pupil)\b|"
                   r"\b(?:faces?|eyes?|skin)\b.{0,40}\bclose[- ]?ups?\b", re.I)
_COMBINED = re.compile(r"\bside[- ]by[- ]side\b|\bnext to each other\b", re.I)
_CUT_OFF = re.compile(r"\b(?:the|a|an|to|of|and|or|in|on|with|for|from|by|is|are)$", re.I)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORD = re.compile(r"\b(hundred|thousand|million|billion|trillion)s?\b", re.I)

REQUIRED = {"table": ("columns", "rows"), "list": ("items",), "chart": ("type", "labels", "values"),
            "stat": ("value", "caption"), "timeline": ("events",), "comparison": ("left", "right"),
            "flow": ("steps",), "labels": ("targets",), "counter": ("to",), "ranking": ("items",),
            "gauge": ("value", "max"), "quote": ("text", "speaker"), "headline": ("text",),
            "definition": ("term", "meaning"), "callout": ("text",), "lower_third": ("title",),
            "equation": ("expression",)}
FIXED_HEIGHT = {"stat": 360, "counter": 360, "gauge": 420, "lower_third": 170, "equation": 300,
                "chart": 560, "labels": 0}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _theme_get(theme: Any, key: str, default: Any) -> Any:
    value = theme.get(key) if isinstance(theme, dict) else getattr(theme, key, None)
    return value if value else default


def _texts(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _texts(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _texts(item)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in _NUMBER.findall(text)} | {w.lower() for w in _NUMBER_WORD.findall(text)}


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def similarity(a: str, b: str) -> float:
    """How alike two briefs are, 0 to 1 (shared content words)."""
    x, y = content_words(a), content_words(b)
    return len(x & y) / len(x | y) if x and y else 0.0


# ----------------------------------------------------------------------
# Checks on one thing
# ----------------------------------------------------------------------

def check_brief(brief: str, banned: List[str]) -> List[Tuple[str, str]]:
    """Problems with one image brief: [(severity, message)], severity 'error' or 'warning'."""
    problems: List[Tuple[str, str]] = []
    low = brief.lower()
    hit = next((t for t in banned if re.search(rf"\b{re.escape(t)}\b", low)), None)
    if hit:
        problems.append(("error", f"it shows the cast ({hit!r}); generated images never show the cast"))
    writing = _WRITING.search(brief)
    if writing:
        problems.append(("error", f"it asks for {writing.group(0)!r}; Remotion draws writing, labels and arrows"))
    cue = _NOT_A_PHOTO.search(brief)
    if cue:
        problems.append(("warning", f"{cue.group(0)!r} is hard for a photo generator; a graphic may fit better"))
    if _COMBINED.search(brief):
        problems.append(("warning", "two subjects in one picture often blend (a lens with eyelashes); use two pictures"))
    if _FACE.search(brief):
        problems.append(("warning", "close-ups of faces, eyes or skin show the AI look; prefer objects or places"))
    return problems


_ELEMENT_WORD = re.compile(r"^\s*(?:" + "|".join(sorted(vc.ELEMENT_KINDS)) + r")\b\s*[:\-]", re.I)


def looks_like_element(brief: str) -> bool:
    """An element written as an image brief, like 'flow: light, cornea, lens' (it belongs in elements)."""
    return bool(_ELEMENT_WORD.match(brief))


def strip_writing(brief: str) -> str:
    """Remove words that ask for writing (labeled, caption, text...) from an image brief."""
    return " ".join(_WRITING.sub(" ", brief).split())


def check_element(element: Dict[str, Any], findings: List[str]) -> List[str]:
    """Problems with one screen element: its kind, its content, and (for data elements) its sources."""
    kind = element.get("kind")
    spec = vc.ELEMENT_KINDS.get(kind)
    if spec is None:
        return [f"unknown element kind {kind!r}"]
    content = element.get("content")
    if not isinstance(content, dict):
        return [f"{kind}: content must be an object"]
    problems = [f"{kind}: missing '{key}'" for key in REQUIRED.get(kind, ()) if content.get(key) in (None, "", [], {})]
    if kind == "chart" and not problems:
        labels, values = content.get("labels"), content.get("values")
        if not (isinstance(labels, list) and isinstance(values, list) and len(labels) == len(values)):
            problems.append("chart: labels and values must be lists of the same length")
        elif not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            problems.append("chart: values must be numbers")
    if spec.get("data"):
        cited = [n for n in element.get("source_findings") or [] if isinstance(n, int) and 1 <= n <= len(findings)]
        if not cited:
            problems.append(f"{kind}: cite the research findings it comes from (source_findings)")
        else:
            known = numbers_in(" ".join(findings[n - 1] for n in cited))
            unknown = sorted(numbers_in(" ".join(_texts(content))) - known)
            if unknown:
                problems.append(f"{kind}: numbers not in the cited findings: {', '.join(unknown)}")
    return problems


def _estimate(kind: str, content: Dict[str, Any], region: Rect) -> Tuple[float, int, str]:
    """(height in px, parts, how). parts: 1 fits as one piece, 2+ must be split, 0 cannot fit readably."""
    if kind == "table":
        columns, rows = [str(c) for c in content.get("columns", [])], content.get("rows", [])
        fit = table_fit(columns, rows, region)
        if fit["fits"]:
            return float(fit["total_height"]), 1, ""
        parts = split_table(columns, rows, region)
        return region.height, len(parts), "; ".join(f"{p['columns']} x {len(p['rows'])} rows" for p in parts)
    if kind == "list":
        items = [str(i) for i in content.get("items", [])]
        fit = list_fit(items, region)
        if fit["fits"]:
            return float(fit["total_height"]), 1, ""
        parts = split_items(items, region)
        return region.height, len(parts), "; ".join(f"{len(p)} items" for p in parts)
    if kind in ("quote", "headline", "definition", "callout"):
        key = "meaning" if kind == "definition" else "text"
        fit = fit_text(str(content.get(key, "")), "attention" if kind == "callout" else "explanatory", region)
        return fit["height"] + (0 if kind == "callout" else 160), (1 if fit["fits"] else 0), ""
    if kind == "ranking":
        height = 90 * len(content.get("items", [])) + 80
    elif kind == "timeline":
        height = 130 * len(content.get("events", [])) + 40
    elif kind == "comparison":
        points = max(len(content.get("left", {}).get("points", [])), len(content.get("right", {}).get("points", [])))
        height = 120 * points + 140
    elif kind == "flow":
        height = 150 * len(content.get("steps", []))
    else:
        height = FIXED_HEIGHT.get(kind, 300)
    parts = 1 if height <= region.height else math.ceil(height / region.height)
    return float(height), parts, ("" if parts == 1 else f"about {parts} parts")


def check_load(shot: Dict[str, Any]) -> Dict[str, Any]:
    """Does the text-like content of a shot fit the screen, with a free strip left for the subtitles?
    (Pictures, backdrops and the avatar do not count: the subtitles go over them.)"""
    region = content_rect()
    heights: List[float] = []
    notes: List[str] = []
    for item in shot.get("text") or []:
        kind = item.get("kind")
        if kind in ("attention", "explanatory"):
            fit = fit_text(str(item.get("content", "")), kind, region)
            heights.append(float(fit["height"]))
            if not fit["fits"]:
                notes.append(f"the {kind} text is too long for the screen")
    for element in shot.get("elements") or []:
        kind = element.get("kind", "")
        try:
            height, parts, how = _estimate(kind, element.get("content") or {}, region)
        except (AttributeError, TypeError):
            height, parts, how = region.height, 0, ""
        heights.append(height)
        if parts == 0:
            notes.append(f"the {kind} cannot fit at a readable size")
        elif parts > 1:
            notes.append(f"the {kind} needs {parts} parts" + (f": {how}" if how else ""))
    rects: List[Rect] = []
    y = region.y
    for h in heights:
        if h > 0:
            rects.append(Rect(region.x, y, region.width, h))
            y += h + GUTTER
    needed = (y - GUTTER - region.y) if rects else 0.0
    available = soft_region(region).height
    kinetic = shot.get("subtitle_style") == "kinetic"
    slot = None if kinetic else subtitle_slot(rects)
    if needed > available * (1 + OVERLOAD_SLACK):
        notes.append(f"too much on screen: {needed:.0f} px needed, {available:.0f} px available")
        if not kinetic and slot is None:
            notes.append("no free strip is left for the subtitles")
    return {"fits": not notes, "needed": round(needed), "available": round(available),
            "subtitle_slot": slot, "notes": notes}


def _allowed_kinetic(count: int, share: float) -> int:
    return max(1, int(share * count)) if share > 0 and count else 0


def check_kinetic(shots: List[Dict[str, Any]], share: float, max_run: int) -> List[str]:
    flags = [s.get("subtitle_style") == "kinetic" for s in shots]
    problems: List[str] = []
    allowed = _allowed_kinetic(len(shots), share)
    if sum(flags) > allowed:
        problems.append(f"{sum(flags)} of {len(shots)} shots use kinetic subtitles; the theme allows {allowed}")
    run = longest = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    if longest > max_run:
        problems.append(f"{longest} kinetic shots in a row; the theme allows {max_run}")
    return problems


# ----------------------------------------------------------------------
# Safe repairs (made in code)
# ----------------------------------------------------------------------

def derive_layout(shot: Dict[str, Any]) -> str:
    """The layout follows from the images the director listed (and an optional arrangement hint)."""
    n = len(shot.get("media") or [])
    hint = shot.get("arrangement")
    if n == 0:
        return "text_only"
    if n >= 2 and hint in ("sequence", "scatter") or (n == 2 and hint == "inset"):
        return hint
    return layout_for_images(n)


def repair_media(plan: Dict[str, Any]) -> List[str]:
    """A new image that repeats an earlier one becomes a reuse (dropped if it is in the same shot); a 'reuse'
    whose brief asks for something new becomes a variation; a reuse shows exactly the brief it reuses."""
    log: List[str] = []
    made: List[Dict[str, Any]] = []              # images that really get generated, in order
    by_id: Dict[str, Dict[str, Any]] = {}
    for shot in plan.get("shots", []):
        kept = []
        for media in shot.get("media") or []:
            media_id, brief = media.get("media_id", ""), str(media.get("brief", "")).strip()
            entry = {"media_id": media_id, "shot_id": shot.get("shot_id"), "brief": brief}
            source = media.get("source", "generate")
            if source == "generate":
                twin = next((m for m in made if similarity(brief, m["brief"]) >= DUPLICATE_SIMILARITY), None)
                if twin and twin["shot_id"] == shot.get("shot_id"):
                    log.append(f"{media_id}: the same picture as {twin['media_id']} in the same shot, dropped")
                    continue
                if twin:
                    media.update({"source": "reuse", "ref": twin["media_id"], "change": "none", "brief": twin["brief"]})
                    log.append(f"{media_id}: repeats {twin['media_id']}, now a reuse of it")
                else:
                    made.append(entry)
            elif source == "variation":
                base = by_id.get(media.get("ref"))
                if base is not None and similarity(brief, str(base.get("brief", ""))) >= VARIATION_SAME_SIMILARITY:
                    if any(k.get("ref") == base.get("media_id") and k.get("source") == "reuse" for k in kept):
                        log.append(f"{media_id}: shows the same thing as {base['media_id']}, already in this shot, dropped")
                        continue
                    media.update({"source": "reuse", "ref": base["media_id"], "change": "none",
                                  "brief": base.get("brief", brief)})
                    log.append(f"{media_id}: a 'variation' that describes the same picture as {base['media_id']}, now a reuse of it")
                else:
                    made.append(entry)
            elif source == "reuse":
                base = by_id.get(media.get("ref"))
                if base is None:
                    media.update({"source": "generate", "ref": "", "change": "none"})
                    made.append(entry)
                    log.append(f"{media_id}: reused an image that does not exist, now a new image")
                elif brief and similarity(brief, str(base.get("brief", ""))) < CHANGED_BRIEF_SIMILARITY:
                    media.update({"source": "variation", "change": "medium"})
                    made.append(entry)
                    log.append(f"{media_id}: a 'reuse' whose brief shows something new, now a variation of {base['media_id']}")
                else:
                    media["brief"] = base.get("brief", brief)
            kept.append(media)
            by_id[media_id] = media
        shot["media"] = kept
    return log


_NOT_CUTOUT = re.compile(r"\b(?:eyes?|faces?|heads?|hands?|fingers?|skin|brain|neurons?|nerves?|cortex|organs?|anatom\w*|"
                         r"retina|cornea|iris|pupil|skull|muscles?|veins?|hearts?|lungs?|cells?|scenes?|rooms?|landscapes?|"
                         r"skyline|streets?|cross[- ]sections?)\b", re.I)


def repair_cutouts(plan: Dict[str, Any]) -> List[str]:
    """A cut-out only works on a solid object with a clear outline. Body parts, anatomy and scenes lose the wrong
    parts of the picture when the background is removed, so they stay full pictures."""
    log: List[str] = []
    for shot in plan.get("shots", []):
        for media in shot.get("media") or []:
            if media.get("cutout") and _NOT_CUTOUT.search(str(media.get("brief", ""))):
                media["cutout"] = False
                log.append(f"{media.get('media_id')}: a cut-out of a body part or a scene does not work, now a full picture")
    return log


def repair_transitions(plan: Dict[str, Any]) -> List[str]:
    """zoom_through only makes sense between related images (a variation or reuse of the previous shot's)."""
    log: List[str] = []
    previous: set = set()
    for shot in plan.get("shots", []):
        media = shot.get("media") or []
        if shot.get("transition_in") == "zoom_through":
            if not any(m.get("source") in ("variation", "reuse") and m.get("ref") in previous for m in media):
                shot["transition_in"] = "fade"
                log.append(f"{shot.get('shot_id')}: zoom_through between unrelated images, now a fade")
        previous = {m.get("media_id") for m in media}
    return log


def repair_assets(plan: Dict[str, Any], run_id: str = "", record: bool = True) -> List[str]:
    """Assets that are not in the library are replaced or dropped, and noted on the wishlist."""
    log: List[str] = []

    def missing(kind: str, name: str, why: str) -> None:
        if record:
            ac.request(kind, name, why, "visual_director", run_id)

    for shot in plan.get("shots", []):
        sid = shot.get("shot_id")
        transition = shot.get("transition_in")
        if transition and not ac.exists("transition", transition):
            missing("transition", transition, f"wanted as the transition into {sid}")
            shot["transition_in"] = "fade"
            log.append(f"{sid}: transition {transition!r} is not in the library, now a fade")
        kept = []
        for name in shot.get("overlays") or []:
            if ac.exists("overlay", name):
                kept.append(name)
            else:
                missing("overlay", name, f"wanted over {sid}")
                log.append(f"{sid}: overlay {name!r} is not in the library, dropped")
        shot["overlays"] = kept
        particles = (shot.get("fx") or {}).get("particles")
        if particles and not ac.exists("particle", str(particles.get("asset", ""))):
            missing("particle", str(particles.get("asset", "")), f"particles wanted in {sid}")
            shot["fx"].pop("particles")
            log.append(f"{sid}: particle asset {particles.get('asset')!r} is not in the library, dropped")
        backdrop = shot.get("backdrop")
        if backdrop and backdrop.get("kind") == "texture" and not ac.exists("texture", str(backdrop.get("asset", ""))):
            missing("texture", str(backdrop.get("asset", "")), f"texture backdrop wanted in {sid}")
            backdrop["kind"] = "flat"
            log.append(f"{sid}: texture {backdrop.get('asset')!r} is not in the library, now a flat backdrop")
    return log


def repair_kinetic(plan: Dict[str, Any], share: float, max_run: int) -> List[str]:
    log: List[str] = []
    shots = plan.get("shots", [])
    allowed = _allowed_kinetic(len(shots), share)
    used = run = 0
    for shot in shots:
        if shot.get("subtitle_style") == "kinetic":
            if used >= allowed or run >= max_run:
                shot["subtitle_style"] = "regular"
                shot.pop("kinetic", None)
                log.append(f"{shot.get('shot_id')}: kinetic subtitles set back to regular (usage limit)")
                run = 0
            else:
                used += 1
                run += 1
        else:
            run = 0
    return log


def repair_plan(plan: Dict[str, Any], run_id: str = "", theme: Any = None, record: bool = True) -> List[str]:
    """All the safe repairs, in order. Returns what was changed."""
    subtitles = _theme_get(theme, "subtitles", {})
    log = repair_media(plan)
    log += repair_cutouts(plan)
    for shot in plan.get("shots", []):
        shot["layout"] = derive_layout(shot)
    log += repair_transitions(plan)
    log += repair_assets(plan, run_id, record)
    log += repair_kinetic(plan, float(subtitles.get("kinetic_max_share", DEFAULT_KINETIC_SHARE)),
                          int(subtitles.get("kinetic_max_run", DEFAULT_KINETIC_RUN)))
    return log


def assign_defaults(plan: Dict[str, Any], theme: Any = None) -> None:
    """Fill the mechanical fields the director left empty, from the theme's lists, rotating for variety."""
    motion = _theme_get(theme, "motion", {})
    image_motions = [m for m in motion.get("image", []) if m in vc.IMAGE_MOTIONS] or ["slow_zoom_in"]
    text_motions = [m for m in motion.get("text", []) if m in vc.TEXT_MOTIONS] or ["fade"]
    easing = motion.get("easing") if motion.get("easing") in vc.EASINGS else "ease_in_out_cubic"
    transitions = [t for t in _theme_get(theme, "transitions", []) if ac.exists("transition", t)] or ["cut"]
    overlays = [o for o in _theme_get(theme, "overlays", []) if ac.exists("overlay", o)][:2]
    count = 0
    for index, shot in enumerate(plan.get("shots", [])):
        for media in shot.get("media") or []:
            if not media.get("motion"):
                media["motion"] = image_motions[count % len(image_motions)]
                count += 1
        for item in shot.get("text") or []:
            if not item.get("motion"):
                item["motion"] = text_motions[index % len(text_motions)]
        shot.setdefault("easing", easing)
        if index and not shot.get("transition_in"):
            shot["transition_in"] = transitions[index % len(transitions)]
        if "overlays" not in shot:
            shot["overlays"] = list(overlays)


# ----------------------------------------------------------------------
# Everything that is wrong with a plan
# ----------------------------------------------------------------------

def check_plan(plan: Dict[str, Any], findings: List[str], banned: List[str], theme: Any = None,
               narration: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    """Every problem a plan has: [{severity, shot_id, beat_id, check, problem, suggestion}]."""
    issues: List[Dict[str, str]] = []

    def add(severity: str, shot: Dict[str, Any], check: str, problem: str, suggestion: str = "") -> None:
        issues.append({"severity": severity, "shot_id": str(shot.get("shot_id", "")),
                       "beat_id": str(shot.get("beat_id", "")), "check": check,
                       "problem": problem, "suggestion": suggestion})

    shots = plan.get("shots", [])
    grouped: Dict[str, List[str]] = {}               # a warning repeats for every image: say it once
    for shot in shots:
        for media in shot.get("media") or []:
            if media.get("source") == "reuse":         # a reuse shows an image that is checked where it is made
                continue
            for severity, message in check_brief(str(media.get("brief", "")), banned):
                if severity == "error":
                    add(severity, shot, "brief", f"{media.get('media_id')}: {message}")
                else:
                    grouped.setdefault(message, []).append(str(media.get("media_id")))
        for element in shot.get("elements") or []:
            for message in check_element(element, findings):
                add("error", shot, "element", message)
        for item in shot.get("text") or []:
            content = str(item.get("content", "")).strip()
            if content and _CUT_OFF.search(content):
                add("error", shot, "text", f"the {item.get('kind')} text looks cut off: ...{content[-30:]!r}",
                    "Write a complete short phrase instead of cutting a sentence.")
            words_here = content_words(content)
            if narration and item.get("kind") == "explanatory" and len(words_here) >= 3:
                spoken = [s for s in re.split(r"(?<=[.!?])\s+", narration.get(str(shot.get("beat_id")), "")) if s.strip()]
                if any(len(words_here & content_words(s)) / len(words_here) >= 0.7 for s in spoken):
                    add("warning", shot, "text", "the explanatory text repeats the narration, which the subtitles already show",
                        "Say something the narration does not, or leave the text out.")
        load = check_load(shot)
        if not load["fits"]:
            add("error", shot, "load", "; ".join(load["notes"]),
                "Ask the Story Director to split this beat into shots, one part per shot.")
    for message, ids in grouped.items():
        issues.append({"severity": "warning", "shot_id": "", "beat_id": "", "check": "brief",
                       "problem": f"{len(ids)} images: {message} ({', '.join(ids)})", "suggestion": ""})
    subtitles = _theme_get(theme, "subtitles", {})
    for problem in check_kinetic(shots, float(subtitles.get("kinetic_max_share", DEFAULT_KINETIC_SHARE)),
                                 int(subtitles.get("kinetic_max_run", DEFAULT_KINETIC_RUN))):
        issues.append({"severity": "warning", "shot_id": "", "beat_id": "", "check": "kinetic",
                       "problem": problem, "suggestion": ""})
    return issues


def check_image_prompts(plan: Dict[str, Any], prompts: Dict[str, Dict[str, Any]], banned: List[str]) -> List[Dict[str, str]]:
    """Problems with the written image prompts: [{severity, shot_id, beat_id, check, problem, suggestion}]."""
    issues: List[Dict[str, str]] = []

    def add(severity: str, shot: Dict[str, Any], problem: str, suggestion: str = "") -> None:
        issues.append({"severity": severity, "shot_id": str(shot.get("shot_id", "")),
                       "beat_id": str(shot.get("beat_id", "")), "check": "prompt",
                       "problem": problem, "suggestion": suggestion})

    earlier: List[Tuple[str, str, str]] = []             # (media_id, shot_id, subject) of new images so far
    for shot in plan.get("shots", []):
        for media in shot.get("media") or []:
            if media.get("source") == "reuse":
                continue
            mid = str(media.get("media_id"))
            prompt = prompts.get(mid)
            if prompt is None:
                add("error", shot, f"{mid}: no prompt was written for this image", "Run the Image Prompter.")
                continue
            if prompt.get("blocked"):
                add("error", shot, f"{mid}: {prompt['blocked']}", "Change this image's brief, or its shot.")
                continue
            subject = str(prompt.get("subject") or prompt.get("positive", ""))
            low = subject.lower()
            hit = next((t for t in banned if re.search(rf"\b{re.escape(t)}\b", low)), None)
            if hit:
                add("error", shot, f"{mid}: the prompt shows the cast ({hit!r})")
            writing = _WRITING.search(subject)
            if writing:
                add("error", shot, f"{mid}: the prompt asks for {writing.group(0)!r}")
            if _COMBINED.search(subject):
                add("warning", shot, f"{mid}: the prompt puts two subjects in one picture, which often blend",
                    "Use two separate pictures.")
            brief_words = content_words(str(media.get("brief", "")))
            if brief_words and len(brief_words & content_words(subject)) / len(brief_words) < 0.4:
                add("warning", shot, f"{mid}: the prompt drifts from its brief",
                    "Check that the picture still shows what the shot needs.")
            if prompt.get("concern"):
                add("warning", shot, f"{mid}: {prompt['concern']}")
            if media.get("source") == "generate":
                twin = next((e for e in earlier if e[1] != shot.get("shot_id") and similarity(str(media.get("brief", "")), e[2]) >= 0.5), None)
                if twin:
                    add("warning", shot, f"{mid} looks very like {twin[0]}",
                        f"Make it a variation of {twin[0]} (image-to-image) or reuse it, so the pictures stay consistent.")
            earlier.append((mid, str(shot.get("shot_id")), str(media.get("brief", ""))))
    return issues


def _images(plan: Dict[str, Any]) -> int:
    return sum(1 for s in plan.get("shots", []) for m in s.get("media") or [] if m.get("source") != "reuse")


def _main() -> None:
    import argparse

    from core.registry_loader import RegistryLoader
    from core.run_context import RunContext

    parser = argparse.ArgumentParser(description="Dry-run the visual rules on a saved plan (nothing is saved).")
    parser.add_argument("--from-run", required=True, help="path to a run's shared_context.json")
    args = parser.parse_args()

    ctx = RunContext.load(args.from_run)
    saved = ctx.read_stage_output("visual_plan")
    if not saved:
        raise SystemExit("This run has no visual_plan stage output.")
    plan = copy.deepcopy(saved)
    findings = (ctx.read_stage_output("researcher") or {}).get("findings", [])
    registry = RegistryLoader()
    channel_id = ctx.channel_id
    channel = registry.get_channel(channel_id) if channel_id else {}
    theme = registry.get_theme(channel["theme_id"]) if channel.get("theme_id") else None
    banned = ["avatar"]
    for member in (registry.character_menu(channel_id) if channel_id else []):
        banned += [member["id"].lower(), str(member["name"]).lower()]
        banned += [str(p["pose_id"]).lower() for p in registry.pose_menu(member["id"]) if p.get("pose_id")]

    before = _images(plan)
    log = repair_plan(plan, ctx.run_id, theme, record=False)
    assign_defaults(plan, theme)
    narration: Dict[str, str] = {}
    for section in (ctx.read_stage_output("story_director") or {}).get("sections", []):
        beat_id = str(section.get("beat_id", ""))
        narration[beat_id] = (narration.get(beat_id, "") + " " + str(section.get("text", ""))).strip()
    issues = check_plan(plan, findings, banned, theme, narration)

    print(f"Images to generate: {before} -> {_images(plan)} after the repairs\n")
    print(f"Repairs made in code ({len(log)}):")
    for line in log:
        print("  -", line)
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] != "error"]
    print(f"\nProblems for the Visual Director to fix ({len(errors)} errors, {len(warnings)} warnings):")
    for issue in errors + warnings:
        print(f"  [{issue['severity']:<7}] {issue['shot_id'] or 'plan':<12}{issue['check']:<8}{issue['problem']}")


if __name__ == "__main__":
    _main()