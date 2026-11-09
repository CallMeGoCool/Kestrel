"""
Kestrel Shorts - Scene compiler
===============================

The code half of the Frame Supervisor. It turns everything the directors decided into one SCENE FILE per beat,
the only thing Remotion has to read: no guessing, no layout maths and no decisions left for the renderer.

  Time        every shot starts when its cue phrase is spoken (found in the real word times of the voice track);
              a shot without a found cue is spaced evenly between its neighbours; none is shorter than
              MIN_SHOT_SECONDS. Shots and the beat are in frames (30 per second) counted from the start of the beat;
              layers inside a shot count from the start of that shot.
  Space       every rectangle comes from engines/visual/layout_spec.py (safe area, image slots, text fitting,
              the avatar's visible pixels, the subtitle strip), so Remotion never decides where anything goes.
  Pictures    from the asset analyzer: where the eye goes (focus) so crops land on the subject, and whether a
              landscape picture becomes a framed card instead of a heavy crop of a portrait screen.
  Sound       the voice file of the beat, music and ambience beds (with a loudness point per sentence), and the sound
              effects, all as frames.
  Notes       anything the compiler had to compromise on (an overloaded shot, an avatar with no free corner).

Human changes arrive later as an overrides file per beat; the scene is always rebuilt from plan + overrides.

    python -m engines.video.scene_compiler --from-run <shared_context.json> [--beat beat_1] [--no-stage]

Writes runs/<run>/scenes/scene_<beat>.json, scene_index.json and preview_<beat>.png (a wireframe of every shot).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core import config
from core.geometry import Rect
from engines.visual import asset_analyzer as aa
from engines.visual import layout_spec as ls

FPS = 30
MIN_SHOT_SECONDS = 1.2
TRANSITION_FRAMES = {"cut": 0, "fade": 10, "slide": 12, "wipe": 12}
TRANSITION_FALLBACK = {"zoom_through": "fade", "page_tear": "wipe"}          # not drawn yet
CARD_MAX_WORDS, CARD_MAX_CHARS, KINETIC_MAX_WORDS = 4, 26, 2
CARD_TAIL_FRAMES = 6                                                       # a card stays a moment after its last word
TEXT_DELAY_S = {"attention": 0.15, "explanatory": 0.7}                     # appears this long after the shot starts
TEXT_HOLD_S = {"attention": 2.8, "explanatory": 5.0}                       # and stays at most this long
STAGGER_FRAMES = 6                                                         # between pictures of a stack or grid
CARD_ASPECT = 0.85                                                         # wider pictures become framed cards
TEXT_PAD = 28
MIN_ELEMENT_HEIGHT = 420                                                    # an element needs at least this much height
AVATAR_ANCHORS = (("bottom_right", "lower_right", "bottom_left"), ("bottom_left", "lower_left", "bottom_right"))
IMAGE_MOTION: Dict[str, Dict[str, Any]] = {           # the catalog's subject motions plus camera moves
    "static": {"scale": (1.0, 1.0), "pan": ((0, 0), (0, 0))},
    "shake_light": {"scale": (1.03, 1.03), "pan": ((0, 0), (0, 0)), "shake": 0.004},
    "wobble": {"scale": (1.0, 1.0), "pan": ((0, 0), (0, 0)), "wobble": 2.0},
    "bob": {"scale": (1.0, 1.0), "pan": ((0, 0), (0, 0)), "bob": 0.012},
    "float": {"scale": (1.0, 1.04), "pan": ((0, 0.01), (0, -0.01)), "bob": 0.008},
    "pop_in": {"scale": (1.0, 1.0), "pan": ((0, 0), (0, 0)), "enter": "pop_in"},
    "slide_in": {"scale": (1.0, 1.0), "pan": ((0, 0), (0, 0)), "enter": "slide_in"},
    "slow_zoom_in": {"scale": (1.0, 1.12), "pan": ((0, 0), (0, 0))},
    "slow_zoom_out": {"scale": (1.12, 1.0), "pan": ((0, 0), (0, 0))},
    "pan_left": {"scale": (1.12, 1.12), "pan": ((0.03, 0), (-0.03, 0))},
    "pan_right": {"scale": (1.12, 1.12), "pan": ((-0.03, 0), (0.03, 0))},
    "pan_up": {"scale": (1.12, 1.12), "pan": ((0, 0.03), (0, -0.03))},
    "pan_down": {"scale": (1.12, 1.12), "pan": ((0, -0.03), (0, 0.03))},
    "drift": {"scale": (1.06, 1.1), "pan": ((-0.012, 0.01), (0.012, -0.01))},
}
DEFAULT_PALETTE = {"background": "#14161B", "surface": "#1F2229", "accent": "#E3A93B", "accent_alt": "#4FA3A5",
                   "highlight": "#F2F2F2", "text": "#F4F1EA", "text_dark": "#14161B"}


class SceneError(RuntimeError):
    pass


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(word).lower())


def _rect(r: Rect) -> Dict[str, float]:
    return {"x": round(r.x, 1), "y": round(r.y, 1), "w": round(r.width, 1), "h": round(r.height, 1)}


# ----------------------------------------------------------------------
# Time: words and shot starts
# ----------------------------------------------------------------------

def beat_words(timing: Dict[str, Any], voice_lines: Dict[str, Dict[str, Any]], beat_id: str) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for unit in timing["units"]:
        if unit["beat_id"] != beat_id:
            continue
        lean = {_norm(w) for phrase in (voice_lines.get(unit["unit_id"], {}).get("emphasis") or []) for w in phrase.split()}
        for w in unit["words"]:
            words.append({"text": w["text"], "norm": _norm(w["text"]), "start": w["start"], "end": w["end"],
                          "unit_id": unit["unit_id"], "emphasis": _norm(w["text"]) in lean})
    return words


def _find_cue(words: List[Dict[str, Any]], cue: str, after: int) -> Optional[Tuple[int, int]]:
    """(index of the first word, number of words matched) of the cue in the narration, searching from `after`.
    The whole phrase is tried first, then its first three words, then its first two."""
    want = [t for t in (_norm(x) for x in cue.split()) if t]
    for size in dict.fromkeys((len(want), min(3, len(want)), min(2, len(want)))):
        part = want[:size]
        if not part:
            continue
        for i in range(after, len(words) - len(part) + 1):
            if [w["norm"] for w in words[i:i + len(part)]] == part:
                return i, len(part)
    return None


def shot_starts(shots: List[Dict[str, Any]], words: List[Dict[str, Any]], start_s: float, end_s: float,
                notes: List[str]) -> List[float]:
    n = len(shots)
    starts: List[Optional[float]] = [None] * n
    starts[0] = start_s
    cursor = 0
    for i in range(1, n):
        cue = str(shots[i].get("cue") or "").strip()
        if not cue:
            continue
        found = _find_cue(words, cue, cursor)
        if found is None:
            notes.append(f"{shots[i].get('shot_id')}: cue {cue!r} not found in the narration, shot spaced evenly")
        else:
            starts[i] = words[found[0]]["start"]
            cursor = found[0] + found[1]
    i = 1
    while i < n:
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        left = starts[i - 1]
        right = starts[j] if j < n else end_s
        for k in range(i, j):
            starts[k] = left + (right - left) * (k - i + 1) / (j - i + 1)
        i = j
    out = [float(s) for s in starts]                                    # type: ignore[arg-type]
    if (end_s - start_s) < MIN_SHOT_SECONDS * n:                         # too short for every shot: spread evenly
        return [start_s + (end_s - start_s) * k / n for k in range(n)]
    for k in range(1, n):
        out[k] = max(out[k], out[k - 1] + MIN_SHOT_SECONDS)
    for k in range(n - 1, 0, -1):
        limit = (out[k + 1] if k + 1 < n else end_s) - MIN_SHOT_SECONDS
        out[k] = min(out[k], limit)
    for k in range(1, n):
        out[k] = max(out[k], out[k - 1] + 0.001)
    return out


# ----------------------------------------------------------------------
# Pictures
# ----------------------------------------------------------------------

def _owner(media_id: str, by_id: Dict[str, Dict[str, Any]]) -> str:
    seen: set = set()
    while by_id.get(media_id, {}).get("source") == "reuse" and by_id[media_id].get("ref") in by_id and media_id not in seen:
        seen.add(media_id)
        media_id = by_id[media_id]["ref"]
    return media_id


def _image_file(images_dir: Path, owner_id: str, cutout: bool) -> Optional[Path]:
    files = sorted(images_dir.glob(f"{owner_id}_*.png"))
    cut = [f for f in files if f.stem.endswith("_cutout")]
    plain = [f for f in files if not f.stem.endswith("_cutout")]
    if cutout and cut:
        return cut[0]
    return plain[0] if plain else (cut[0] if cut else None)


def _motion(name: str, tile: bool, easing: str, hold: int) -> Dict[str, Any]:
    spec = IMAGE_MOTION.get(name, IMAGE_MOTION["static"])
    (s0, s1), (p0, p1) = spec["scale"], spec["pan"]
    k = 0.5 if tile else 1.0                                              # smaller moves on small pictures
    out = {"name": name if name in IMAGE_MOTION else "static", "scale": [round(1 + (s0 - 1) * k, 3), round(1 + (s1 - 1) * k, 3)],
           "pan": [[round(p0[0] * k, 4), round(p0[1] * k, 4)], [round(p1[0] * k, 4), round(p1[1] * k, 4)]],
           "easing": easing, "frames": hold}
    out.update({key: spec[key] for key in ("bob", "wobble", "shake") if key in spec})
    return out


def _cutout_rect(analysis: Dict[str, Any]) -> Rect:
    """Where the whole PNG goes so that its visible part is centred in the content area, about 80% of its width."""
    content = ls.content_rect()
    x0, y0, x1, y1 = analysis["alpha_box"]
    vis_w, vis_h = (x1 - x0), (y1 - y0)
    width_px, height_px = analysis["size"]
    scale = min(content.width * 0.8 / (vis_w * width_px), content.height * 0.7 / (vis_h * height_px))
    w, h = width_px * scale, height_px * scale
    cx, cy = content.x + content.width / 2, content.y + content.height * 0.46
    return Rect(cx - (x0 + vis_w / 2) * w, cy - (y0 + vis_h / 2) * h, w, h)


# ----------------------------------------------------------------------
# Subtitles
# ----------------------------------------------------------------------

def make_cards(words: List[Dict[str, Any]], shot_from: int, shot_frames: int, beat_start_s: float, style: str) -> List[Dict[str, Any]]:
    limit = KINETIC_MAX_WORDS if style == "kinetic" else CARD_MAX_WORDS
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for w in words:
        current.append(w)
        text_len = sum(len(x["text"]) for x in current) + len(current) - 1
        if len(current) >= limit or text_len > CARD_MAX_CHARS or (len(current) >= 2 and re.search(r"[.,;:?!\u2014]$", w["text"])):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    def frame(t: float) -> int:
        return round((t - beat_start_s) * FPS) - shot_from

    cards = []
    for i, group in enumerate(groups):
        begin = max(0, frame(group[0]["start"]))
        nxt = frame(groups[i + 1][0]["start"]) if i + 1 < len(groups) else frame(group[-1]["end"]) + CARD_TAIL_FRAMES
        end = max(begin + 3, min(nxt, shot_frames))
        cards.append({"from": begin, "to": end, "words": [
            {"text": w["text"], "from": max(0, frame(w["start"])), "to": max(0, frame(w["end"])), "emphasis": w["emphasis"]} for w in group]})
    return cards


# ----------------------------------------------------------------------
# One shot
# ----------------------------------------------------------------------

def compile_shot(shot: Dict[str, Any], frames: int, from_frame: int, ctx: Dict[str, Any], words: List[Dict[str, Any]],
                 beat_start_s: float, notes: List[str]) -> Dict[str, Any]:
    palette, plan_by_id, images_dir = ctx["palette"], ctx["media_by_id"], ctx["images_dir"]
    layout = str(shot.get("layout") or "single")
    kind = layout.split("_")[0]
    easing = shot.get("easing") or "ease_in_out_cubic"
    sid = str(shot.get("shot_id"))
    layers: List[Dict[str, Any]] = []
    blockers: List[Rect] = []                                            # text and elements: subtitles must not cover them

    # ---- backdrop
    backdrop = shot.get("backdrop") or {"type": "flat", "color": "background"}
    color_name = backdrop.get("color", "background")
    shot_backdrop = {"type": backdrop.get("type", "flat"), "color": palette.get(color_name, palette["background"]),
                     "color_alt": palette.get("accent", "#E3A93B")}

    # ---- text, measured and placed from the top of the content area
    content = ls.content_rect()
    texts = [t for t in (shot.get("text") or []) if str(t.get("content", "")).strip()]
    cursor_y = content.y
    text_layers: List[Dict[str, Any]] = []
    for item in sorted(texts, key=lambda t: 0 if t.get("kind") == "attention" else 1):
        tkind = "attention" if item.get("kind") == "attention" else "explanatory"
        region = Rect(content.x, cursor_y, content.width, min(content.height * (0.2 if tkind == "attention" else 0.16), content.bottom - cursor_y))
        fit = ls.fit_text(str(item["content"]).strip(), tkind, region)
        width = min(region.width, fit["width"] + 2 * TEXT_PAD)
        box = ls.anchor_rect("top_center", width, min(region.height, fit["height"] + 2 * TEXT_PAD), region)
        if not fit["fits"]:
            notes.append(f"{sid}: the {tkind} text does not fit at its smallest size")
        delay = round(TEXT_DELAY_S[tkind] * FPS)
        text_layers.append({"type": "text", "kind": tkind, "content": str(item["content"]).strip(), "lines": fit["lines"],
                            "font_px": fit["font_px"], "rect": _rect(box), "from": min(delay, max(0, frames - 6)),
                            "frames": min(round(TEXT_HOLD_S[tkind] * FPS), frames), "motion": item.get("motion") or "fade",
                            "panel": tkind == "explanatory"})
        blockers.append(box)
        cursor_y = box.bottom + ls.GUTTER

    # ---- avatar: placed before the elements, so an element never lands under it
    avatar = shot.get("avatar")
    avatar_layer: Optional[Dict[str, Any]] = None
    bottom_limit = content.bottom
    if avatar and ctx.get("avatar_loader"):
        try:
            geometry, path = ctx["avatar_loader"](avatar["character_id"], avatar.get("pose_id"))
            order = ctx["cast"].index(avatar["character_id"]) if avatar["character_id"] in ctx["cast"] else 0
            placed = None
            sizes = [str(avatar.get("size") or "medium")]
            if shot.get("elements") and "small" not in sizes:
                sizes.append("small")                                   # an element needs room: try a smaller avatar
            for size in sizes:
                for anchor in AVATAR_ANCHORS[order % 2]:
                    p = ls.avatar_placement(geometry, anchor, size)
                    if any(p["visible"].intersects(b) for b in blockers):
                        continue
                    if shot.get("elements") and p["visible"].y - ls.GUTTER - cursor_y < MIN_ELEMENT_HEIGHT:
                        continue
                    placed = (anchor, p)
                    break
                if placed:
                    break
            if placed is None:
                notes.append(f"{sid}: no free corner or no room beside the element, avatar left out")
            else:
                anchor, p = placed
                if shot.get("elements"):
                    bottom_limit = p["visible"].y - ls.GUTTER
                if placed is not None:
                    avatar_layer = {"type": "avatar", "character_id": avatar["character_id"], "pose_id": avatar.get("pose_id"),
                                    "src": str(path), "rect": _rect(p["canvas"]), "visible": _rect(p["visible"]), "anchor": anchor,
                                    "motion": "bob", "from": 0, "frames": frames}
        except Exception as exc:
            notes.append(f"{sid}: avatar skipped ({type(exc).__name__}: {exc})")

    # ---- elements below the text
    elements: List[Dict[str, Any]] = []
    for element in shot.get("elements") or []:
        region = Rect(content.x, cursor_y, content.width, max(MIN_ELEMENT_HEIGHT, bottom_limit - cursor_y))
        entry = {"type": "element", "kind": element.get("kind"), "id": element.get("element_id") or element.get("id"),
                 "content": element.get("content") or {}, "rect": _rect(region), "from": round(0.3 * FPS), "frames": frames,
                 "easing": easing}
        body = entry["content"]
        items = body.get("steps") or body.get("items") or []
        if items and hasattr(ls, "list_fit"):
            try:
                fit = ls.list_fit([str(x) for x in items], region)
                entry["font_px"], entry["fits"] = fit.get("font_px"), fit.get("fits", True)
                if not fit.get("fits", True):
                    notes.append(f"{sid}: the {element.get('kind')} element does not fit its area")
            except Exception as exc:                                    # a measuring problem must not stop the compile
                notes.append(f"{sid}: could not measure the {element.get('kind')} element ({exc})")
        elements.append(entry)
        blockers.append(region)

    # ---- pictures
    media = list(shot.get("media") or [])
    picture_layers: List[Dict[str, Any]] = []
    prepared: List[Tuple[Dict[str, Any], Path, Dict[str, Any], bool]] = []
    for m in media:
        owner = _owner(m["media_id"], plan_by_id)
        cut = bool(plan_by_id.get(owner, {}).get("cutout"))
        path = _image_file(images_dir, owner, cut)
        if path is None:
            notes.append(f"{sid}: no picture file for {m['media_id']} (made from {owner})")
            continue
        prepared.append((m, path, aa.analyze(path), cut))
    aspects = [a["aspect"] for _, _, a, _ in prepared] or [2 / 3]
    if prepared:
        mean_aspect = sum(aspects) / len(aspects)
        slots = ls.slots(layout, len(prepared), mean_aspect)
        tile = kind in ("stack", "grid", "scatter", "inset") and len(prepared) > 1
        for i, (m, path, analysis, cut) in enumerate(prepared):
            slot = slots[min(i, len(slots) - 1)]
            rect, fit_mode, card = slot["rect"], "cover", False
            if cut:
                rect, fit_mode = _cutout_rect(analysis), "contain"
            elif kind == "single" and analysis["aspect"] >= CARD_ASPECT:
                width = ls.media_rect().width
                height = min(ls.media_rect().height, width / analysis["aspect"])
                rect = ls.anchor_rect("center", width, height, ls.media_rect())
                card = True
            motion_name = str(m.get("motion") or "static")
            if motion_name not in IMAGE_MOTION:
                notes.append(f"{sid}: picture motion {motion_name!r} is not drawn yet, using static")
            start = 0
            span = frames
            if kind == "sequence" and len(prepared) > 1:
                span = frames // len(prepared)
                start = i * span
                span = frames - start if i == len(prepared) - 1 else span + STAGGER_FRAMES
            elif tile:
                start = min(i * STAGGER_FRAMES, max(0, frames - 6))
                span = frames - start
            picture_layers.append({"type": "image", "id": m["media_id"], "src": str(path), "rect": _rect(rect), "rotation": slot.get("rotation", 0.0),
                                   "fit": fit_mode, "card": card, "cutout": cut, "focus": analysis["focus"],
                                   "motion": _motion(motion_name, tile or card, easing, span),
                                   "from": start, "frames": span,
                                   "enter": IMAGE_MOTION.get(motion_name, {}).get("enter") or ("pop_in" if cut else "fade"),
                                   "tone": analysis["tone"]})
    layers = picture_layers + text_layers + elements + ([avatar_layer] if avatar_layer else [])

    # ---- subtitles
    style = shot.get("subtitle_style") or "regular"
    slot = ls.kinetic_rect() if style == "kinetic" else ls.subtitle_slot(blockers)
    if slot is None:
        slot = ls.subtitle_rect()
        notes.append(f"{sid}: no free strip for the subtitles, the shot is overloaded (they will lie over content)")
    in_shot = [w for w in words if from_frame <= round((w["start"] - beat_start_s) * FPS) < from_frame + frames]
    subtitles = {"style": style, "rect": _rect(slot), "cards": make_cards(in_shot, from_frame, frames, beat_start_s, style)}

    transition = shot.get("transition_in") or "cut"
    transition = TRANSITION_FALLBACK.get(transition, transition)
    return {"shot_id": sid, "from": from_frame, "frames": frames, "layout": layout, "intent": shot.get("intent", ""),
            "transition_in": {"type": transition, "frames": min(TRANSITION_FRAMES.get(transition, 10), frames // 3)},
            "backdrop": shot_backdrop, "layers": layers, "subtitles": subtitles, "overlays": list(shot.get("overlays") or []),
            "easing": easing}


# ----------------------------------------------------------------------
# Audio of a beat
# ----------------------------------------------------------------------

def _audio_src(kind: str, file: str) -> str:
    """The path of a library sound relative to Remotion's public folder (a bare file name lives in audio/<kind>/)."""
    if not file or "/" in file or "\\" in file:
        return file
    return f"audio/{kind}/{file}"


def beat_audio(timing: Dict[str, Any], beat_id: str, total_frames: int, beat_order: List[str], run_dir: Path) -> Dict[str, Any]:
    beat = timing["beats"][beat_id]
    start_s, end_s = beat["start"], beat["end"]
    events = timing.get("events", {})
    index = beat_order.index(beat_id)

    def frame(t: float) -> int:
        return round((t - start_s) * FPS)

    def beds(kind: str) -> List[Dict[str, Any]]:
        out = []
        for run in events.get(kind, []):
            if beat_order.index(run["from_beat"]) <= index <= beat_order.index(run["to_beat"]):
                out.append({"asset": run["asset"], "src": _audio_src(kind, run.get("file", "")), "offset_s": round(start_s - run["start"], 3),
                            "fade_in": run["fade_in"] if run["from_beat"] == beat_id else 0.0,
                            "fade_out": run["fade_out"] if run["to_beat"] == beat_id else 0.0})
        return out
    music = beds("music")
    points = [{"frame": max(0, frame(u["start"])), "db": timing["music_db"].get(u["unit_id"], -24.0)}
              for u in timing["units"] if u["beat_id"] == beat_id]
    voice_file = str(run_dir / beat["file"]) if beat.get("file") else ""
    return {"voice": {"src": voice_file, "db": timing.get("mix", {}).get("voice_db", -3.0)},
            "music": music, "ambience": beds("ambience"), "music_curve": points,
            "duck_db": timing.get("mix", {}).get("duck_db", -9.0),
            "sfx": [{"asset": e["asset"], "src": _audio_src("sfx", e.get("file", "")), "at": frame(e["at"]), "db": e.get("db", -24.0)}
                    for e in events.get("sfx", []) if start_s <= e["at"] < end_s],
            "dynamics": [{"type": e["type"], "at": frame(e["at"]), "frames": round(e["seconds"] * FPS), "db": e["db"]}
                         for e in events.get("dynamics", []) if start_s <= e["at"] < end_s]}


# ----------------------------------------------------------------------
# A whole beat, and a whole run
# ----------------------------------------------------------------------

def compile_beat(beat_id: str, plan: Dict[str, Any], timing: Dict[str, Any], voice_plan: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    shots = [s for s in plan["shots"] if s.get("beat_id") == beat_id]
    if not shots:
        raise SceneError(f"Beat {beat_id} has no shots in the visual plan.")
    voice_lines = {l["unit_id"]: l for l in voice_plan.get("lines", [])}
    words = beat_words(timing, voice_lines, beat_id)
    beat = timing["beats"][beat_id]
    start_s, end_s = beat["start"], beat["end"]
    total = round((end_s - start_s) * FPS)
    notes: List[str] = []
    starts = shot_starts(shots, words, start_s, end_s, notes)
    froms = [round((s - start_s) * FPS) for s in starts]
    compiled = []
    for i, shot in enumerate(shots):
        end_frame = froms[i + 1] if i + 1 < len(shots) else total
        compiled.append(compile_shot(shot, max(1, end_frame - froms[i]), froms[i], ctx, words, start_s, notes))
    return {"version": 1, "run_id": ctx["run_id"], "beat_id": beat_id, "fps": FPS, "width": ls.CANVAS.width, "height": ls.CANVAS.height,
            "start_s": round(start_s, 3), "frames": total, "theme": ctx["theme_subset"], "layout": ls.spec(),
            "shots": compiled, "audio": beat_audio(timing, beat_id, total, ctx["beat_order"], Path(ctx["run_dir"])), "notes": notes}


def stage_assets(scene: Dict[str, Any], run_id: str, public_dir: Path) -> None:
    """Copy the files a scene uses into Remotion's public folder and point `src` at them (relative, for staticFile)."""
    target = public_dir / "runs" / run_id
    for shot in scene["shots"]:
        for layer in shot["layers"]:
            if layer.get("type") in ("image", "avatar") and layer.get("src") and Path(layer["src"]).is_file():
                folder = "images" if layer["type"] == "image" else "avatars"
                (target / folder).mkdir(parents=True, exist_ok=True)
                shutil.copy2(layer["src"], target / folder / Path(layer["src"]).name)
                layer["src"] = f"runs/{run_id}/{folder}/{Path(layer['src']).name}"
    voice = scene["audio"]["voice"]
    if voice.get("src") and Path(voice["src"]).is_file():
        (target / "audio").mkdir(parents=True, exist_ok=True)
        shutil.copy2(voice["src"], target / "audio" / Path(voice["src"]).name)
        voice["src"] = f"runs/{run_id}/audio/{Path(voice['src']).name}"


def compile_run(plan: Dict[str, Any], timing: Dict[str, Any], voice_plan: Dict[str, Any], ctx: Dict[str, Any],
                only: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    scenes = {}
    for beat_id in timing["beats"]:
        if only and beat_id != only:
            continue
        scenes[beat_id] = compile_beat(beat_id, plan, timing, voice_plan, ctx)
    return scenes


# ----------------------------------------------------------------------
# Wireframe preview: see the layout of every shot without rendering anything
# ----------------------------------------------------------------------

def preview(scene: Dict[str, Any], out: Path, scale: float = 0.3) -> None:
    from PIL import Image, ImageDraw
    W, H = round(scene["width"] * scale), round(scene["height"] * scale)
    shots = scene["shots"]
    sheet = Image.new("RGB", (len(shots) * (W + 10) + 10, H + 56), (12, 12, 14))
    for i, shot in enumerate(shots):
        panel = Image.new("RGB", (W, H), shot["backdrop"]["color"])
        draw = ImageDraw.Draw(panel)
        for layer in shot["layers"]:
            r = layer["rect"]
            box = (round(r["x"] * scale), round(r["y"] * scale), round((r["x"] + r["w"]) * scale), round((r["y"] + r["h"]) * scale))
            if layer["type"] == "image" and Path(layer["src"]).is_file():
                picture = Image.open(layer["src"]).convert("RGBA").resize((max(1, box[2] - box[0]), max(1, box[3] - box[1])))
                if layer["fit"] == "cover":
                    full = Image.open(layer["src"]).convert("RGBA")
                    ratio = max((box[2] - box[0]) / full.width, (box[3] - box[1]) / full.height)
                    resized = full.resize((round(full.width * ratio), round(full.height * ratio)))
                    fx, fy = layer["focus"]
                    left = round(min(max(0, fx * resized.width - (box[2] - box[0]) / 2), resized.width - (box[2] - box[0])))
                    top = round(min(max(0, fy * resized.height - (box[3] - box[1]) / 2), resized.height - (box[3] - box[1])))
                    picture = resized.crop((left, top, left + box[2] - box[0], top + box[3] - box[1]))
                panel.paste(picture.convert("RGB"), box[:2], picture.split()[3])
                draw.rectangle(box, outline=(90, 160, 90))
            elif layer["type"] == "text":
                draw.rectangle(box, fill=(40, 40, 46) if layer["panel"] else None, outline=(230, 190, 60))
                draw.text((box[0] + 4, box[1] + 3), layer["content"][:34], fill=(255, 255, 255))
            elif layer["type"] == "element":
                draw.rectangle(box, outline=(90, 150, 230))
                draw.text((box[0] + 4, box[1] + 3), f"{layer['kind']}: " + " | ".join(str(x)[:18] for x in (layer["content"].get("steps") or layer["content"].get("items") or [])[:4]), fill=(150, 190, 250))
            elif layer["type"] == "avatar":
                v = layer["visible"]
                vb = (round(v["x"] * scale), round(v["y"] * scale), round((v["x"] + v["w"]) * scale), round((v["y"] + v["h"]) * scale))
                if Path(layer["src"]).is_file():
                    pose = Image.open(layer["src"]).convert("RGBA").resize((max(1, box[2] - box[0]), max(1, box[3] - box[1])))
                    panel.paste(pose.convert("RGB"), box[:2], pose.split()[3])
                draw.rectangle(vb, outline=(220, 90, 90))
        s = shot["subtitles"]["rect"]
        sb = (round(s["x"] * scale), round(s["y"] * scale), round((s["x"] + s["w"]) * scale), round((s["y"] + s["h"]) * scale))
        draw.rectangle(sb, outline=(240, 240, 240))
        if shot["subtitles"]["cards"]:
            draw.text((sb[0] + 4, sb[1] + 4), " ".join(w["text"] for w in shot["subtitles"]["cards"][0]["words"]), fill=(255, 255, 255))
        sf = ls.safe_rect()
        draw.rectangle((round(sf.x * scale), round(sf.y * scale), round(sf.right * scale), round(sf.bottom * scale)), outline=(80, 80, 90))
        sheet.paste(panel, (10 + i * (W + 10), 8))
        d2 = ImageDraw.Draw(sheet)
        d2.text((12 + i * (W + 10), H + 14), f"{shot['shot_id']}  {shot['from'] / FPS:.1f}s +{shot['frames'] / FPS:.1f}s  {shot['layout']}", fill=(220, 220, 220))
        d2.text((12 + i * (W + 10), H + 30), f"in: {shot['transition_in']['type']}  subs: {shot['subtitles']['style']}", fill=(150, 150, 160))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def _summary(scene: Dict[str, Any]) -> None:
    print(f"\n== {scene['beat_id']}: {scene['frames']} frames ({scene['frames'] / FPS:.1f} s)")
    for shot in scene["shots"]:
        kinds = ", ".join(f"{l['type']}{':' + l['kind'] if l.get('kind') else ''}" for l in shot["layers"])
        print(f"  {shot['shot_id']:<12}{shot['from'] / FPS:>5.1f}s +{shot['frames'] / FPS:>4.1f}s  {shot['layout']:<9}in {shot['transition_in']['type']:<6}"
              f"subs {shot['subtitles']['style']} y={shot['subtitles']['rect']['y']:.0f} ({len(shot['subtitles']['cards'])} cards)  [{kinds}]")
    a = scene["audio"]
    print(f"  audio: voice {a['voice']['src'] or '-'}, music {[m['asset'] for m in a['music']] or '-'}, sfx {len(a['sfx'])}, dynamics {len(a['dynamics'])}")
    for note in scene["notes"]:
        print("  NOTE:", note)


def main() -> None:
    from agents._common import read_brief
    from core.registry_loader import RegistryLoader
    from core.run_context import RunContext
    from engines.visual import asset_geometry as ag

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run", required=True)
    parser.add_argument("--beat", default="")
    parser.add_argument("--no-stage", action="store_true", help="do not copy files into Remotion's public folder")
    args = parser.parse_args()
    run = RunContext.load(args.from_run)
    plan, timing, voice_plan = run.read_stage_output("visual_plan"), run.read_stage_output("voice_timing"), run.read_stage_output("voice_plan")
    if not (plan and timing and voice_plan):
        raise SceneError("The run needs a visual_plan, a voice_plan and a voice_timing (run the Voice Director and the voice engine).")
    info = read_brief(run)
    registry = RegistryLoader()
    channel_id = info["channel_id"]
    theme = registry.get_theme(registry.get_channel(channel_id).get("theme_id")) or {}
    palette = {**DEFAULT_PALETTE, **(theme.get("palette") or {})}
    cast = [m["id"] for m in registry.character_menu(channel_id)]

    def avatar_loader(character_id: str, pose_id: Optional[str]):
        avatar = registry.get_avatar(registry.get_character(character_id)["avatar_id"])
        path = ag.pose_path(avatar, pose_id or avatar["default_pose"])
        return ag.measure(path), path

    run_dir = Path(run.run_directory)
    ctx = {"run_id": run_dir.name, "run_dir": run_dir, "palette": palette, "images_dir": run_dir / "images", "cast": cast, "avatar_loader": avatar_loader,
           "media_by_id": {m["media_id"]: m for s in plan["shots"] for m in (s.get("media") or [])},
           "beat_order": list(timing["beats"]),
           "theme_subset": {k: theme.get(k) for k in ("palette", "typography", "motion", "subtitles", "image_treatment", "overlays") if k in theme}}
    ctx["theme_subset"]["palette"] = palette
    scenes = compile_run(plan, timing, voice_plan, ctx, args.beat or None)
    out_dir = run_dir / "scenes"
    out_dir.mkdir(exist_ok=True)
    public = Path(getattr(config, "REMOTION_PUBLIC_DIR", r"C:\AI_Studio\Projects\Remotion\public"))
    index = {"run_id": ctx["run_id"], "fps": FPS, "beats": []}
    for beat_id, scene in scenes.items():
        preview(scene, out_dir / f"preview_{beat_id}.png")           # before staging: it needs the real file paths
        if not args.no_stage:
            stage_assets(scene, ctx["run_id"], public)
        (out_dir / f"scene_{beat_id}.json").write_text(json.dumps(scene, indent=1), encoding="utf-8")
        index["beats"].append({"beat_id": beat_id, "file": f"scene_{beat_id}.json", "frames": scene["frames"], "start_s": scene["start_s"]})
        _summary(scene)
    (out_dir / "scene_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"\nScenes and wireframe previews: {out_dir}")


if __name__ == "__main__":
    main()