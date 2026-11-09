"""
Kestrel Shorts - Image Prompter
===============================

Writes the prompts for the images the Visual Director asked for. It reads the
plan (what each image shows, how it is framed, why it is there), the video's
visual style and the visual identity of the channel and world, and writes:

  - ONE "look" for the whole video (light, palette, materials, camera feel)
    plus a base negative prompt. Code adds the look to every prompt, so all
    images of a video share one style and cannot drift.
  - per image: a subject description, an image-specific negative prompt, the
    LoRAs it wants (only from the allowed menu) and a concern when a photo
    generator cannot deliver the brief (a diagram, ray lines, readable text).

It never touches text, subtitles or avatars, and it does not decide what is
shown: the Visual Director does. It does not run ComfyUI either: the Image
Service (engines/image/image_service.py) does, using the prompts saved here
as the run's `image_prompts` stage. Finished work is saved after every chunk,
so a stopped run resumes.

    python -m agents.image_prompter --from-run <path to shared_context.json> [--redo]
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents._common import read_brief
from core import llm
from core.registry_loader import RegistryLoader
from creative.catalogs import visual_style_catalog
from engines.image.image_service import collect, lora_menu, slot_aspect

STAGE = "image_prompter"
CHUNK = 6                      # images written per model call
SUBJECT_WORDS = (10, 70)
LOOK_WORDS = (5, 30)
_WRITING = re.compile(r"\b(text|texts|caption|captions|label|labels|labeled|labelled|letters|lettering|"
                      r"words|typography|subtitle|subtitles|logo|logos|arrow|arrows|diagram|diagrams)\b", re.IGNORECASE)


_MEDIA_ID = re.compile(r"\bm\d+\b", re.IGNORECASE)
_NOT_A_PHOTO = re.compile(r"\b(cross[- ]sections?|diagrams?|schematics?|labell?ed|light rays?|rays|arrows?|charts?|"
                          r"graphs?|maps?|infographics?|blueprints?|x-rays?)\b", re.IGNORECASE)


def _dedupe(text: str) -> str:
    """Drop repeated comma-separated phrases (the model sometimes echoes the look)."""
    seen, out = set(), []
    for part in (p.strip() for p in re.split(r"[,.]", text)):
        key = part.lower().rstrip(".")
        if key and key not in seen:
            seen.add(key)
            out.append(part)
    return ", ".join(out)


def _bare(negative: str) -> str:
    """A negative prompt lists concepts: remove 'no', 'without', 'avoid' and stray 'or'."""
    text = re.sub(r"\b(?:no|without|avoid)\s+", "", negative, flags=re.IGNORECASE)
    text = re.sub(r"\bor\s+", "", text, flags=re.IGNORECASE)
    return _dedupe(text.replace(".", ","))


def _shape(aspect: float) -> str:
    return "portrait" if aspect < 0.85 else "landscape" if aspect > 1.2 else "square"


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split()).strip(" ,.")


def _identity(registry: RegistryLoader, channel_id: Optional[str], world_id: Optional[str]) -> Dict[str, str]:
    channel = registry.get_channel(channel_id) if channel_id else {}
    world = registry.get_world(world_id) if world_id else {}
    return {"channel": _clean(channel.get("visual_identity")), "world": _clean(world.get("visual_identity")),
            "family_id": str(channel.get("family_id") or "")}


def _beat_context(script: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for section in (script or {}).get("sections", []):
        beat = out.setdefault(section.get("beat_id") or "", {"about": "", "narration": ""})
        beat["about"] = beat["about"] or str(section.get("setting_context") or "")
        beat["narration"] = (beat["narration"] + " " + str(section.get("text") or "")).strip()
    return out


# ----------------------------------------------------------------------
# The look (one call per video)
# ----------------------------------------------------------------------

def _write_look(info: Dict[str, Any], style_text: str, identity: Dict[str, str]) -> Dict[str, str]:
    lines = "\n".join(f"- {name}: {text}" for name, text in (("channel", identity["channel"]),
                                                              ("world", identity["world"])) if text)
    prompt = f"""Task: define the visual LOOK shared by every image of one short video, so that they all feel like one video.

Topic: {info['topic']}
Visual style: {style_text}
Visual identity (the look and mood this channel and world always keep):
{lines or '- (none given)'}

Write:
- look: {LOOK_WORDS[0]} to {LOOK_WORDS[1]} words of comma-separated phrases describing how every image looks: lighting, colour palette, materials, camera and film feel. Follow the identity. Do not describe any subject, place or object, and mention no text. Images must be sharp and well exposed: never ask for soft focus, blur, haze or heavy grain.
- base_negative: up to 15 words naming what must never appear because it would break this look (another era, another art style, and so on).
Return JSON only."""
    schema = {"type": "object", "properties": {"look": {"type": "string"}, "base_negative": {"type": "string"}},
              "required": ["look", "base_negative"]}

    def validate(data: Dict[str, Any]) -> Dict[str, str]:
        look = _clean(data.get("look"))
        count = len(look.split())
        if not LOOK_WORDS[0] <= count <= LOOK_WORDS[1]:
            raise ValueError(f"The look has {count} words; it must have {LOOK_WORDS[0]} to {LOOK_WORDS[1]}.")
        return {"look": look, "base_negative": _clean(data.get("base_negative"))}

    return llm.generate_json(STAGE, prompt, schema=schema, validate=validate, retries=1)


# ----------------------------------------------------------------------
# The prompts (one call per chunk of a beat)
# ----------------------------------------------------------------------

def _prompt_chunk(info: Dict[str, Any], beat_id: str, beat: Dict[str, str], images: List[Dict[str, Any]],
                  look: str, menu: Dict[str, str], banned: List[str]) -> Dict[str, Dict[str, Any]]:
    media_ids = [i["media_id"] for i in images]
    mode_of = {i["media_id"]: i["mode"] for i in images}
    rows = []
    for i in images:
        row = (f"- {i['media_id']} [{i['mode']}{', cut-out' if i.get('cutout') else ''}, {i['shape']}, framing: {i['framing']}] {i['brief']} "
               f"(shown because: {i['intent']})")
        if i["mode"] == "variation":
            row += f"\n    made from the earlier image: {i['base_text']} (change level: {i['change']})"
        rows.append(row)
    menu_text = "\n".join(f"- {name}: {hint}" for name, hint in menu.items()) or "(none: always use an empty list)"

    prompt = f"""Task: write the image prompts for part of ONE beat of a short video.

Topic: {info['topic']}
Beat: {beat_id}, about: {beat.get('about') or 'n/a'}
Narration: {(beat.get('narration') or '')[:300]}
Code adds this look to every image, so do NOT repeat it: {look}

Images to write prompts for:
{chr(10).join(rows)}

LoRAs you may choose from (file names):
{menu_text}

Rules:
- subject: 20 to 45 words saying exactly what the image shows: the subject, its setting, the lighting and time of day, the framing. Concrete and pictureable. No style words (the look covers them).
- Never describe text, letters, labels, captions, numbers, arrows or diagrams. Show objects without any writing on them. Never name or describe the video's characters; people only as distant figures or silhouettes.
- negative: up to 12 bare terms (no 'no' or 'without') naming what must NOT appear in this image because it would contradict what you wrote: for a night scene add daylight, sun, blue sky; for an indoor scene add outdoor. Do not repeat the base negative or add generic quality words.
- A variation is made from an earlier image with image-to-image. Write a complete standalone description: restate what the earlier image contains (its main subjects, by name), say "same composition", then describe the change strongly and specifically. Never refer to image ids (like m10): the generator cannot see them. Its negative lists the opposite of the change.
- A cut-out image is made so its background can be removed: the subject stands alone, whole and in frame, on a plain evenly lit background in a colour that strongly contrasts with the subject (deep teal or charcoal behind pale subjects, pale grey behind dark ones; never close to the subject's own colours), with no scenery, props or other objects around it. Say so in the subject.
- Never write a person, a face, a head, a body, skin or clothing into a subject unless the brief names that person. If a brief is about a body part, an organ or anatomy, write it as an object: 'a detailed anatomical teaching model of X, on its own, on a plain background', never a living body.
- loras: 0 to 2 file names from the list, only when they clearly help this image; otherwise [].
- concern: "" normally. If a photo generator cannot deliver the brief (a diagram, light rays or arrows, labelled parts, a chart, a map, readable text), say so in one sentence and say what would work instead; still write the best photographic prompt you can.
Return JSON only."""

    lora_items = {"type": "string", "enum": list(menu)} if menu else {"type": "string"}
    schema = {"type": "object", "properties": {"images": {"type": "array", "items": {
        "type": "object", "properties": {
            "media_id": {"type": "string", "enum": media_ids}, "subject": {"type": "string"},
            "negative": {"type": "string"}, "loras": {"type": "array", "items": lora_items},
            "concern": {"type": "string"}},
        "required": ["media_id", "subject", "negative", "loras", "concern"]}}}, "required": ["images"]}

    def validate(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        for item in data.get("images") if isinstance(data.get("images"), list) else []:
            mid = item.get("media_id") if isinstance(item, dict) else None
            if mid not in media_ids or mid in found:
                continue
            subject = _clean(item.get("subject"))
            count = len(subject.split())
            if not SUBJECT_WORDS[0] <= count <= SUBJECT_WORDS[1]:
                raise ValueError(f"{mid}: the subject has {count} words; it must have "
                                 f"{SUBJECT_WORDS[0]} to {SUBJECT_WORDS[1]}.")
            bad = _WRITING.search(subject)
            if bad:
                raise ValueError(f"{mid}: do not mention {bad.group(0)!r}; show the object without any writing, "
                                 "labels or arrows.")
            lowered = subject.lower()
            if _MEDIA_ID.search(subject):
                raise ValueError(f"{mid}: do not mention image ids like {_MEDIA_ID.search(subject).group(0)!r}; "
                                 "describe the scene itself, in full.")
            if mode_of[mid] == "new" and "same composition" in lowered:
                raise ValueError(f"{mid}: this is a new image, not a variation; describe it on its own, "
                                 "without 'same composition'.")
            hit = next((t for t in banned if re.search(rf"\b{re.escape(t)}\b", lowered)), None)
            if hit:
                raise ValueError(f"{mid}: do not mention {hit!r}; the video's characters never appear in generated images.")
            if mode_of[mid] == "variation" and "same composition" not in lowered:
                raise ValueError(f"{mid}: a variation must say 'same composition' and restate what the earlier image contains.")
            found[mid] = {"subject": subject, "negative": _clean(item.get("negative")),
                          "loras": [n for n in (item.get("loras") or []) if n in menu][:2],
                          "concern": _clean(item.get("concern"))}
        missing = [m for m in media_ids if m not in found]
        if missing:
            raise ValueError(f"There is no prompt for: {', '.join(missing)}.")
        return found

    return llm.generate_json(STAGE, prompt, schema=schema, validate=validate, retries=1)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def _cast_hit(text: str, banned: List[str]) -> Optional[str]:
    low = text.lower()
    return next((t for t in banned if re.search(rf"\b{re.escape(t)}\b", low)), None)


def _blocked(prompts: Dict[str, Dict[str, Any]], media_id: str, reason: str) -> None:
    prompts[media_id] = {"blocked": reason, "concern": reason, "positive": "", "negative": "",
                         "loras": [], "subject": ""}
    print(f"[Image Prompter] BLOCKED {media_id}: {reason}")


def _write_chunk(info, beat_id, beat, images, look, menu, banned, prompts) -> Dict[str, Dict[str, Any]]:
    """Prompts for a chunk. If the model cannot satisfy the rules, each image is tried alone;
    an image that still fails is blocked with the reason (never a made-up prompt)."""
    try:
        return _prompt_chunk(info, beat_id, beat, images, look, menu, banned)
    except llm.LLMError as exc:
        if exc.kind != "invalid_json":
            raise
        print(f"[Image Prompter] A chunk failed the rules ({exc}); trying each image alone.")
    done: Dict[str, Dict[str, Any]] = {}
    for image in images:
        try:
            done.update(_prompt_chunk(info, beat_id, beat, [image], look, menu, banned))
        except llm.LLMError as exc:
            if exc.kind != "invalid_json":
                raise
            _blocked(prompts, image["media_id"], f"The model could not write a valid prompt for this brief: {exc}")
    return done


def run_image_prompter(plan: Dict[str, Any], *, context: Any = None, script: Optional[Dict[str, Any]] = None,
                       registry: Optional[RegistryLoader] = None, redo: bool = False) -> Dict[str, Dict[str, Any]]:
    """Prompts for every image the plan must generate: {media_id: {positive, negative, loras, concern, subject}}."""
    info = read_brief(context)
    ctx = info["ctx"]
    registry = registry or RegistryLoader()
    channel_id = info["channel_id"]
    identity = _identity(registry, channel_id, ctx.world_id if ctx else None)
    interpretation = ctx.interpretation if ctx else {}
    menu = lora_menu(str(interpretation.get("content_rating") or "general"), identity["family_id"],
                     str(interpretation.get("audience_override") or ""))

    style_id = info["visual_style"]
    style = visual_style_catalog.get_visual_style_catalog().get(style_id, {})
    style_text = f"{style_id}: {style.get('description', '')}" if style_id else "not specified"

    banned = ["avatar"]                      # the cast never appears in generated images
    for member in (registry.character_menu(channel_id) if channel_id else []):
        banned.append(member["id"].lower())
        if member.get("name"):
            banned.append(str(member["name"]).lower())        # e.g. 'old man': people only as distant figures
        banned += [str(p["pose_id"]).lower() for p in registry.pose_menu(member["id"]) if p.get("pose_id")]

    meta: Dict[str, Dict[str, str]] = {}
    brief_of: Dict[str, str] = {}
    for shot in plan.get("shots", []):
        for media in shot.get("media") or []:
            meta[media["media_id"]] = {"beat_id": shot.get("beat_id", ""), "intent": shot.get("intent", ""),
                                       "framing": media.get("framing", "medium")}
            brief_of[media["media_id"]] = media.get("brief", "")
    items = [i for i in collect(plan) if i["source"] != "reuse"]
    if not items:
        print("[Image Prompter] The plan has no images to generate.")
        return {}

    prompts: Dict[str, Dict[str, Any]] = {}
    if ctx is not None and not redo:
        prompts = {k: v for k, v in (ctx.read_stage_output("image_prompts") or {}).items()
                   if k in brief_of and not v.get("blocked") and v.get("brief") == brief_of[k]}   # blocked ones are tried again
    todo = [i for i in items if i["media_id"] not in prompts]
    if not todo:
        print("[Image Prompter] All prompts already exist (use --redo to write them again).")
        return prompts

    look_data = ctx.read_stage_output("image_look") if (ctx is not None and not redo) else None
    if not look_data:
        look_data = _write_look(info, style_text, identity)
        if ctx is not None:
            ctx.write_stage_output("image_look", look_data)
    look, base_negative = look_data["look"], look_data.get("base_negative", "")
    print(f"[Image Prompter] look: {look}")

    beat_info = _beat_context(script)
    beats: Dict[str, List[Dict[str, Any]]] = {}
    for item in todo:
        beats.setdefault(meta[item["media_id"]]["beat_id"], []).append(item)

    for beat_id, beat_items in beats.items():
        for start in range(0, len(beat_items), CHUNK):
            images = []
            for item in beat_items[start:start + CHUNK]:
                cast = _cast_hit(item["brief"], banned)
                if cast:
                    _blocked(prompts, item["media_id"],
                             f"The brief shows the cast ({cast!r}). Generated images never show the cast: "
                             "the Visual Director should use the avatar or describe a place or object instead.")
                    continue
                mode = "variation" if item["source"] == "variation" and item["base"] else "new"
                base = item["base"]
                base_text = ((prompts[base]["subject"] if base in prompts else brief_of.get(base, ""))
                             if mode == "variation" else "")
                images.append({"media_id": item["media_id"], "mode": mode, "brief": item["brief"],
                               "shape": _shape(slot_aspect(item["layout"], item["slot"], item["count"])), "cutout": item["cutout"],
                               "framing": meta[item["media_id"]]["framing"],
                               "intent": meta[item["media_id"]]["intent"],
                               "change": item["change"], "base_text": base_text})
            if not images:
                continue
            written = _write_chunk(info, beat_id, beat_info.get(beat_id, {}), images, look, menu, banned, prompts)
            for image in images:
                w = written.get(image["media_id"])
                if w is None:
                    continue
                concern = w["concern"]
                cue = _NOT_A_PHOTO.search(image["brief"])
                if not concern and cue:
                    concern = (f"The brief asks for '{cue.group(0)}', which a photo generator draws poorly; "
                               "a Remotion graphic or diagram would fit better.")
                prompts[image["media_id"]] = {
                    "positive": _dedupe(f"{w['subject']}, {look}"),
                    "negative": _bare(", ".join(x for x in (base_negative, w["negative"]) if x)),
                    "loras": w["loras"], "concern": concern, "subject": w["subject"], "brief": image["brief"]}

            if ctx is not None:
                ctx.write_stage_output("image_prompts", prompts)     # saved as we go, so a stop loses little
        print(f"[Image Prompter] {beat_id}: {len(beat_items)} prompt(s)")
    return prompts


def _main() -> None:
    import argparse

    from core.run_context import RunContext

    parser = argparse.ArgumentParser(description="Write the image prompts for a run's visual plan.")
    parser.add_argument("--from-run", required=True, help="path to a run's shared_context.json")
    parser.add_argument("--redo", action="store_true", help="write all prompts again")
    args = parser.parse_args()

    ctx = RunContext.load(args.from_run)
    plan = ctx.read_stage_output("visual_plan")
    if not plan:
        raise SystemExit("This run has no visual_plan stage output.")
    prompts = run_image_prompter(plan, context=ctx, script=ctx.read_stage_output("story_director"), redo=args.redo)
    ctx.write_stage_output("image_prompts", prompts)


    blocked = [m for m, p in prompts.items() if p.get("blocked")]
    for media_id, p in prompts.items():
        if p.get("blocked"):
            print(f"\n{media_id}\n  BLOCKED: {p['blocked']}")
            continue
        print(f"\n{media_id}\n  positive: {p['positive']}\n  negative: {p['negative']}"
              + (f"\n  loras: {', '.join(p['loras'])}" if p["loras"] else "")
              + (f"\n  CONCERN: {p['concern']}" if p["concern"] else ""))
    print(f"\n{len(prompts) - len(blocked)} prompt(s) saved, {len(blocked)} blocked, in the run's image_prompts stage.")
    if blocked:
        print("Blocked images (the Visual Director must change these):", ", ".join(blocked))


if __name__ == "__main__":
    _main()