"""
Kestrel Shorts - Image Service
==============================

The one door to image generation, like core/search.py is for search. Agents
never talk to ComfyUI: they hand over prompts and get files back.

It knows what is installed on this machine (checkpoint, LoRAs) and how to run
it (sampler, image sizes, how far an image-to-image variation may drift from
its source). It decides nothing creative: the Visual Director says WHAT and
WHERE, the Image Prompter writes the prompts, and this file only runs them.

What it does:
  - collects every image of a visual plan, follows reuse chains to the image
    that really exists, and skips pure reuse;
  - builds one job per image: a size from the slot's shape (snapped to an
    SDXL-friendly size), a seed derived from the run and image ids, LoRAs
    checked against an allowed list, and a negative prompt that always keeps a
    minimal floor (text, watermark);
  - runs the jobs through comfyui_api, skipping files that already exist (a
    stopped run resumes), unloads Ollama first and frees the GPU afterwards;
  - records how long images take, so the cost preview uses real numbers.

Adult LoRAs: a LoRA row with rating "adult" is offered only when the user's
own prompt asked for an 18+ video (content_rating == "adult"), and never on a
children's channel or with a kids/teens audience override. Any LoRA the Image
Prompter returns that is not on the allowed list is dropped.

    python -m engines.image_service --from-run <path to shared_context.json> --use-briefs
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import config
from engines.image import comfyui_api

# ----------------------------------------------------------------------
# What is installed on this machine (edit here when you add a model)
# ----------------------------------------------------------------------

CHECKPOINT = "CyberRealisticXLPlay_V9.0_FP16.safetensors"      # photorealistic SDXL

# file name -> hint shown to the Image Prompter, default strength, rating.
# To add a LoRA, add a row. A row with rating "adult" is offered only for an
# explicit 18+ request (see adult_allowed).
LORAS: Dict[str, Dict[str, Any]] = {
    "Touch_of_Realism_SDXL_V2.safetensors": {
        "hint": "natural, photographic realism; skip it for illustrated or graphic styles",
        "strength": 0.75, "rating": "general"},
    "add-detail-xl.safetensors": {
        "hint": "fine detail and texture; good for objects, mechanisms, close-ups and diagrams",
        "strength": 0.8, "rating": "general"},
}

SAMPLER = {"steps": 25, "cfg": 6.5, "sampler_name": "euler_ancestral", "scheduler": "normal"}
DENOISE = {"small": 0.35, "medium": 0.50, "large": 0.65}       # provisional; tuned from tests
BUCKETS = [(1024, 1024), (896, 1152), (832, 1216), (768, 1344), (640, 1536),
           (1152, 896), (1216, 832), (1344, 768), (1536, 640)]  # SDXL-friendly sizes, about 1 megapixel
NEGATIVE_FLOOR = ("text", "watermark")      # the only prompt words this file insists on
CUTOUT_ASPECT = 0.8                         # a cut-out subject is generated nearly square (4:5)
USABLE_HEIGHT_SHARE = 0.8

class ImageServiceError(RuntimeError):
    """The image batch could not be prepared or finished."""


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------

def adult_allowed(content_rating: str, family_id: str = "", audience_override: str = "") -> bool:
    """True only for an explicit adult request on a channel that is not for children."""
    if content_rating != "adult" or audience_override in ("kids", "teens"):
        return False
    try:
        from core.models.content_family import CONTENT_FAMILIES
    except Exception:
        return False                                    # cannot check the audience: stay closed
    family = CONTENT_FAMILIES.get(family_id)
    return not (family and family.audience_profile_id == "children")


def lora_menu(content_rating: str = "general", family_id: str = "", audience_override: str = "") -> Dict[str, str]:
    """LoRA file -> hint: what the Image Prompter may choose from."""
    adult_ok = adult_allowed(content_rating, family_id, audience_override)
    return {name: row["hint"] for name, row in LORAS.items()
            if row["rating"] == "general" or (row["rating"] == "adult" and adult_ok)}


def slot_aspect(layout: str, index: int = 0, count: int = 1) -> float:
    """Width / height of an image slot, from the layout spec (Remotion crops the picture to fit)."""
    from engines.visual import layout_spec
    found = layout_spec.slots(layout, max(1, count))
    rect = found[min(index, len(found) - 1)]["rect"] if found else layout_spec.CANVAS
    return rect.width / rect.height


def _slot_aspect_old(layout: str, index: int = 0) -> float:
    """Width / height of an image slot in a layout. Approximate; Remotion crops to fit."""
    w, h = float(config.VIDEO_WIDTH), float(config.VIDEO_HEIGHT)
    if layout in ("single", "sequence", "text_only"):
        return w / h
    if layout == "inset":
        return w / h if index == 0 else 1.0
    if layout == "scatter":
        return 0.75
    cols, rows = {"stack_2": (1, 2), "stack_3": (1, 3), "grid_4": (2, 2), "grid_6": (2, 3)}.get(layout, (1, 1))
    return (w / cols) / (h * USABLE_HEIGHT_SHARE / rows)


def pick_bucket(aspect: float) -> Tuple[int, int]:
    """The SDXL-friendly size whose shape is closest to `aspect`."""
    return min(BUCKETS, key=lambda size: abs(math.log(aspect / (size[0] / size[1]))))


import re as _re

CFG_OBJECT, CFG_PEOPLE = 4.5, 6.5
_PEOPLE = _re.compile(r"\b(?:man|men|woman|women|person|people|portrait|face|faces|eye|eyes|crowd|soldiers?|workers?|"
                      r"child|children|king|queen|emperor|merchant|craftsman|monk|priest)\b", _re.IGNORECASE)


def _cfg_for(positive: str) -> float:
    """Guidance: 4.5 for objects, places and illustrations (cleaner shapes, less gloss); 6.5 when the picture
    shows a person, where skin needs the stronger pull. Chosen from the realism lab."""
    return CFG_PEOPLE if _PEOPLE.search(positive) else CFG_OBJECT


SAFETY_NEGATIVE = ("(nsfw:1.4)", "(nude:1.4)", "(naked:1.4)", "(topless:1.4)", "nipples", "breasts", "genitals",
                   "cleavage", "lingerie", "underwear", "erotic")


def with_floor(negative: str, safe: bool = True) -> str:
    """Keep the Image Prompter's negative prompt, adding any floor word it forgot and, unless the user asked for an
    18+ video, the safety words: this checkpoint can draw a body for almost any anatomy prompt."""
    text = (negative or "").strip().strip(",").strip()
    wanted = list(NEGATIVE_FLOOR) + (list(SAFETY_NEGATIVE) if safe else [])
    missing = [word for word in wanted if word.lower() not in text.lower()]
    return ", ".join(part for part in [text, *missing] if part)


def _with_floor_old(negative: str) -> str:
    """Keep the Image Prompter's negative prompt, adding any floor word it forgot."""
    text = (negative or "").strip().strip(",").strip()
    missing = [word for word in NEGATIVE_FLOOR if word not in text.lower()]
    return ", ".join(part for part in [text, *missing] if part)


def _slug(text: str, words: int = 3) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:words]) or "image"


def _seed(run_id: str, media_id: str) -> int:
    return int(hashlib.sha256(f"{run_id}:{media_id}".encode("utf-8")).hexdigest()[:8], 16)


# ----------------------------------------------------------------------
# Plan -> jobs
# ----------------------------------------------------------------------

def collect(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every image of the plan, in order. 'owner' is the image whose file really exists
    (reuse chains are followed); 'base' is a variation's source image."""
    items: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for shot in plan.get("shots", []):
        for slot, media in enumerate(shot.get("media") or []):
            item = {"media_id": media["media_id"], "source": media.get("source", "generate"),
                    "ref": media.get("ref") or "", "change": media.get("change") or "none",
                    "brief": media.get("brief", ""), "layout": shot.get("layout") or "single",
                    "slot": slot, "shot_id": shot.get("shot_id", ""),
                    "count": len(shot.get("media") or []), 
                    "cutout": bool(media.get("cutout"))}

            items.append(item)
            by_id[item["media_id"]] = item

    def owner(media_id: str) -> str:
        seen: List[str] = []
        item = by_id.get(media_id)
        while item is not None and item["source"] == "reuse":
            if item["media_id"] in seen:
                raise ImageServiceError(f"Image {media_id} reuses itself in a loop.")
            seen.append(item["media_id"])
            item = by_id.get(item["ref"])
        if item is None:
            raise ImageServiceError(f"Image {media_id} points to an image that does not exist.")
        return item["media_id"]

    for item in items:
        item["owner"] = owner(item["media_id"])
        item["base"] = owner(item["ref"]) if item["source"] == "variation" and item["ref"] in by_id else ""
        item["cutout_owner"] = by_id[item["owner"]]["cutout"]
    return items


def _cut_path(path: Path) -> Path:
    """The transparent version of a cut-out image."""
    return path.with_name(path.stem + "_cutout.png")


def _paths(items: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
    return {i["media_id"]: Path(images_dir) / f"{i['media_id']}_{_slug(i['brief'])}.png"
            for i in items if i["source"] != "reuse"}


def _signature(job: Dict[str, Any]) -> str:
    """A fingerprint of everything that decides how an image looks."""
    data = {k: job[k] for k in ("positive", "negative", "width", "height", "seed", "denoise", "cutout","cfg")}
    data["loras"] = [list(x) for x in job["loras"]]
    data["source"] = job["source_path"].name if job["source_path"] else ""
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def is_done(job: Dict[str, Any]) -> bool:
    """The image exists AND was made from exactly this job (a changed prompt means: make it again)."""
    try:
        sidecar = json.loads(job["path"].with_suffix(".json").read_text(encoding="utf-8"))
        return (job["path"].exists() and sidecar.get("signature") == job["signature"]
                and (not job["cutout"] or _cut_path(job["path"]).exists()))
    except (OSError, ValueError):
        return False


def build_jobs(plan: Dict[str, Any], prompts: Dict[str, Dict[str, Any]], run_id: str, images_dir: Path,
               content_rating: str = "general", family_id: str = "", audience_override: str = "",
               ) -> List[Dict[str, Any]]:
    """One job per image that must be generated. `prompts` maps media_id to
    {"positive": str, "negative": str, "loras": [file names]} (from the Image Prompter)."""
    items = collect(plan)
    allowed = lora_menu(content_rating, family_id, audience_override)
    safe = not adult_allowed(content_rating, family_id, audience_override)
    paths = _paths(items, images_dir)
    jobs: List[Dict[str, Any]] = []
    for item in items:
        if item["source"] == "reuse":
            continue
        prompt = prompts.get(item["media_id"]) or {}
        if prompt.get("blocked"):
            print(f"[Image Service] Skipped {item['media_id']} (blocked): {prompt['blocked']}")
            continue
        positive = str(prompt.get("positive", "")).strip()
        if not positive:
            raise ImageServiceError(f"There is no prompt for image {item['media_id']}.")
        loras: List[Tuple[str, float]] = []
        for name in prompt.get("loras") or []:
            if name in allowed and name not in [n for n, _ in loras]:
                loras.append((name, float(LORAS[name]["strength"])))
            else:
                print(f"[Image Service] Dropped LoRA {name!r} for {item['media_id']}: not on the allowed list.")
        aspect = CUTOUT_ASPECT if item["cutout"] else slot_aspect(item["layout"], item["slot"], item["count"])
        width, height = pick_bucket(aspect)
        variation = item["source"] == "variation" and bool(item["base"])
        job = ({
            "media_id": item["media_id"], "mode": "variation" if variation else "new",
            "path": paths[item["media_id"]], "cutout": item["cutout"], "cfg": _cfg_for(positive), "positive": positive,
            "negative": with_floor(str(prompt.get("negative", "")), safe), "loras": loras,
            "width": width, "height": height, "seed": _seed(run_id, item["media_id"]),
            "source_path": paths.get(item["base"]) if variation else None,
            "denoise": DENOISE.get(item["change"], DENOISE["medium"]) if variation else 1.0,
        })
        job["signature"] = _signature(job)
        jobs.append(job)
    return jobs


def image_files(plan: Dict[str, Any], images_dir: Path) -> Dict[str, Path]:
    """media_id -> file, for every image whose file exists (reuse points at the original)."""
    items = collect(plan)
    paths = _paths(items, images_dir)
    return {i["media_id"]: (_cut_path(paths[i["owner"]]) if i["cutout_owner"] else paths[i["owner"]])
            for i in items if paths[i["owner"]].exists()}


# ----------------------------------------------------------------------
# Timing and preview
# ----------------------------------------------------------------------

def _timing_file() -> Path:
    return Path(config.RUNS_DIR).parent / "image_timing.json"


def _timings() -> Dict[str, List[float]]:
    try:
        return json.loads(_timing_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _record(mode: str, seconds: float) -> None:
    data = _timings()
    data[mode] = (data.get(mode, []) + [round(seconds, 1)])[-20:]
    try:
        _timing_file().write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def preview(jobs: List[Dict[str, Any]]) -> str:
    """What running these jobs would involve, with an estimate from real timings."""
    pending = [j for j in jobs if not is_done(j)]
    new = sum(j["mode"] == "new" for j in pending)
    variations = len(pending) - new
    line = (f"{len(pending)} image(s) to generate ({new} new, {variations} variation), "
            f"{len(jobs) - len(pending)} already on disk.")
    times = _timings()

    def average(mode: str) -> Optional[float]:
        values = times.get(mode) or []
        return sum(values) / len(values) if values else None

    total, known = 0.0, True
    for mode, count in (("new", new), ("variation", variations)):
        if count:
            seconds = average(mode) or average("new")
            if seconds is None:
                known = False
            else:
                total += seconds * count
    if pending and known:
        line += f" Estimated time: about {total / 60:.1f} min (from your earlier runs)."
    elif pending:
        line += " Estimated time: unknown until the first images have been timed."
    return line


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

def run(jobs: List[Dict[str, Any]], *, only: Optional[List[str]] = None, limit: Optional[int] = None,
        unload_llm: bool = True) -> None:
    """Generate the images that do not exist yet. Raises ImageServiceError at the end if any failed."""
    todo = [j for j in jobs if not is_done(j) and (not only or j["media_id"] in only)]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("[Image Service] Nothing to generate.")
        return
    if unload_llm:                                       # the GPU is shared with Ollama
        from core import llm
        print("[Image Service] Ollama unloaded:", llm.unload_model())

    failures: List[str] = []
    try:
        for number, job in enumerate(todo, 1):
            print(f"[Image Service] {number}/{len(todo)}: {job['media_id']} ({job['mode']})")
            started = time.monotonic()
            try:
                comfyui_api.generate(comfyui_api.ImageJob(
                    positive=job["positive"], negative=job["negative"], output_path=job["path"],
                    width=job["width"], height=job["height"], seed=job["seed"], checkpoint=CHECKPOINT,
                    loras=job["loras"], source_image=job["source_path"], denoise=job["denoise"], **{**SAMPLER, "cfg": job["cfg"]}))
                if job["cutout"]:
                    comfyui_api.cutout(job["path"], _cut_path(job["path"]))
                _record(job["mode"], time.monotonic() - started)
                job["path"].with_suffix(".json").write_text(json.dumps({"signature": job["signature"]}), encoding="utf-8")
            except (comfyui_api.ComfyError, ValueError) as exc:
                failures.append(f"{job['media_id']}: {exc}")
                print(f"[Image Service] FAILED {job['media_id']}: {exc}")
    finally:
        comfyui_api.free()
    if failures:
        raise ImageServiceError("Some images failed (finished ones are kept; run again to retry only these):\n  "
                                + "\n  ".join(failures))


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def _main() -> None:
    import argparse

    from core.registry_loader import RegistryLoader
    from core.run_context import RunContext

    parser = argparse.ArgumentParser(description="Preview or run the image batch of a run.")
    parser.add_argument("--from-run", required=True, help="path to a run's shared_context.json")
    parser.add_argument("--use-briefs", action="store_true", help="TEST ONLY: use each image brief as its prompt")
    parser.add_argument("--only", default="", help="comma-separated image ids, for example m3,m7")
    parser.add_argument("--limit", type=int, default=0, help="generate at most this many images")
    parser.add_argument("--go", action="store_true", help="really generate (without it, only a preview is shown)")
    args = parser.parse_args()

    ctx = RunContext.load(args.from_run)
    plan = ctx.read_stage_output("visual_plan")
    if not plan:
        raise SystemExit("This run has no visual_plan stage output.")
    prompts = ctx.read_stage_output("image_prompts") or {}
    if args.use_briefs:
        prompts = {i["media_id"]: {"positive": i["brief"], "negative": "", "loras": []}
                   for i in collect(plan) if i["source"] != "reuse"}
    family = str(RegistryLoader().get_channel(ctx.channel_id).get("family_id", "")) if ctx.channel_id else ""

    jobs = build_jobs(plan, prompts, ctx.run_id, ctx.images_directory,
                      str(ctx.interpretation.get("content_rating") or "general"),
                      family, str(ctx.interpretation.get("audience_override") or ""))
    only = [x.strip() for x in args.only.split(",") if x.strip()]
    print(preview([j for j in jobs if not only or j["media_id"] in only]))
    if not args.go:
        print("Preview only. Add --go to generate.")
        return
    try:
        run(jobs, only=only or None, limit=args.limit or None)
    except ImageServiceError as exc:
        raise SystemExit(str(exc))
    print("Images are in:", ctx.images_directory)


if __name__ == "__main__":
    _main()