"""
Kestrel Shorts - Realism lab
============================

The same prompts drawn with several recipes (sampler, steps, guidance, LoRAs, a few photographic words) on the
same seed, laid out as ONE contact sheet, so the recipe for "less plastic" is chosen by eye instead of by guess.
Nothing here changes the pipeline: when a recipe wins, its numbers go into the image service's tables.

    python -m engines.image.realism_lab                       preview: what would be drawn
    python -m engines.image.realism_lab --go                  draw everything (about 40 s per image)
    python -m engines.image.realism_lab --go --prompts eye,object --only R2,R4

Results: runtime/realism_lab/<recipe>_<prompt>.png and sheet.png (rows = recipes, columns = prompts).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core import config
from engines.image import comfyui_api
from engines.image import image_service as svc

SEED = 1234
SIZE = (832, 1216)                       # portrait, like most of the shorts' pictures
LOOK = "natural light, earthy tones, weathered textures, soft shadows, muted palette"
STUDIO = {"retina3d", "brain3d", "eye3d"}          # illustrations get a studio look, not the photo look
STUDIO_LOOK = "soft studio lighting, clean, high detail, educational render"
STUDIO_NEGATIVE = ", photograph, skin, person, face, text, labels, gore, blood"
BASE_NEGATIVE = "text, watermark, modern objects, cartoon style, digital artifacts"
PHOTO_WORDS = ", shot on 35mm film, natural skin and surface texture, subtle film grain, slight imperfections, shallow depth of field"
PHOTO_NEGATIVE = ", plastic, airbrushed, smooth skin, 3d render, cgi, oversaturated, glossy, waxy"

PROMPTS: Dict[str, str] = {
    "eye": "close-up photograph of a human eye, visible iris detail and eyelashes",
    "object": "an antique brass camera lens on a worn wooden workbench, window light from the left",
    "person": "portrait of an elderly craftsman examining a lens in his workshop, medium shot",
    "retina3d": "3D medical illustration of the human retina with photoreceptor cells, plain dark background",
    "brain3d": "3D medical illustration of the human brain seen from the side, plain dark background",
    "eye3d": "3D medical illustration of a human eyeball in cross-section showing cornea, lens and retina, plain dark background",
}

# id: (label, sampler overrides, loras, extra positive words, extra negative words)
RECIPES: Dict[str, Tuple[str, Dict[str, Any], List[Tuple[str, float]], str, str]] = {
    "R0": ("R0 baseline (today)", {}, [("add-detail-xl.safetensors", 0.8)], "", ""),
    "R1": ("R1 guidance 4.5", {"cfg": 4.5}, [("add-detail-xl.safetensors", 0.8)], "", ""),
    "R2": ("R2 dpmpp_2m_sde karras 30, cfg 5", {"steps": 30, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde", "scheduler": "karras"},
           [("add-detail-xl.safetensors", 0.8)], "", ""),
    "R3": ("R3 R2 + realism LoRA", {"steps": 30, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde", "scheduler": "karras"},
           [("Touch_of_Realism_SDXL_V2.safetensors", 0.75), ("add-detail-xl.safetensors", 0.5)], "", ""),
    "R4": ("R4 R3 + photo words", {"steps": 30, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde", "scheduler": "karras"},
           [("Touch_of_Realism_SDXL_V2.safetensors", 0.75), ("add-detail-xl.safetensors", 0.5)], PHOTO_WORDS, PHOTO_NEGATIVE),
    "R5": ("R5 photo words, no LoRA", {"steps": 30, "cfg": 5.0, "sampler_name": "dpmpp_2m_sde", "scheduler": "karras"},
           [], PHOTO_WORDS, PHOTO_NEGATIVE),
}


def _job(recipe: str, prompt_id: str, out: Path) -> "comfyui_api.ImageJob":
    label, sampler, loras, plus, minus = RECIPES[recipe]
    allowed = {name for name in svc.LORAS}
    use = [(n, s) for n, s in loras if n in allowed]
    return comfyui_api.ImageJob(
        positive=f"{PROMPTS[prompt_id]}, {STUDIO_LOOK if prompt_id in STUDIO else LOOK}{plus}",
        negative=svc.with_floor(BASE_NEGATIVE + minus + (STUDIO_NEGATIVE if prompt_id in STUDIO else "")),
        output_path=out, width=SIZE[0], height=SIZE[1], seed=SEED, checkpoint=svc.CHECKPOINT, loras=use,
        **{**svc.SAMPLER, **sampler})


def _sheet(cells: Dict[Tuple[str, str], Path], recipes: List[str], prompts: List[str], out: Path) -> None:
    from PIL import Image, ImageDraw
    height, label_h, pad = 440, 34, 8
    width = round(height * SIZE[0] / SIZE[1])
    left = 330
    canvas = Image.new("RGB", (left + len(prompts) * (width + pad), label_h + len(recipes) * (height + pad)), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for c, prompt_id in enumerate(prompts):
        draw.text((left + c * (width + pad) + 6, 10), prompt_id, fill=(230, 230, 230))
    for r, recipe in enumerate(recipes):
        y = label_h + r * (height + pad)
        draw.text((8, y + 8), RECIPES[recipe][0], fill=(230, 200, 120))
        for c, prompt_id in enumerate(prompts):
            path = cells.get((recipe, prompt_id))
            if path and path.exists():
                canvas.paste(Image.open(path).convert("RGB").resize((width, height)), (left + c * (width + pad), y))
    canvas.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true", help="really draw (without it, only a preview is shown)")
    parser.add_argument("--prompts", default="eye,object", help=f"comma-separated, from: {', '.join(PROMPTS)}")
    parser.add_argument("--only", default="", help="comma-separated recipe ids, for example R2,R4")
    args = parser.parse_args()
    prompts = [p for p in args.prompts.split(",") if p in PROMPTS]
    recipes = [r for r in (args.only.split(",") if args.only else list(RECIPES)) if r in RECIPES]
    folder = Path(config.SHORTS_ROOT) / "runtime" / "realism_lab"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{len(recipes) * len(prompts)} image(s): recipes {', '.join(recipes)} x prompts {', '.join(prompts)} "
          f"(about {len(recipes) * len(prompts) * 40 / 60:.0f} min)")
    for r in recipes:
        print(f"  {RECIPES[r][0]}")
    if not args.go:
        print("Preview only. Add --go to draw.")
        return
    try:
        from core import llm
        llm.unload_model()
    except Exception:
        pass
    cells: Dict[Tuple[str, str], Path] = {}
    try:
        for r in recipes:
            for p in prompts:
                out = folder / f"{r}_{p}.png"
                started = time.monotonic()
                try:
                    comfyui_api.generate(_job(r, p, out))
                    cells[(r, p)] = out
                    print(f"[Lab] {r} {p} ready in {time.monotonic() - started:.0f} s")
                except (comfyui_api.ComfyError, ValueError) as exc:
                    print(f"[Lab] {r} {p} FAILED: {exc}")
    finally:
        comfyui_api.free()
    _sheet(cells, recipes, prompts, folder / "sheet.png")
    print(f"[Lab] Contact sheet: {folder / 'sheet.png'}")


if __name__ == "__main__":
    main()