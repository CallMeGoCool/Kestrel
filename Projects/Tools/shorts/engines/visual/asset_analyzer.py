"""
Kestrel Shorts - Asset analyzer
===============================

What the Frame Supervisor must know about a picture before it places text on it, crops it or moves it. Plain
image maths (PIL + numpy), no model, a few milliseconds per picture, cached by file name, size and date.

  size, aspect          the picture's pixel shape
  alpha_box, visible    for a cut-out: the visible rectangle (0-1 of the picture) and its share of the area
  tone, luma, contrast  how light the picture is overall (0-1), and how much it varies
  colors, accent        the three dominant colours and the most vivid one (hex), for matching text and accents
  focus                 where the eye goes (x, y in 0-1): the centre of the busiest part of the picture, pulled a
                        little toward the middle, so a crop or a zoom lands on the subject and not on the sky
  grid                  3 x 3 zones, each: luma, detail, "ink" (which text colour reads there: dark | light) and
                        "quiet" (calm enough to carry text)
  sharpness             detail of the whole picture; very low means a soft or blurry picture

    python -m engines.visual.asset_analyzer --from-run <shared_context.json>
    python -m engines.visual.asset_analyzer some_picture.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from core import config

WORK_SIDE = 256                       # pictures are analysed at this longest side
QUIET_DETAIL = 0.35                   # a zone with less detail than this share of the busiest zone is "quiet"
SOFT_SHARPNESS = 0.012                # below this the picture is soft
CACHE_NAME = "asset_analysis.json"


def _cache_path() -> Path:
    return Path(config.SHORTS_ROOT) / "runtime" / CACHE_NAME


def _hex(rgb: np.ndarray) -> str:
    return "#{:02X}{:02X}{:02X}".format(*[int(v) for v in rgb])


def _detail(luma: np.ndarray) -> np.ndarray:
    """Local detail: gradient strength per pixel (0 where the picture is flat)."""
    gy, gx = np.gradient(luma)
    return np.hypot(gx, gy)


def _blocks(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = values.shape
    out = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            out[r, c] = values[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols].mean()
    return out


def analyze_image(path: Path | str) -> Dict[str, Any]:
    image = Image.open(path)
    width, height = image.size
    rgba = image.convert("RGBA")
    rgba.thumbnail((WORK_SIDE, WORK_SIDE))
    data = np.asarray(rgba, dtype=np.float32) / 255.0
    alpha, rgb = data[..., 3], data[..., :3]
    seen = alpha > 0.06
    out: Dict[str, Any] = {"size": [width, height], "aspect": round(width / height, 4), "has_alpha": bool(image.mode in ("RGBA", "LA")
                                                                                                        and alpha.min() < 0.99)}
    if out["has_alpha"] and seen.any():
        ys, xs = np.where(seen)
        h, w = alpha.shape
        out["alpha_box"] = [round(xs.min() / w, 3), round(ys.min() / h, 3), round((xs.max() + 1) / w, 3), round((ys.max() + 1) / h, 3)]
        out["visible"] = round(float(seen.mean()), 3)
    else:
        out["alpha_box"], out["visible"] = [0.0, 0.0, 1.0, 1.0], 1.0

    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    weight = seen.astype(np.float32)
    mean_luma = float((luma * weight).sum() / max(weight.sum(), 1.0))
    out.update({"luma": round(mean_luma, 3), "contrast": round(float(luma[seen].std()) if seen.any() else 0.0, 3),
                "tone": "dark" if mean_luma < 0.33 else "light" if mean_luma > 0.62 else "mid"})

    detail = _detail(luma) * weight
    out["sharpness"] = round(float(detail[seen].mean()) if seen.any() else 0.0, 4)
    out["soft"] = out["sharpness"] < SOFT_SHARPNESS

    # dominant colours: 4 bits per channel, the three most common bins, and the most vivid of the top eight
    pixels = (rgb[seen] * 255).astype(np.int32)
    if len(pixels):
        keys = (pixels[:, 0] >> 4) * 256 + (pixels[:, 1] >> 4) * 16 + (pixels[:, 2] >> 4)
        bins, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)[:8]
        top = [(np.array([(b >> 8) * 16 + 8, ((b >> 4) & 15) * 16 + 8, (b & 15) * 16 + 8]), counts[i]) for b, i in zip(bins[order], order)]
        out["colors"] = [_hex(c) for c, _ in top[:3]]
        total = float(counts.sum())
        out["accent"] = _hex(max(top, key=lambda item: (item[0].max() - item[0].min()) * (0.4 + item[1] / total))[0])
    else:
        out["colors"], out["accent"] = [], "#FFFFFF"

    # focus: centre of the busiest 20% of the picture, pulled 25% toward the middle
    h, w = detail.shape
    if detail.max() > 0:
        cut = np.quantile(detail[seen], 0.8) if seen.any() else 0
        busy = np.where(detail >= cut, detail, 0.0)
        ys, xs = np.mgrid[0:h, 0:w]
        fx, fy = (busy * xs).sum() / busy.sum() / w, (busy * ys).sum() / busy.sum() / h
    else:
        fx = fy = 0.5
    out["focus"] = [round(float(0.75 * fx + 0.25 * 0.5), 3), round(float(0.75 * fy + 0.25 * 0.5), 3)]

    # 3 x 3 grid: which ink reads on each zone, and which zones are calm enough to carry text
    luma_grid, detail_grid = _blocks(luma, 3, 3), _blocks(detail, 3, 3)
    top_detail = max(float(detail_grid.max()), 1e-6)
    out["grid"] = [[{"luma": round(float(luma_grid[r, c]), 3), "detail": round(float(detail_grid[r, c] / top_detail), 3),
                     "ink": "dark" if luma_grid[r, c] > 0.5 else "light",
                     "quiet": bool(detail_grid[r, c] / top_detail < QUIET_DETAIL)} for c in range(3)] for r in range(3)]
    return out


def analyze(path: Path | str, use_cache: bool = True) -> Dict[str, Any]:
    path = Path(path)
    stamp = f"{path.stat().st_size}:{int(path.stat().st_mtime)}"
    cache_file = _cache_path()
    cache: Dict[str, Any] = {}
    if use_cache and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    hit = cache.get(str(path))
    if hit and hit.get("stamp") == stamp:
        return hit["analysis"]
    result = analyze_image(path)
    cache[str(path)] = {"stamp": stamp, "analysis": result}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return result


def analyze_many(files: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    return {media_id: analyze(path) for media_id, path in files.items() if Path(path).exists()}


def quiet_cells(analysis: Dict[str, Any]) -> List[str]:
    """Names of the calm zones, for example ['top_left', 'bottom_center']."""
    rows, cols = ["top", "middle", "bottom"], ["left", "center", "right"]
    return [f"{rows[r]}_{cols[c]}" for r in range(3) for c in range(3) if analysis["grid"][r][c]["quiet"]]


def _line(media_id: str, a: Dict[str, Any]) -> str:
    return (f"{media_id:<26}{a['size'][0]}x{a['size'][1]:<6}{a['tone']:<6}luma {a['luma']:<5} focus {a['focus']}  "
            f"{'SOFT ' if a['soft'] else ''}{'cut-out ' + str(a['visible']) + ' ' if a['has_alpha'] else ''}"
            f"accent {a['accent']}  quiet: {', '.join(quiet_cells(a)) or '-'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", help="one picture")
    parser.add_argument("--from-run", help="shared_context.json: analyse every picture of that run")
    args = parser.parse_args()
    if args.target:
        print(json.dumps(analyze(args.target, use_cache=False), indent=2))
        return
    if not args.from_run:
        parser.error("give a picture or --from-run")
    from core.run_context import RunContext
    ctx = RunContext.load(args.from_run)
    folder = Path(ctx.run_directory) / "images"
    files = {p.stem: p for p in sorted(folder.glob("*.png"), key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.stem)])}
    for name, path in files.items():
        print(_line(name, analyze(path)))
    print(f"\n{len(files)} picture(s) analysed; cached in {_cache_path()}")


if __name__ == "__main__":
    main()