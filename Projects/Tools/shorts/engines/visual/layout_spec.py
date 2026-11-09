"""
Kestrel Shorts - Layout spec
============================

The shared numbers of the screen, and the arithmetic built on them. Remotion
draws what it is told; this file owns where things may go and how much fits.
Everything tunable is at the top.

  Regions    SAFE (where text, tables, faces and key items must stay), MEDIA (where images and
             posters may go: wider), CONTENT (the safe area minus the least strip a subtitle
             needs: where text-like content is planned) and the whole screen (kinetic text).
  Subtitles  An overlay layer, not a reserved band. They may lie over pictures, backdrops and
             cut-outs, but never over text, tables or other elements (an avatar does not block them). 
             subtitle_slot() finds a free strip (the preferred place at the bottom, else under or above the
             content, else a gap) or says there is none: the shot is overloaded.
  Anchors    The 15 named positions, placed inside an area.
  Avatars    Size and position from the VISIBLE pixels of a pose.
  Images     The best grid for N images of a given shape, and whether they are still big
             enough to recognise (MIN_IMAGE_SIDE).
  Text       Measured with the same Inter files Remotion loads: does it fit, at what size, and
             how to split a table or list that does not. A small tolerance (FIT_TOLERANCE) keeps
             content that is only slightly too big from being shrunk or split.

    python -m engines.visual.layout_spec                 # a summary and examples
    python -m engines.visual.layout_spec --guide         # a PNG to look at on your phone
    python -m engines.visual.layout_spec --export        # layout_spec.json for Remotion
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from core import config
from core.geometry import Rect
from engines.visual.asset_geometry import AssetGeometry

# ----------------------------------------------------------------------
# Tuning constants (pixels on the 1080x1920 canvas)
# ----------------------------------------------------------------------

CANVAS = Rect(0, 0, config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
# Text, tables, faces and key items stay inside SAFE. It is strict because Shorts are posted to YouTube
# Shorts, Instagram Reels and TikTok, whose interfaces cover different parts of the screen.
SAFE = {"top": 200, "bottom": 520, "left": 140, "right": 140}
# Images, posters and graphics may use more of the screen (a button may cover a corner of them).
MEDIA = {"top": 120, "left": 60, "right": 60}
SUBTITLE_PREFERRED_HEIGHT = 280     # where regular subtitles like to sit: the lowest strip of the safe area
SUBTITLE_MIN_HEIGHT = 180           # the least a regular subtitle needs (about two lines)
GUTTER = 24                         # space between neighbouring items
MIN_IMAGE_SIDE = 220                # an image smaller than this is too small to recognise
LINE_HEIGHT = 1.22
FIT_TOLERANCE = 0.08                # content may overshoot its region by this share before it counts as not fitting
FIT_TOP_GIVE = 40                   # ...but at most this many pixels upwards (the top 200 px belong to the apps)
CELL_PAD = (20, 14)                 # table cell padding: vertical, horizontal
FONT_DIR = Path(r"C:\AI_Studio\Projects\Remotion\public\fonts\static")
FONT_FILES = {"bold": "Inter_28pt-Bold.ttf", "text": "Inter_28pt-Medium.ttf"}
FONT_PX = {                         # kind: (smallest readable size, largest size)
    "attention": (72, 140), "explanatory": (44, 64), "table": (38, 52), "list": (42, 60),
    "label": (36, 48), "subtitle": (52, 64)}
AVATAR_HEIGHT_SHARE = {"small": 0.22, "medium": 0.32, "large": 0.45}    # visible height / canvas height

ANCHORS = ["top_left", "top_center", "top_right", "upper_left", "upper_center", "upper_right",
           "center_left", "center", "center_right", "lower_left", "lower_center", "lower_right",
           "bottom_left", "bottom_center", "bottom_right"]
_ROW_FRACTION = {"upper": 0.25, "center": 0.5, "lower": 0.75}


# ----------------------------------------------------------------------
# Regions and anchors
# ----------------------------------------------------------------------

def safe_rect() -> Rect:
    return Rect(SAFE["left"], SAFE["top"], CANVAS.width - SAFE["left"] - SAFE["right"],
                CANVAS.height - SAFE["top"] - SAFE["bottom"])


def subtitle_rect() -> Rect:
    """Where regular subtitles prefer to sit: the lowest strip of the safe area. Not reserved."""
    safe = safe_rect()
    return Rect(safe.x, safe.bottom - SUBTITLE_PREFERRED_HEIGHT, safe.width, SUBTITLE_PREFERRED_HEIGHT)


def content_rect() -> Rect:
    """Where text-like content (text, tables, lists, elements, avatars) is planned: the safe area minus the
    least strip a subtitle needs. Where the content and that strip really sit is decided later: the
    subtitles can move to the bottom, the top or a gap (see subtitle_slot)."""
    safe = safe_rect()
    return Rect(safe.x, safe.y, safe.width, safe.height - SUBTITLE_MIN_HEIGHT - GUTTER)


def media_rect() -> Rect:
    """Where images, posters and graphics go: wider than the safe area, down to its bottom.
    Subtitles may lie over them."""
    return Rect(MEDIA["left"], MEDIA["top"], CANVAS.width - MEDIA["left"] - MEDIA["right"],
                safe_rect().bottom - MEDIA["top"])


def kinetic_rect() -> Rect:
    """Kinetic subtitles may use the whole screen."""
    return CANVAS


def ui_overlap(rect: Rect) -> float:
    """The share of a rectangle that lies outside the safe area, where the apps may cover it (0 to 1)."""
    if rect.area <= 0:
        return 0.0
    inside = rect.intersection(safe_rect())
    return 1.0 - (inside.area / rect.area if inside else 0.0)


def subtitle_slot(occupied: List[Rect], height: float = SUBTITLE_MIN_HEIGHT) -> Optional[Rect]:
    """Where regular subtitles can go, given the rectangles of everything they must not cover (text,
    tables, other elements; pictures, backdrops, cut-outs and the avatar do not count: subtitles win).
    Full-width strips only.
    Preference: the lowest strip of the safe area, then the top, then the roomiest gap in between.
    Returns None when no strip of at least `height` is free: the shot is overloaded."""
    safe = safe_rect()
    blocks = sorted((max(safe.y, r.y - GUTTER), min(safe.bottom, r.bottom + GUTTER))
                    for r in occupied if r.intersects(safe))
    free: List[tuple] = []
    cursor = safe.y
    for top, bottom in blocks:
        if top > cursor:
            free.append((cursor, top))
        cursor = max(cursor, bottom)
    if cursor < safe.bottom:
        free.append((cursor, safe.bottom))
    roomy = [gap for gap in free if gap[1] - gap[0] >= height]
    if not roomy:
        return None
    if roomy[-1][1] >= safe.bottom - 1:                     # a free strip reaches the bottom of the safe area
        start, end = roomy[-1]
        h = min(SUBTITLE_PREFERRED_HEIGHT, end - start)
        y = end - h
    elif roomy[0][0] <= safe.y + 1:                         # a free strip at the top
        start, end = roomy[0]
        h = min(SUBTITLE_PREFERRED_HEIGHT, end - start)
        y = start
    else:                                                   # the roomiest gap in between
        start, end = max(roomy, key=lambda gap: gap[1] - gap[0])
        h = min(SUBTITLE_PREFERRED_HEIGHT, end - start)
        y = start + (end - start - h) / 2
    return Rect(safe.x, y, safe.width, h)


def anchor_rect(anchor: str, width: float, height: float, area: Optional[Rect] = None) -> Rect:
    """A width x height box placed at a named anchor inside `area` (default: the content area)."""
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor: {anchor!r}")
    area = area or content_rect()
    parts = anchor.split("_")
    row, column = parts[0], (parts[1] if len(parts) > 1 else "center")
    x = {"left": area.x, "center": area.x + (area.width - width) / 2, "right": area.right - width}[column]
    if row == "top":
        y = area.y
    elif row == "bottom":
        y = area.bottom - height
    else:
        y = area.y + _ROW_FRACTION[row] * area.height - height / 2
    return Rect(x, y, width, height).clamp_inside(area)


def avatar_placement(geometry: AssetGeometry, anchor: str, size: str) -> Dict[str, Any]:
    """Where an avatar pose goes: its visible rectangle sits at the anchor; the PNG is placed to match."""
    scale = geometry.scale_for_height(AVATAR_HEIGHT_SHARE[size] * CANVAS.height)
    width, height = geometry.visible_size(scale)
    visible = anchor_rect(anchor, width, height)
    return {"visible": visible, "canvas": geometry.canvas_rect_for(visible.x, visible.y, scale), "scale": scale}


def free_area(taken: Rect, area: Optional[Rect] = None) -> Rect:
    """The largest rectangle left in `area` after removing `taken` (the space beside an avatar)."""
    area = area or content_rect()
    if not area.intersects(taken):
        return area
    candidates = [
        Rect(area.x, area.y, area.width, max(0.0, taken.y - area.y)),                     # above
        Rect(area.x, taken.bottom, area.width, max(0.0, area.bottom - taken.bottom)),     # below
        Rect(area.x, area.y, max(0.0, taken.x - area.x), area.height),                    # left
        Rect(taken.right, area.y, max(0.0, area.right - taken.right), area.height),       # right
    ]
    return max(candidates, key=lambda r: r.area)


# ----------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------

def fit_images(n: int, aspect: float = 2 / 3, area: Optional[Rect] = None) -> Dict[str, Any]:
    """The best grid for n images of shape width/height = aspect: cols, rows, item size, slot
    rectangles, and whether each image is still about MIN_IMAGE_SIDE on its short side (with tolerance)."""
    area = area or media_rect()
    n = max(1, n)
    best: Optional[Dict[str, Any]] = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = (area.width - GUTTER * (cols - 1)) / cols
        cell_h = (area.height - GUTTER * (rows - 1)) / rows
        if cell_w <= 0 or cell_h <= 0:
            continue
        item_w = min(cell_w, cell_h * aspect)
        if best is None or item_w > best["item_w"]:
            best = {"cols": cols, "rows": rows, "item_w": item_w, "item_h": item_w / aspect}
    if best is None:
        raise ValueError("The area is too small for any grid.")
    grid_h = best["rows"] * best["item_h"] + GUTTER * (best["rows"] - 1)
    y0 = area.y + (area.height - grid_h) / 2
    slots_: List[Rect] = []
    for i in range(n):
        r, c = divmod(i, best["cols"])
        in_row = min(best["cols"], n - r * best["cols"])
        row_w = in_row * best["item_w"] + GUTTER * (in_row - 1)
        x0 = area.x + (area.width - row_w) / 2
        slots_.append(Rect(x0 + c * (best["item_w"] + GUTTER), y0 + r * (best["item_h"] + GUTTER),
                           best["item_w"], best["item_h"]))
    best["slots"] = slots_
    best["feasible"] = min(best["item_w"], best["item_h"]) >= MIN_IMAGE_SIDE * (1 - FIT_TOLERANCE)
    return best


def max_images(aspect: float = 2 / 3, area: Optional[Rect] = None, limit: int = 30) -> int:
    """The most images of this shape that still fit at a recognisable size."""
    return max((n for n in range(1, limit + 1) if fit_images(n, aspect, area)["feasible"]), default=0)


def layout_for_images(n: int) -> str:
    """A layout name that holds n images (the Visual Director lists images; code picks the layout)."""
    return "text_only" if n <= 0 else "single" if n == 1 else "stack" if n <= 3 else "grid"


def slots(layout: str, n: int = 1, aspect: float = 2 / 3) -> List[Dict[str, Any]]:
    """Image slots of a layout: [{'rect': Rect, 'rotation': degrees}]. Names like 'grid_4' are accepted."""
    kind = layout.split("_")[0]
    if kind == "text":
        return []
    if kind in ("single", "sequence"):
        return [{"rect": CANVAS, "rotation": 0.0} for _ in range(max(1, n))]
    if kind == "inset":
        side = safe_rect().width * 0.38
        return [{"rect": CANVAS, "rotation": 0.0},
                {"rect": anchor_rect("upper_right", side, side, safe_rect()), "rotation": 0.0}]
    if kind == "stack":
        area = media_rect()
        height = (area.height - GUTTER * (n - 1)) / n
        return [{"rect": Rect(area.x, area.y + i * (height + GUTTER), area.width, height), "rotation": 0.0}
                for i in range(n)]
    angles = [-5.0, 4.0, -3.0, 6.0, -4.0, 3.0]
    rects = fit_images(n, aspect)["slots"]
    return [{"rect": r, "rotation": angles[i % len(angles)] if kind == "scatter" else 0.0}
            for i, r in enumerate(rects)]


# ----------------------------------------------------------------------
# Text, tables and lists (measured with the same fonts Remotion loads)
# ----------------------------------------------------------------------

_WARNED = False


@lru_cache(maxsize=128)
def _font(bold: bool, size: int) -> Any:
    global _WARNED
    name = FONT_FILES["bold" if bold else "text"]
    for candidate in (FONT_DIR / name, name, "arial.ttf"):
        try:
            font = ImageFont.truetype(str(candidate), size)
            if candidate == "arial.ttf" and not _WARNED:
                print(f"[Layout] WARNING: {name} not found in {FONT_DIR}; measuring with Arial instead.")
                _WARNED = True
            return font
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(text: str, font: Any, max_width: float) -> List[str]:
    """Greedy word wrap. A single word wider than max_width stays on its own line."""
    max_width = max(max_width, 1.0)
    lines: List[str] = []
    line = ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if not line or font.getlength(trial) <= max_width:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def soft_region(region: Rect) -> Rect:
    """A region with the fit tolerance applied: wider by FIT_TOLERANCE (split between both sides) and taller
    upwards by FIT_TOLERANCE, at most FIT_TOP_GIVE pixels. Never downwards: below the safe area the
    apps' interface starts. The Frame Supervisor places tolerant content in this rectangle."""
    grow_w = region.width * FIT_TOLERANCE
    grow_h = min(region.height * FIT_TOLERANCE, FIT_TOP_GIVE)
    return Rect(region.x - grow_w / 2, region.y - grow_h, region.width + grow_w, region.height + grow_h)


def fit_text(text: str, kind: str, region: Rect) -> Dict[str, Any]:
    """The largest size (between the kind's minimum and maximum) at which `text` fits `region`.
    'fits' is False when even the smallest readable size overflows."""
    region = soft_region(region)
    low, high = FONT_PX[kind]
    size = high
    while True:
        font = _font(kind in ("attention", "subtitle"), size)
        lines = wrap(text, font, region.width)
        width = max(font.getlength(line) for line in lines)
        height = len(lines) * size * LINE_HEIGHT
        fits = width <= region.width and height <= region.height
        if fits or size <= low:
            return {"fits": fits, "font_px": size, "lines": lines, "width": round(width), "height": round(height)}
        size = max(low, size - 4)


def _table_at(columns: List[str], rows: List[List[str]], region: Rect, size: int) -> Dict[str, Any]:
    head, body = _font(True, size), _font(False, size)
    pad_v, pad_h = CELL_PAD
    n = len(columns)
    table = [[str(c) for c in columns]] + [[str(r[i]) if i < len(r) else "" for i in range(n)] for r in rows]
    fonts = [head] + [body] * len(rows)
    natural = [max(f.getlength(row[i]) for row, f in zip(table, fonts)) + 2 * pad_h for i in range(n)]
    narrow = [max((f.getlength(w) for row, f in zip(table, fonts) for w in row[i].split()), default=0) + 2 * pad_h
              for i in range(n)]
    if sum(narrow) > region.width:
        return {"fits": False, "font_px": size, "reason": "the columns are too wide", "col_widths": [],
                "total_height": None, "rows_that_fit": 0}
    extra = region.width - sum(narrow)
    need = [max(0.0, natural[i] - narrow[i]) for i in range(n)]
    used = min(extra, sum(need))
    widths = [narrow[i] + (used * need[i] / sum(need) if sum(need) else 0.0) + (extra - used) / n for i in range(n)]

    def height_of(row: List[str], font: Any) -> float:
        return max(len(wrap(row[i], font, widths[i] - 2 * pad_h)) for i in range(n)) * size * LINE_HEIGHT + 2 * pad_v

    header = height_of(table[0], head)
    heights = [height_of(row, body) for row in table[1:]]
    total = header + sum(heights)
    fit_rows, running = 0, header
    for h in heights:
        if running + h > region.height:
            break
        running += h
        fit_rows += 1
    return {"fits": total <= region.height, "font_px": size, "col_widths": [round(w) for w in widths],
            "header_height": round(header), "row_heights": [round(h) for h in heights],
            "total_height": round(total), "rows_that_fit": fit_rows}


def table_fit(columns: List[str], rows: List[List[str]], region: Optional[Rect] = None) -> Dict[str, Any]:
    """Does the table fit the region at a readable size? (largest size first)"""
    region = region or content_rect()
    region = soft_region(region)
    low, high = FONT_PX["table"]
    size = high
    while True:
        result = _table_at(columns, rows, region, size)
        if result["fits"] or size <= low:
            return result
        size = max(low, size - 4)


def _list_at(items: List[str], region: Rect, size: int) -> Dict[str, Any]:
    font = _font(False, size)
    indent = size * 1.2
    wrapped = [wrap(item, font, region.width - indent) for item in items]
    widest = max((font.getlength(line) for lines in wrapped for line in lines), default=0)
    total = sum(len(lines) * size * LINE_HEIGHT for lines in wrapped) + size * 0.45 * max(0, len(items) - 1)
    return {"fits": widest <= region.width - indent and total <= region.height, "font_px": size,
            "total_height": round(total)}


def list_fit(items: List[str], region: Optional[Rect] = None) -> Dict[str, Any]:
    """Does the list fit the region at a readable size? (largest size first)"""
    region = region or content_rect()
    region = soft_region(region)
    low, high = FONT_PX["list"]
    size = high
    while True:
        result = _list_at(items, region, size)
        if result["fits"] or size <= low:
            return result
        size = max(low, size - 4)


def _split(items: List[Any], fits: Callable[[List[Any]], bool], max_parts: int = 8) -> List[List[Any]]:
    if fits(list(items)):
        return [list(items)]
    for parts in range(2, min(max_parts, len(items)) + 1):
        size = math.ceil(len(items) / parts)
        groups = [list(items[i:i + size]) for i in range(0, len(items), size)]
        if all(fits(g) for g in groups):
            return groups
    return []                                          # even one item per part does not fit


def split_rows(columns: List[str], rows: List[List[str]], region: Optional[Rect] = None) -> List[List[List[str]]]:
    """Row groups that each fit the region ([] if impossible); a table that fits comes back as one group."""
    region = region or content_rect()
    return _split(rows, lambda g: table_fit(columns, g, region)["fits"])


def split_table(columns: List[str], rows: List[List[str]], region: Optional[Rect] = None,
                key_columns: int = 1) -> List[Dict[str, Any]]:
    """How to show a table that does not fit. Columns are split into groups (the first `key_columns`
    columns repeat in every group, like the angle column of a trigonometry table) and, if needed, rows
    into parts too. Returns parts: [{'columns', 'rows', 'column_indexes', 'row_indexes'}] ([] if impossible).
    A table that already fits comes back as one part."""
    region = region or content_rect()
    n = len(columns)
    key = list(range(min(key_columns, n)))
    rest = [i for i in range(n) if i not in key]
    everything = list(range(len(rows)))

    def cells(col_idx: List[int], row_idx: List[int]) -> List[List[str]]:
        return [[str(rows[r][i]) if i < len(rows[r]) else "" for i in col_idx] for r in row_idx]

    for groups in range(1, max(1, len(rest)) + 1):                      # fewest column groups first
        size = math.ceil(len(rest) / groups) if rest else 1
        col_groups = [key + rest[i:i + size] for i in range(0, len(rest), size)] or [key]
        row_parts = _split(everything, lambda g: all(
            table_fit([columns[i] for i in cg], cells(cg, g), region)["fits"] for cg in col_groups))
        if row_parts:
            return [{"columns": [columns[i] for i in cg], "rows": cells(cg, rp),
                     "column_indexes": cg, "row_indexes": rp} for cg in col_groups for rp in row_parts]
    return []


def split_items(items: List[str], region: Optional[Rect] = None) -> List[List[str]]:
    """The same for a list."""
    region = region or content_rect()
    return _split(items, lambda g: list_fit(g, region)["fits"])


# ----------------------------------------------------------------------
# Export, guide and examples
# ----------------------------------------------------------------------

def spec() -> Dict[str, Any]:
    """The numbers Remotion needs (for safe-area overlays and consistent text sizes)."""
    return {"version": 2, "canvas": CANVAS.to_dict(), "safe": safe_rect().to_dict(),
            "content": content_rect().to_dict(), "media": media_rect().to_dict(),
            "subtitles": {"preferred": subtitle_rect().to_dict(), "min_height": SUBTITLE_MIN_HEIGHT, "overlay": True},
            "kinetic": kinetic_rect().to_dict(),
            "gutter": GUTTER, "min_image_side": MIN_IMAGE_SIDE, "line_height": LINE_HEIGHT,
            "fit_tolerance": FIT_TOLERANCE,
            "font_px": {k: {"min": v[0], "max": v[1]} for k, v in FONT_PX.items()},
            "fonts": {"folder": "fonts/static", **FONT_FILES}, "avatar_height_share": AVATAR_HEIGHT_SHARE}


def export(path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec(), indent=2), encoding="utf-8")
    return path


def write_guide(path: Path | str) -> Path:
    """A 1080x1920 picture of the regions with every text kind at its smallest size, to judge on a phone."""
    width, height = int(CANVAS.width), int(CANVAS.height)
    image = Image.new("RGB", (width, height), (24, 24, 28))
    backdrop = ImageDraw.Draw(image)
    for y in range(height):                      # a stand-in background: backgrounds fill the whole screen
        shade = int(30 + 60 * y / height)
        backdrop.line([(0, y), (width, y)], fill=(shade, shade + 10, shade + 35))
    draw = ImageDraw.Draw(image, "RGBA")
    safe, content, media, sub = safe_rect(), content_rect(), media_rect(), subtitle_rect()
    danger = (200, 60, 60, 90)
    draw.rectangle([0, 0, width, safe.y], fill=danger)
    draw.rectangle([0, safe.bottom, width, height], fill=danger)
    draw.rectangle([0, safe.y, safe.x, safe.bottom], fill=danger)
    draw.rectangle([safe.right, safe.y, width, safe.bottom], fill=danger)
    draw.rectangle([content.x, content.y, content.right, content.bottom], outline=(80, 200, 120, 255), width=4)
    draw.rectangle([media.x, media.y, media.right, media.bottom], outline=(240, 200, 60, 255), width=4)
    draw.rectangle([sub.x, sub.y, sub.right, sub.bottom], outline=(90, 160, 255, 255), width=4)
    draw.text((sub.x + 20, sub.y + 20), "subtitles prefer this strip", fill=(255, 255, 255, 255), font=_font(False, 40))
    y = int(content.y) + 20
    for kind, (low, _high) in FONT_PX.items():
        draw.text((content.x + 20, y), f"{kind} {low}px", fill=(255, 255, 255, 255),
                  font=_font(kind in ("attention", "subtitle"), low))
        y += int(low * LINE_HEIGHT) + 16
    mock = (255, 255, 255, 60)                       # a rough picture of what the apps draw over the video
    for i in range(3):
        draw.rounded_rectangle([60, 1540 + i * 70, 760 - i * 150, 1580 + i * 70], radius=20, fill=mock)
    for i in range(4):
        draw.ellipse([940, 1000 + i * 150, 1020, 1080 + i * 150], fill=mock)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Show the layout spec, or write its guide picture or JSON.")
    parser.add_argument("--guide", nargs="?", const=str(Path(config.SHORTS_TEMP_DIR) / "layout_guide.png"))
    parser.add_argument("--export", nargs="?", const=str(Path(config.SHORTS_ROOT) / "runtime" / "layout_spec.json"))
    args = parser.parse_args()
    if args.guide:
        print("Guide picture:", write_guide(args.guide))
    if args.export:
        print("Spec JSON:", export(args.export))

    print("Regions (pixels on a 1080x1920 canvas):")
    for name, rect in (("safe", safe_rect()), ("content", content_rect()), ("media", media_rect()),
                       ("subtitles (preferred)", subtitle_rect())):
        print(f"  {name:<22}{rect}")
    print(f"\nImages of 2:3 (posters): up to {max_images(2 / 3)} fit at about {MIN_IMAGE_SIDE} px; "
          f"of 3:4: up to {max_images(3 / 4)}; square: up to {max_images(1.0)}")
    for n in (4, 8, 12):
        g = fit_images(n, 2 / 3)
        print(f"  {n:>2} posters -> {g['cols']}x{g['rows']} grid, each {g['item_w']:.0f}x{g['item_h']:.0f} px"
              f"{'' if g['feasible'] else '  (too small)'}")

    columns = ["Angle", "sin", "cos", "tan", "cosec", "sec", "cot"]
    rows = [["0", "0", "1", "0", "-", "1", "-"], ["30", "1/2", "sqrt3/2", "1/sqrt3", "2", "2/sqrt3", "sqrt3"],
            ["45", "1/sqrt2", "1/sqrt2", "1", "sqrt2", "sqrt2", "1"],
            ["60", "sqrt3/2", "1/2", "sqrt3", "2/sqrt3", "2", "1/sqrt3"], ["90", "1", "0", "-", "1", "-", "0"]]
    t = table_fit(columns, rows)
    print(f"\nTable 5 rows x 7 columns: fits={t['fits']} ({t.get('reason', 'ok')})")
    for label, data in (("5 rows", rows), ("25 rows", rows * 5)):
        parts = split_table(columns, data)
        print(f"  {label}: {len(parts)} part(s): "
              + "; ".join(f"{p['columns']} x {len(p['rows'])} rows" for p in parts))

    print("\nWhere the subtitles go:")
    for label, blocks in (("nothing else on screen", []),
                          ("a table filling the content area", [content_rect()]),
                          ("text at the top and a table at the bottom", [Rect(140, 200, 800, 300), Rect(140, 1000, 800, 400)]),
                          ("content filling the whole safe area", [safe_rect()])):
        slot = subtitle_slot(blocks)
        print(f"  {label}: {slot if slot else 'NO ROOM: the shot is overloaded and must be split'}")
    print(f"\nA kinetic text box inside the apps' zone: {ui_overlap(Rect(100, 1500, 600, 200)):.0%} covered")

    sample = "The retina turns light into electrical signals"
    f = fit_text(sample, "attention", Rect(0, 0, safe_rect().width, 400))
    print(f"Attention text '{sample}': fits={f['fits']} at {f['font_px']} px in {len(f['lines'])} line(s)")


if __name__ == "__main__":
    _main()