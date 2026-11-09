"""
Kestrel Shorts - Asset geometry
===============================

Measures where the VISIBLE pixels of an image are (avatar poses have
transparent padding) and does the scaling arithmetic, so a character is placed
by what the viewer sees, not by its padded PNG. It replaces the old
assets/asset_analyzer.py, assets/asset_geometry.py and the AssetGeometry class
in composition/layout_engine.py.

The alpha threshold comes from config.AVATAR_ALPHA_THRESHOLD. Results are
cached in runtime/asset_geometry.json (keyed by file and modified time), so a
pose is measured once.

    python -m engines.visual.asset_geometry      # a table of every registered avatar pose
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image

from core import config
from core.geometry import Rect


@dataclass(frozen=True)
class AssetGeometry:
    path: str
    canvas_width: int
    canvas_height: int
    content: Rect                      # the visible pixels, in the image's own pixel coordinates

    @property
    def has_content(self) -> bool:
        return self.content.width > 0 and self.content.height > 0

    def scale_for_height(self, visible_height: float) -> float:
        if not self.has_content:
            raise ValueError(f"{self.path} has no visible pixels.")
        return visible_height / self.content.height

    def scale_for_width(self, visible_width: float) -> float:
        if not self.has_content:
            raise ValueError(f"{self.path} has no visible pixels.")
        return visible_width / self.content.width

    def visible_size(self, scale: float) -> Tuple[float, float]:
        return self.content.width * scale, self.content.height * scale

    def canvas_rect_for(self, visible_x: float, visible_y: float, scale: float) -> Rect:
        """Where the whole PNG goes so that its visible part starts at (visible_x, visible_y)."""
        return Rect(visible_x - self.content.x * scale, visible_y - self.content.y * scale,
                    self.canvas_width * scale, self.canvas_height * scale)


def _cache_file() -> Path:
    return Path(config.RUNS_DIR).parent / "asset_geometry.json"


def _load_cache() -> Dict[str, Any]:
    try:
        return json.loads(_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        _cache_file().parent.mkdir(parents=True, exist_ok=True)
        _cache_file().write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


def measure(path: Path | str, threshold: Optional[int] = None) -> AssetGeometry:
    """Measure an image's visible pixels (cached)."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    threshold = int(config.AVATAR_ALPHA_THRESHOLD if threshold is None else threshold)
    stamp = f"{path.stat().st_mtime_ns}:{threshold}"
    cache = _load_cache()
    hit = cache.get(str(path))
    if hit and hit.get("stamp") == stamp:
        return AssetGeometry(str(path), hit["w"], hit["h"], Rect(*hit["box"]))

    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        width, height = rgba.size
        alpha = rgba.getchannel("A")
        if threshold > 1:
            alpha = alpha.point(lambda value: 255 if value >= threshold else 0)
        box = alpha.getbbox()
    content = Rect(box[0], box[1], box[2] - box[0], box[3] - box[1]) if box else Rect(0, 0, 0, 0)
    cache[str(path)] = {"stamp": stamp, "w": width, "h": height,
                        "box": [content.x, content.y, content.width, content.height]}
    _save_cache(cache)
    return AssetGeometry(str(path), width, height, content)


def pose_path(avatar: Dict[str, Any], pose_id: str) -> Path:
    """The PNG of a pose, from a registry avatar entry (folder is relative to the project root)."""
    directory = Path(config.SHORTS_ROOT) / str((avatar.get("assets") or {}).get("poses_directory", ""))
    for pose in avatar.get("poses", []):
        if pose.get("pose_id") == pose_id:
            return directory / str(pose.get("file"))
    raise KeyError(f"Avatar {avatar.get('avatar_id')!r} has no pose {pose_id!r}.")


def pose_geometry(character_id: str, pose_id: Optional[str] = None, registry: Any = None) -> AssetGeometry:
    """Geometry of a character's pose (the default pose when none is given)."""
    from core.registry_loader import RegistryLoader
    registry = registry or RegistryLoader()
    avatar = registry.get_avatar(registry.get_character(character_id)["avatar_id"])
    return measure(pose_path(avatar, pose_id or avatar["default_pose"]))


def _main() -> None:
    from core.registry_loader import RegistryLoader

    registry = RegistryLoader()
    print(f"alpha threshold: {config.AVATAR_ALPHA_THRESHOLD}\n")
    print(f"{'avatar / pose':<44}{'canvas':>11}  {'visible x, y, w x h':<26}{'padding L/T/R/B':<22}note")
    for avatar in registry.list_entries("avatars"):
        default_height: Optional[float] = None
        poses = sorted(avatar.get("poses", []), key=lambda p: p.get("pose_id") != avatar.get("default_pose"))
        for pose in poses:
            name = f"{avatar['avatar_id']} / {pose['pose_id']}"
            try:
                geometry = measure(pose_path(avatar, pose["pose_id"]))
            except (OSError, KeyError) as exc:
                print(f"{name:<44}PROBLEM: {exc}")
                continue
            c, note = geometry.content, ""
            if not geometry.has_content:
                note = "NO VISIBLE PIXELS"
            elif default_height is None:
                default_height = c.height
            elif abs(c.height - default_height) / default_height > 0.25:
                note = f"height is {c.height / default_height:.0%} of the default pose"
            pad = f"{c.x:.0f}/{c.y:.0f}/{geometry.canvas_width - c.right:.0f}/{geometry.canvas_height - c.bottom:.0f}"
            print(f"{name:<44}{geometry.canvas_width:>5}x{geometry.canvas_height:<5}  "
                  f"{c.x:.0f}, {c.y:.0f}, {c.width:.0f} x {c.height:.0f}".ljust(81) + f"{pad:<22}{note}")


if __name__ == "__main__":
    _main()