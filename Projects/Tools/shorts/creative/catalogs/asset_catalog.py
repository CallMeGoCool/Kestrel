"""
Kestrel Shorts - Asset catalog
==============================

The visual and audio assets the directors may choose from, found by scanning folders in the Remotion
project. A file's name (without extension) is its catalog name; an optional _hints.json in the folder
maps a name to a one-line hint. Drop a file in a folder and it is available: no code change.

  visual kinds  overlay, transition, line, texture, particle
  audio kinds   music, sfx, ambience

Folders (created by --init), under C:\\AI_Studio\\Projects\\Remotion\\public:
  kestrel/overlays | transitions | lines | textures | particles        audio/music | sfx | ambience

Effects Remotion can draw by itself (grain, vignette, cut, fade, slide, wipe, zoom_through) are always
available. Others (page_tear, paper_texture, light_leak, dust) appear once a file with that name exists.
When a director wants something the library lacks, it adds it to the wishlist; the end-of-run report then
tells you what to add.

    python -m creative.catalogs.asset_catalog            # what is in each folder, and the wishlist
    python -m creative.catalogs.asset_catalog --init     # create the folders
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from core import config
from creative.catalogs import visual_catalog as vc

REMOTION_PUBLIC = Path(r"C:\AI_Studio\Projects\Remotion\public")
FOLDERS = {
    "overlay": "kestrel/overlays", "transition": "kestrel/transitions", "line": "kestrel/lines",
    "texture": "kestrel/textures", "particle": "kestrel/particles",
    "music": "audio/music", "sfx": "audio/sfx", "ambience": "audio/ambience",
}
_PICTURE, _SOUND = {".png", ".webp", ".svg", ".jpg", ".jpeg"}, {".mp3", ".wav", ".ogg"}
EXTENSIONS = {
    "overlay": _PICTURE | {".mp4", ".webm"}, "transition": _PICTURE | {".mp4", ".webm"},
    "line": _PICTURE, "texture": _PICTURE, "particle": _PICTURE,
    "music": _SOUND, "sfx": _SOUND, "ambience": _SOUND,
}
BUILT_IN = {
    "overlay": {name: vc.OVERLAYS[name] for name in ("grain", "vignette")},
    "transition": {name: vc.TRANSITIONS[name] for name in (
        "cut", "fade", "slide", "wipe", "zoom_through", "flip", "clock_wipe", "whip_pan", "flash", "iris", "glitch")},
}
LARGE_FILE_MB = 5.0


class AssetError(ValueError):
    """An unknown asset kind."""


def folder(kind: str) -> Path:
    if kind not in FOLDERS:
        raise AssetError(f"Unknown asset kind {kind!r}; use one of: {', '.join(FOLDERS)}.")
    return REMOTION_PUBLIC / FOLDERS[kind]


def _hints(path: Path) -> Dict[str, str]:
    try:
        data = json.loads((path / "_hints.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def scan(kind: str) -> Dict[str, Dict[str, Any]]:
    """name -> {file, hint, builtin, size_mb}. Built-in effects come first; a file with the same name replaces one."""
    found: Dict[str, Dict[str, Any]] = {
        name: {"file": "", "hint": hint, "builtin": True, "size_mb": 0.0}
        for name, hint in BUILT_IN.get(kind, {}).items()}
    path = folder(kind)
    if path.is_dir():
        hints = _hints(path)
        for file in sorted(path.iterdir()):
            if file.is_file() and file.suffix.lower() in EXTENSIONS[kind]:
                found[file.stem] = {"file": file.relative_to(REMOTION_PUBLIC).as_posix(),
                                    "hint": hints.get(file.stem, ""), "builtin": False,
                                    "size_mb": round(file.stat().st_size / 1e6, 2)}
    return found


def menu(kind: str) -> str:
    """One line per asset, for a prompt: '- name: hint'."""
    return "\n".join(f"- {name}" + (f": {info['hint']}" if info["hint"] else "")
                     for name, info in scan(kind).items())


def exists(kind: str, name: str) -> bool:
    return name in scan(kind)


# ----------------------------------------------------------------------
# Wishlist: what the directors wanted and the library lacked
# ----------------------------------------------------------------------

def _wishlist_file() -> Path:
    return Path(config.RUNS_DIR).parent / "asset_wishlist.json"


def wishlist() -> List[Dict[str, Any]]:
    try:
        data = json.loads(_wishlist_file().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def request(kind: str, name: str, description: str, requested_by: str = "", run_id: str = "") -> None:
    """Note an asset a director wants but the library lacks (counted, so popular wishes rise to the top)."""
    folder(kind)                                              # checks the kind
    name = name.strip()
    if not name or exists(kind, name):
        return
    items = wishlist()
    for item in items:
        if item.get("kind") == kind and str(item.get("name", "")).lower() == name.lower():
            item["count"] = int(item.get("count", 1)) + 1
            item["runs"] = sorted(set(item.get("runs", [])) | ({run_id} if run_id else set()))
            break
    else:
        items.append({"kind": kind, "name": name, "description": description.strip(),
                      "requested_by": requested_by, "count": 1, "runs": [run_id] if run_id else []})
    try:
        _wishlist_file().parent.mkdir(parents=True, exist_ok=True)
        _wishlist_file().write_text(json.dumps(items, indent=2), encoding="utf-8")
    except OSError:
        pass


def init_folders() -> List[Path]:
    """Create the library folders (and an empty _hints.json in each, if missing)."""
    made = []
    for kind in FOLDERS:
        path = folder(kind)
        path.mkdir(parents=True, exist_ok=True)
        if not (path / "_hints.json").exists():
            (path / "_hints.json").write_text("{}\n", encoding="utf-8")
        made.append(path)
    return made


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Show the asset library and the wishlist.")
    parser.add_argument("--init", action="store_true", help="create the library folders")
    args = parser.parse_args()
    if args.init:
        for path in init_folders():
            print("Folder ready:", path)
        print()
    for kind in FOLDERS:
        assets = scan(kind)
        files = {n: i for n, i in assets.items() if not i["builtin"]}
        built = [n for n, i in assets.items() if i["builtin"]]
        note = "" if folder(kind).is_dir() else "   (folder missing: run with --init)"
        print(f"{kind:<11}{len(files):>3} file(s), {len(built)} built-in{note}")
        for name, info in list(files.items())[:6]:
            flag = f"   LARGE: {info['size_mb']} MB" if info["size_mb"] > LARGE_FILE_MB else ""
            print(f"    {name}" + (f" - {info['hint']}" if info["hint"] else "") + flag)
    wishes = sorted(wishlist(), key=lambda w: -int(w.get("count", 1)))
    print(f"\nWishlist: {len(wishes)} item(s)")
    for w in wishes[:10]:
        print(f"  {w.get('kind', ''):<11}{w.get('name', '')}  (x{w.get('count', 1)})  {w.get('description', '')}")


if __name__ == "__main__":
    _main()