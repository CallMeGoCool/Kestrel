"""
Kestrel Shorts - Voice environment probe
========================================

One command that reports everything the voice engine depends on, so the engine is written for the machine
as it really is: where F5-TTS and Kokoro live, which Python can import what (the shorts venv and the Python
that ships with ComfyUI), whether the models and reference clips exist, and how long each reference clip is.
It changes nothing.

    python -m engines.voice.probe
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from core import config

COMFY_PYTHON = Path(getattr(config, "COMFYUI_PYTHON", r"D:\AI Tools\ComfyUI_windows_portable\python_embeded\python.exe"))
COMFY_DIR = Path(getattr(config, "COMFYUI_DIR", r"D:\AI Tools\ComfyUI_windows_portable\ComfyUI"))
MODULES = ["f5_tts", "torch", "torchaudio", "transformers", "vocos", "faster_whisper", "ctranslate2", "kokoro_onnx",
           "onnxruntime", "soundfile", "librosa", "scipy", "numpy", "pyloudnorm", "pydub", "whisper"]

_CODE = r'''
import importlib.util, json, sys
out = {"python": sys.version.split()[0], "modules": {}}
for name in NAMES:
    try:
        out["modules"][name] = importlib.util.find_spec(name) is not None
    except Exception:
        out["modules"][name] = False
try:
    import torch
    out["torch"] = {"version": torch.__version__, "cuda": bool(torch.cuda.is_available()),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""}
except Exception as exc:
    out["torch"] = "not importable: " + str(exc)[:100]
print("@@" + json.dumps(out))
'''.replace("NAMES", repr(MODULES))


def _run(python: str) -> Dict[str, Any]:
    try:
        done = subprocess.run([python, "-c", _CODE], capture_output=True, text=True, timeout=180)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    for line in done.stdout.splitlines():
        if line.startswith("@@"):
            return json.loads(line[2:])
    return {"error": (done.stderr or done.stdout or "no output").strip()[-300:]}


def _show_env(title: str, info: Dict[str, Any]) -> None:
    print(f"\n== {title}")
    if "error" in info:
        print("   could not run:", info["error"])
        return
    have = [m for m, ok in info["modules"].items() if ok]
    missing = [m for m, ok in info["modules"].items() if not ok]
    print(f"   python {info['python']}")
    print(f"   torch: {info['torch']}")
    print(f"   importable: {', '.join(have) or 'none'}")
    print(f"   not installed: {', '.join(missing) or 'none'}")


def _tree(root: Path, depth: int = 2, limit: int = 40) -> List[str]:
    lines: List[str] = []
    if not root.exists():
        return [f"(missing) {root}"]
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if len(rel.parts) > depth:
            continue
        size = f"  {path.stat().st_size / 1e6:.1f} MB" if path.is_file() else ""
        lines.append(f"{rel}{'/' if path.is_dir() else ''}{size}")
        if len(lines) >= limit:
            lines.append("...")
            break
    return lines or ["(empty)"]


def _seconds(path: Path) -> str:
    try:
        import soundfile
        info = soundfile.info(str(path))
        return f"{info.duration:.1f} s, {info.samplerate} Hz, {info.channels} ch"
    except Exception:
        pass
    try:
        import wave
        with wave.open(str(path), "rb") as w:
            return f"{w.getnframes() / w.getframerate():.1f} s, {w.getframerate()} Hz, {w.getnchannels()} ch"
    except Exception as exc:
        return f"unreadable ({type(exc).__name__})"


def main() -> None:
    print("Kestrel voice environment")
    print(f"  models dir : {getattr(config, 'MODELS_DIR', '(not set)')}")
    print(f"  F5-TTS dir : {config.F5_TTS_MODEL_DIR}")
    print(f"  Kokoro dir : {config.KOKORO_MODEL_DIR}")
    print(f"  ComfyUI python : {COMFY_PYTHON}  ({'found' if COMFY_PYTHON.exists() else 'NOT FOUND'})")
    print(f"  ffmpeg on PATH : {shutil.which('ffmpeg') or 'not found'}")

    _show_env("shorts venv (this Python)", _run(sys.executable))
    if COMFY_PYTHON.exists():
        _show_env("ComfyUI's Python", _run(str(COMFY_PYTHON)))

    for title, root in (("F5-TTS folder", Path(config.F5_TTS_MODEL_DIR)), ("Kokoro folder", Path(config.KOKORO_MODEL_DIR))):
        print(f"\n== {title}")
        for line in _tree(root):
            print("   " + line)

    nodes = COMFY_DIR / "custom_nodes"
    print("\n== ComfyUI custom nodes")
    print("   " + (", ".join(sorted(p.name for p in nodes.iterdir() if p.is_dir())) if nodes.exists() else f"(missing) {nodes}"))

    print("\n== Voice profiles")
    voices = Path(config.SHORTS_ROOT) / "core" / "registry" / "voices"
    for file in sorted(voices.glob("*.json")) if voices.exists() else []:
        data = json.loads(file.read_text(encoding="utf-8-sig"))
        ref = Path(config.F5_TTS_MODEL_DIR) / str(data.get("reference_audio", ""))
        print(f"   {data.get('voice_profile_id', file.stem)}: provider {data.get('provider')}, ref {data.get('reference_audio')} "
              f"-> {_seconds(ref) if ref.is_file() else 'FILE MISSING'}; ref_text {len(str(data.get('ref_text', '')))} chars"
              f"; styles {sorted((data.get('styles') or {}))}")
    if not voices.exists():
        print(f"   (missing) {voices}")


if __name__ == "__main__":
    main()