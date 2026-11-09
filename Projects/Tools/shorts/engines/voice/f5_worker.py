"""
Kestrel Shorts - F5-TTS worker
==============================

Runs inside the Python that has F5-TTS and torch (the one that ships with ComfyUI), so the shorts venv needs
no second torch. It knows nothing about Kestrel: it reads one job file, loads the model ONCE, speaks every
job and reports each result on a line starting with @@RESULT, so the caller can follow along.

    <ComfyUI python> f5_worker.py <job.json>

Job file:  {"model_dir": ".../F5-TTS", "model": "" or "F5TTS_v1_Base", "nfe_step": 32, "ffmpeg_dll_dir": "...",
            "jobs": [{"id", "text", "ref_audio", "ref_text", "speed", "seed", "out"}]}
ffmpeg_dll_dir is the folder of the SHARED FFmpeg build (avcodec-*.dll, avutil-*.dll, ...) that TorchCodec needs to
decode audio on Windows; a static ffmpeg.exe is not enough. Torch must also see it in THIS process, so it is
registered here, before torch and f5_tts are imported.
Offline by design: models come from the Hugging Face cache inside model_dir/hub, nothing is downloaded.
A job without ref_text fails (F5 would otherwise download a Whisper model to transcribe the clip).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

_DLL_HANDLE = None          # keep a reference: Windows drops the DLL folder when the handle is collected


def _register_ffmpeg(folder: str) -> str:
    """Make the shared FFmpeg DLLs loadable in this process. Returns a warning, or "" when all is well."""
    global _DLL_HANDLE
    if not folder:
        return ""
    path = Path(folder)
    if not path.is_dir():
        return f"FFmpeg shared DLL folder not found: {path}"
    if hasattr(os, "add_dll_directory"):
        _DLL_HANDLE = os.add_dll_directory(str(path))
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
    return ""


def _say(kind: str, payload: dict) -> None:
    print(f"@@{kind} " + json.dumps(payload, ensure_ascii=False), flush=True)


def _pick_model(hub: Path, wanted: str) -> str:
    """The F5 checkpoint that exists in the local cache (v1 base first), unless one is asked for."""
    if wanted:
        return wanted
    found = {p.parent.name for p in hub.glob("models--SWivid--F5-TTS/snapshots/*/*/model_*.*")}
    for name in ("F5TTS_v1_Base", "F5TTS_Base"):
        if name in found:
            return name
    return "F5TTS_v1_Base"


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    model_dir = Path(job["model_dir"])
    hub = model_dir / "hub"
    os.environ["HF_HOME"] = str(model_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    started = time.monotonic()
    warning = _register_ffmpeg(str(job.get("ffmpeg_dll_dir", "")))
    if warning:
        _say("LOG", {"message": warning})
    try:
        import soundfile
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        from f5_tts.api import F5TTS
        name = _pick_model(hub, str(job.get("model", "")))
        _say("LOG", {"message": f"loading {name} on {device}"})
        tts = F5TTS(model=name, device=device, hf_cache_dir=str(hub))
        _say("LOG", {"message": f"model ready in {time.monotonic() - started:.0f} s"})
    except Exception as exc:
        _say("FATAL", {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1200:]})
        return 2

    for item in job["jobs"]:
        begun = time.monotonic()
        try:
            if not str(item.get("ref_text", "")).strip():
                raise ValueError("ref_text is empty for this voice (it would make F5 download a Whisper model)")
            if not Path(item["ref_audio"]).is_file():
                raise FileNotFoundError(f"reference clip not found: {item['ref_audio']}")
            wav, rate, _ = tts.infer(ref_file=item["ref_audio"], ref_text=item["ref_text"], gen_text=item["text"],
                                     nfe_step=int(job.get("nfe_step", 32)), speed=float(item.get("speed", 1.0)),
                                     seed=item.get("seed"), remove_silence=False, show_info=lambda *a, **k: None)
            Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(item["out"], wav, rate)
            _say("RESULT", {"id": item["id"], "ok": True, "seconds": round(time.monotonic() - begun, 1)})
        except Exception as exc:
            _say("RESULT", {"id": item["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    _say("DONE", {"seconds": round(time.monotonic() - started, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())