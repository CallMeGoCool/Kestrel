"""
Kestrel Shorts - ComfyUI client
===============================

A thin client: ONE job in, ONE image file out, or a clear error.

It decides nothing creative. It knows no style words, LoRA names or prompts;
those come from the Image Prompter and the Image Service. It never returns an
empty path on failure: it raises ComfyError.

    from engines import comfyui_api
    path = comfyui_api.generate(comfyui_api.ImageJob(positive=..., negative=..., output_path=...))
    comfyui_api.free()          # give the GPU back (call after the last image)

Test one image from PowerShell (ComfyUI must be running):
    python -m engines.comfyui_api "a stone bridge over a river at dawn"
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from core import config

POLL_SECONDS = 1.0


class ComfyError(RuntimeError):
    """ComfyUI could not produce the image (not running, rejected the job, timed out...)."""


@dataclass
class ImageJob:
    positive: str
    negative: str
    output_path: Path
    width: int = 768                      # SDXL-friendly size; the Image Service picks the real one
    height: int = 1344
    seed: int = 0
    checkpoint: str = ""
    loras: List[Tuple[str, float]] = field(default_factory=list)   # (file name, strength)
    steps: int = 25
    cfg: float = 6.5
    sampler_name: str = "euler_ancestral"
    scheduler: str = "normal"
    source_image: Optional[Path] = None   # set for image-to-image
    denoise: float = 1.0                  # 1.0 = a new image; lower keeps more of source_image

    def check(self) -> None:
        if not self.positive.strip():
            raise ValueError("The positive prompt is empty.")
        if not self.checkpoint:
            raise ValueError("No checkpoint given.")
        if self.width % 8 or self.height % 8 or self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive multiples of 8.")
        if not 0.0 < self.denoise <= 1.0:
            raise ValueError("denoise must be between 0 and 1.")
        if self.source_image is not None and not Path(self.source_image).is_file():
            raise ValueError(f"Source image not found: {self.source_image}")


# ----------------------------------------------------------------------
# Server helpers
# ----------------------------------------------------------------------

def _url() -> str:
    return str(getattr(config, "COMFYUI_URL", "http://127.0.0.1:8188")).rstrip("/")


def is_up() -> bool:
    try:
        return requests.get(_url() + "/system_stats", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def installed(node: str, field_name: str) -> List[str]:
    """Names ComfyUI has for a node input, e.g. installed('LoraLoader', 'lora_name')."""
    try:
        response = requests.get(f"{_url()}/object_info/{node}", timeout=10)
        response.raise_for_status()
        return list(response.json()[node]["input"]["required"][field_name][0])
    except Exception as exc:
        raise ComfyError(f"Could not read the installed {field_name} list from ComfyUI: {exc}") from exc


def free() -> bool:
    """Unload models and free GPU memory. Returns success; never raises."""
    try:
        requests.post(f"{_url()}/free", json={"unload_models": True, "free_memory": True}, timeout=15)
        return True
    except requests.RequestException as exc:
        print(f"[ComfyUI] Could not free GPU memory: {exc}")
        return False


def _upload(path: Path) -> str:
    name = f"kestrel_{uuid.uuid4().hex[:8]}_{Path(path).name}"
    with Path(path).open("rb") as file:
        response = requests.post(f"{_url()}/upload/image", files={"image": (name, file, "image/png")},
                                 data={"overwrite": "true", "type": "input"}, timeout=60)
    if response.status_code != 200:
        raise ComfyError(f"ComfyUI refused the source image: {response.text[:300]}")
    return response.json()["name"]


# ----------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------

def build_graph(job: ImageJob, source_name: Optional[str] = None) -> Dict[str, Any]:
    """The ComfyUI node graph for a job (text-to-image, or image-to-image with source_name)."""
    graph: Dict[str, Any] = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": job.checkpoint}}}
    model, clip, node = ["1", 0], ["1", 1], 20
    for lora_file, strength in job.loras:
        graph[str(node)] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": lora_file, "strength_model": strength, "strength_clip": strength,
            "model": model, "clip": clip}}
        model, clip, node = [str(node), 0], [str(node), 1], node + 1

    graph["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": job.positive, "clip": clip}}
    graph["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": job.negative, "clip": clip}}

    if source_name:
        graph["4"] = {"class_type": "LoadImage", "inputs": {"image": source_name}}
        graph["5"] = {"class_type": "ImageScale", "inputs": {
            "image": ["4", 0], "upscale_method": "lanczos", "width": job.width, "height": job.height, "crop": "center"}}
        graph["6"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}}
    else:
        graph["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": job.width, "height": job.height, "batch_size": 1}}

    graph["7"] = {"class_type": "KSampler", "inputs": {
        "seed": job.seed, "steps": job.steps, "cfg": job.cfg, "sampler_name": job.sampler_name,
        "scheduler": job.scheduler, "denoise": job.denoise if source_name else 1.0,
        "model": model, "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["6", 0]}}
    graph["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    graph["9"] = {"class_type": "PreviewImage", "inputs": {"images": ["8", 0]}}
    return graph


# ----------------------------------------------------------------------
# Generate
# ----------------------------------------------------------------------

def _error_text(status: Dict[str, Any]) -> str:
    for kind, detail in status.get("messages", []):
        if kind == "execution_error" and isinstance(detail, dict):
            return str(detail.get("exception_message", "")).strip() or "execution error"
    return "execution error"


def generate(job: ImageJob, timeout: Optional[float] = None) -> Path:
    """Make one image. Returns the saved file; raises ComfyError on any failure."""
    job.check()
    if not is_up():
        raise ComfyError(f"ComfyUI is not reachable at {_url()}. Start it and retry.")
    if job.checkpoint not in installed("CheckpointLoaderSimple", "ckpt_name"):
        raise ComfyError(f"Checkpoint not installed in ComfyUI: {job.checkpoint}")
    if job.loras:
        have = installed("LoraLoader", "lora_name")
        missing = [name for name, _ in job.loras if name not in have]
        if missing:
            raise ComfyError(f"LoRA(s) not installed in ComfyUI: {', '.join(missing)}")

    started = time.monotonic()
    source_name = _upload(Path(job.source_image)) if job.source_image else None
    response = requests.post(f"{_url()}/prompt", json={"prompt": build_graph(job, source_name),
                                                       "client_id": uuid.uuid4().hex}, timeout=30)
    if response.status_code != 200:
        raise ComfyError(f"ComfyUI rejected the job: {response.text[:500]}")
    prompt_id = response.json()["prompt_id"]

    limit = float(timeout or getattr(config, "COMFYUI_TIMEOUT", 600))   # the first image also loads the model
    while time.monotonic() - started < limit:
        history = requests.get(f"{_url()}/history/{prompt_id}", timeout=10)
        entry = history.json().get(prompt_id) if history.status_code == 200 else None
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyError(f"ComfyUI failed: {_error_text(status)}")
            for output in entry.get("outputs", {}).values():
                for image in output.get("images", []):
                    data = requests.get(f"{_url()}/view", timeout=60, params={
                        "filename": image["filename"], "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "temp")})
                    if data.status_code != 200:
                        raise ComfyError("ComfyUI made the image but it could not be fetched.")
                    path = Path(job.output_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data.content)
                    print(f"[ComfyUI] {path.name} ready in {time.monotonic() - started:.1f}s")
                    return path
            if status.get("completed"):
                raise ComfyError("ComfyUI finished but returned no image.")
        time.sleep(POLL_SECONDS)

    try:
        requests.post(f"{_url()}/interrupt", timeout=10)
    except requests.RequestException:
        pass
    raise ComfyError(f"ComfyUI took longer than {limit:.0f}s; the job was interrupted.")


# ----------------------------------------------------------------------
# One-image test
# ----------------------------------------------------------------------

def cutout(source: Path | str, output: Path | str, timeout: Optional[float] = None) -> Path:
    """Remove an image's background (ComfyUI's InspyrenetRembg node). The result is a PNG with transparency."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ComfyError(f"Image not found: {source}")
    if not is_up():
        raise ComfyError(f"ComfyUI is not reachable at {_url()}. Start it and retry.")
    started = time.monotonic()
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": _upload(source)}},
        "2": {"class_type": "InspyrenetRembg", "inputs": {"image": ["1", 0], "torchscript_jit": "default"}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
    }
    response = requests.post(f"{_url()}/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex}, timeout=30)
    if response.status_code != 200:
        raise ComfyError(f"ComfyUI rejected the cut-out job: {response.text[:500]}")
    prompt_id = response.json()["prompt_id"]
    limit = float(timeout or getattr(config, "COMFYUI_TIMEOUT", 600))
    while time.monotonic() - started < limit:
        history = requests.get(f"{_url()}/history/{prompt_id}", timeout=10)
        entry = history.json().get(prompt_id) if history.status_code == 200 else None
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyError(f"ComfyUI failed: {_error_text(status)}")
            for result in entry.get("outputs", {}).values():
                for image in result.get("images", []):
                    data = requests.get(f"{_url()}/view", timeout=60, params={
                        "filename": image["filename"], "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "temp")})
                    if data.status_code != 200:
                        raise ComfyError("ComfyUI made the cut-out but it could not be fetched.")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(data.content)
                    from PIL import Image
                    with Image.open(output) as check:
                        if "A" not in check.getbands():
                            raise ComfyError("The cut-out has no transparency channel; "
                                             "the node's MASK output has to be used instead.")
                    print(f"[ComfyUI] {output.name} ready in {time.monotonic() - started:.1f}s")
                    return output
            if status.get("completed"):
                raise ComfyError("ComfyUI finished but returned no image.")
        time.sleep(POLL_SECONDS)
    try:
        requests.post(f"{_url()}/interrupt", timeout=10)
    except requests.RequestException:
        pass
    raise ComfyError(f"ComfyUI took longer than {limit:.0f}s; the cut-out was interrupted.")


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate one test image.")
    parser.add_argument("prompt", nargs="?", default="")
    parser.add_argument("--negative", default="text, letters, watermark, blurry, low quality")
    parser.add_argument("--out", default=str(Path(config.SHORTS_TEMP_DIR) / "comfy_test.png"))
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1344)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint", default="", help="default: the first checkpoint ComfyUI has")
    parser.add_argument("--lora", action="append", default=[], help="file.safetensors=0.75 (repeatable)")
    parser.add_argument("--source", help="an image file, for image-to-image")
    parser.add_argument("--denoise", type=float, default=0.55)
    parser.add_argument("--keep-loaded", action="store_true", help="do not free GPU memory afterwards")
    parser.add_argument("--cutout", help="remove the background of this image file instead of generating")
    args = parser.parse_args()

    try:
        from core import llm        # the GPU is shared: let Ollama let go of it first
        print("[ComfyUI] Ollama unloaded:", llm.unload_model())
    except Exception as exc:
        print("[ComfyUI] Could not ask Ollama to unload:", exc)

    if args.cutout:
        target = Path(args.cutout)
        try:
            print("Saved:", cutout(target, target.with_name(target.stem + "_cutout.png")))
        except ComfyError as exc:
            raise SystemExit(f"FAILED: {exc}")
        finally:
            if not args.keep_loaded:
                free()
        return

    loras = []
    for item in args.lora:
        name, _, strength = item.partition("=")
        loras.append((name.strip(), float(strength or 0.8)))
    checkpoint = args.checkpoint or installed("CheckpointLoaderSimple", "ckpt_name")[0]
    job = ImageJob(positive=args.prompt, negative=args.negative, output_path=Path(args.out),
                   width=args.width, height=args.height, seed=args.seed, checkpoint=checkpoint, loras=loras,
                   source_image=Path(args.source) if args.source else None,
                   denoise=args.denoise if args.source else 1.0)
    try:
        print("Saved:", generate(job))
    except (ComfyError, ValueError) as exc:
        raise SystemExit(f"FAILED: {exc}")
    finally:
        if not args.keep_loaded:
            free()


if __name__ == "__main__":
    _main()