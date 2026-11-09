"""
Kestrel Shorts - Shared LLM Client
==================================

The one place that talks to Ollama. Agents call:

    from core import llm
    data = llm.generate_json("story_director", prompt, system, schema=SCHEMA)
    text = llm.generate("publisher", prompt).text

What it handles, so agents don't have to:
  - Per-stage settings (num_ctx, num_predict, temperature) from
    config.LLM_STAGES, with per-call overrides.
  - Streaming with an IDLE timeout: only a long silence fails, not a long
    but healthy generation.
  - Schema-constrained JSON (`schema=`), or plain JSON mode.
  - Retry with the error fed back to the model; larger output limit when
    the reply was cut off.
  - Token/speed diagnostics, with warnings for a probably-truncated prompt
    or an output that hit num_predict.
  - unload_model() so ComfyUI can use the GPU.

Run `python -m core.llm` for a quick health check (speed + GPU offload).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from core import config

logger = logging.getLogger(__name__)

_DEFAULTS = {"num_ctx": 8192, "num_predict": 2000, "temperature": 0.3}
_MAX_NUM_PREDICT = 8192

LAST_RESULT: Optional["LLMResult"] = None


class LLMError(RuntimeError):
    """
    kind: connection | http | server | timeout | incomplete | empty | invalid_json
    `partial` holds any text received before a failure.
    """

    def __init__(self, message: str, kind: str = "error", partial: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.partial = partial


@dataclass
class LLMResult:
    text: str
    stage: str
    model: str
    num_ctx: int
    num_predict: int
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    elapsed: float = 0.0
    load_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    done_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        """Output stopped because it hit num_predict."""
        return self.done_reason == "length"

    @property
    def prompt_probably_cut(self) -> bool:
        """Prompt filled the window, so Ollama likely dropped its start."""
        return bool(self.prompt_tokens and self.prompt_tokens >= 0.95 * self.num_ctx)


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

def _base_url() -> str:
    return str(getattr(config, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")


def _stage_settings(stage: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    settings = dict(_DEFAULTS)
    table = getattr(config, "LLM_STAGES", {})
    if stage in table:
        settings.update(table[stage])
    else:
        logger.warning("No LLM_STAGES entry for %r; using defaults.", stage)
    settings.update({k: v for k, v in overrides.items() if v is not None})
    settings["model"] = settings.get("model") or getattr(config, "OLLAMA_MODEL", "qwen3:8b")
    return settings


def _say(line: str) -> None:
    if getattr(config, "LLM_VERBOSE", True):
        print(line)
    logger.info(line)


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------

def extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object from model text (tolerates think blocks, fences, chatter)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("The model returned an empty reply.")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("The reply contained no JSON object.")
        try:
            data, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"The reply was not valid JSON ({exc.msg} at char {exc.pos}).") from exc

    if not isinstance(data, dict):
        raise ValueError("The reply must be a JSON object.")
    return data


# ----------------------------------------------------------------------
# Core call
# ----------------------------------------------------------------------

def generate(
    stage: str,
    prompt: str,
    system: str = "",
    *,
    schema: Optional[Dict[str, Any]] = None,
    json_mode: bool = False,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    idle_timeout: Optional[float] = None,
    keep_alive: Optional[str] = None,
) -> LLMResult:
    """One streamed call to Ollama. Raises LLMError on failure."""
    global LAST_RESULT

    s = _stage_settings(stage, {"num_ctx": num_ctx, "num_predict": num_predict,
                                "temperature": temperature, "model": model})
    idle = float(idle_timeout or getattr(config, "LLM_IDLE_TIMEOUT", 120))

    estimated = (len(prompt) + len(system)) / 3.5
    if estimated + s["num_predict"] > s["num_ctx"]:
        logger.warning(
            "[LLM %s] prompt (~%d tokens) + num_predict (%d) may exceed num_ctx (%d).",
            stage, estimated, s["num_predict"], s["num_ctx"],
        )

    payload: Dict[str, Any] = {
        "model": s["model"],
        "prompt": prompt,
        "system": system,
        "stream": True,
        "think": False,
        "keep_alive": keep_alive or getattr(config, "LLM_KEEP_ALIVE", "30m"),
        "options": {
            "temperature": s["temperature"],
            "num_ctx": s["num_ctx"],
            "num_predict": s["num_predict"],
        },
    }
    if schema is not None:
        payload["format"] = schema
    elif json_mode:
        payload["format"] = "json"

    parts: List[str] = []
    final: Dict[str, Any] = {}
    started = last_chunk = time.monotonic()

    try:
        with requests.post(
            f"{_base_url()}/api/generate", json=payload, stream=True, timeout=(10, idle)
        ) as response:
            if response.status_code != 200:
                raise LLMError(f"Ollama HTTP {response.status_code}: {response.text[:300]}", "http")
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_chunk = time.monotonic()
                if "error" in chunk:
                    raise LLMError(f"Ollama error: {chunk['error']}", "server", "".join(parts))
                parts.append(chunk.get("response", ""))
                if chunk.get("done"):
                    final = chunk
                    break
    except LLMError:
        raise
    except requests.ReadTimeout as exc:
        raise LLMError(f"No output from Ollama for {idle:.0f}s.", "timeout", "".join(parts)) from exc
    except requests.ConnectionError as exc:
        # requests reports a read timeout while streaming as a ConnectionError.
        if "timed out" in str(exc).lower() or time.monotonic() - last_chunk >= idle - 1:
            raise LLMError(f"No output from Ollama for {idle:.0f}s.", "timeout", "".join(parts)) from exc
        raise LLMError(f"Cannot reach Ollama at {_base_url()}. Is it running?", "connection") from exc

    text = "".join(parts).strip()
    if not final:
        raise LLMError("Ollama's stream ended before completion.", "incomplete", text)
    if not text:
        raise LLMError("Ollama returned an empty reply.", "empty")

    def seconds(ns: Any) -> Optional[float]:
        return ns / 1e9 if isinstance(ns, (int, float)) and ns >= 0 else None

    eval_s = seconds(final.get("eval_duration"))
    out_tokens = final.get("eval_count")
    result = LLMResult(
        text=text,
        stage=stage,
        model=s["model"],
        num_ctx=s["num_ctx"],
        num_predict=s["num_predict"],
        prompt_tokens=final.get("prompt_eval_count"),
        output_tokens=out_tokens,
        elapsed=time.monotonic() - started,
        load_seconds=seconds(final.get("load_duration")),
        tokens_per_second=(out_tokens / eval_s) if out_tokens and eval_s else None,
        done_reason=final.get("done_reason"),
    )
    LAST_RESULT = result

    tps = f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "?"
    _say(
        f"[LLM {stage}] prompt={result.prompt_tokens}tok out={result.output_tokens}tok "
        f"{tps}tok/s ctx={s['num_ctx']} total={result.elapsed:.1f}s"
    )
    if result.prompt_probably_cut:
        _say(f"[LLM {stage}] WARNING: prompt filled the context window; its start was probably cut. Shrink the prompt or raise num_ctx.")
    if result.truncated:
        _say(f"[LLM {stage}] WARNING: reply hit num_predict={s['num_predict']} and was cut off.")
    return result


def generate_json(
    stage: str,
    prompt: str,
    system: str = "",
    *,
    schema: Optional[Dict[str, Any]] = None,
    validate: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    retries: Optional[int] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Call the model and return a JSON object.

    `validate(data)` may repair and return the data, or raise ValueError
    describing what is wrong; the description is sent back to the model for
    the next attempt. Connection and server errors are not retried. The
    last call's diagnostics are in `llm.LAST_RESULT`.
    """
    attempts = 1 + (getattr(config, "LLM_MAX_RETRIES", 2) if retries is None else retries)
    overrides = dict(overrides)
    current_prompt = prompt

    for attempt in range(1, attempts + 1):
        try:
            result = generate(stage, current_prompt, system, schema=schema, json_mode=True, **overrides)
        except LLMError as exc:
            if exc.kind in ("connection", "http", "server") or attempt == attempts:
                raise
            _say(f"[LLM {stage}] attempt {attempt}/{attempts} failed ({exc.kind}); retrying.")
            continue

        try:
            data = extract_json(result.text)
            return validate(data) if validate else data
        except ValueError as exc:
            if attempt == attempts:
                raise LLMError(
                    f"[{stage}] no valid JSON after {attempts} attempts: {exc}",
                    "invalid_json", result.text,
                ) from exc
            _say(f"[LLM {stage}] attempt {attempt}/{attempts} rejected: {exc}")
            current_prompt = (
                f"{prompt}\n\nYour previous reply was rejected: {exc}\n"
                "Return the complete, corrected JSON object only."
            )
            if result.truncated:
                overrides["num_predict"] = min(int(result.num_predict * 1.5), _MAX_NUM_PREDICT)

    raise LLMError("Unreachable.", "error")  # pragma: no cover


# ----------------------------------------------------------------------
# Model management and diagnostics
# ----------------------------------------------------------------------

def loaded_models() -> List[Dict[str, Any]]:
    """Models Ollama currently holds in memory, with GPU/CPU split."""
    try:
        response = requests.get(f"{_base_url()}/api/ps", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as exc:
        logger.warning("Could not query Ollama /api/ps: %s", exc)
        return []
    report = []
    for m in models:
        size, vram = m.get("size") or 0, m.get("size_vram") or 0
        report.append({
            "model": m.get("name"),
            "size_gb": round(size / 1e9, 2),
            "vram_percent": round(100 * vram / size) if size else None,
        })
    return report


def unload_model(model: Optional[str] = None) -> bool:
    """Free the model's memory (call before ComfyUI runs). Returns success."""
    name = model or getattr(config, "OLLAMA_MODEL", "qwen3:8b")
    try:
        response = requests.post(
            f"{_base_url()}/api/generate",
            json={"model": name, "keep_alive": 0},
            timeout=30,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Could not unload %s: %s", name, exc)
        return False


if __name__ == "__main__":
    print("Kestrel LLM health check")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    try:
        data = generate_json("orchestrator", 'Reply with the JSON {"answer": "ok"}.', schema=schema)
        print("Reply:", data)
    except LLMError as exc:
        raise SystemExit(f"FAILED ({exc.kind}): {exc}")
    for m in loaded_models():
        print(f"Loaded: {m['model']} {m['size_gb']} GB, {m['vram_percent']}% on GPU")