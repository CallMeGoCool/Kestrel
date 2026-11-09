"""
Kestrel Shorts - Word timings
=============================

When does each word of a spoken sentence start and end? The script's words are known; faster-whisper (CPU,
small, no torch) listens to the finished audio and reports the words it heard with times; the script's words
are matched to those, and any word the recogniser got wrong or missed is placed between its neighbours in
proportion to its length. Without faster-whisper, or when almost nothing matches, all words are spread over
the sentence in proportion to their length and the result says so (method "estimated").

    words = align(text, samples_24k, duration)  ->  {"words": [{text, start, end}], "method", "matched"}
"""
from __future__ import annotations

import importlib.util
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from core import config

_MODEL: Any = None
WEAK_MATCH = 0.5            # below this share of matched words the result counts as estimated


def tokens(text: str) -> List[str]:
    """The words of a sentence as they will be shown (punctuation kept; dashes separate words)."""
    padded = text.replace("\u2014", " \u2014 ").replace("\u2013", " \u2013 ")
    return [t for t in padded.split() if re.search(r"\w", t)]


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _model() -> Any:
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from faster_whisper import WhisperModel
        root = Path(getattr(config, "MODELS_DIR", ".")) / "Whisper"
        _MODEL = WhisperModel(getattr(config, "WHISPER_MODEL", "base.en"), device="cpu", compute_type="int8",
                              download_root=str(root))
    return _MODEL


def heard_words(samples: np.ndarray, rate: int) -> List[Tuple[str, float, float]]:
    audio = resample_poly(samples, 16000, rate).astype("float32") if rate != 16000 else samples.astype("float32")
    segments, _ = _model().transcribe(audio, language="en", word_timestamps=True, beam_size=1,
                                      condition_on_previous_text=False, vad_filter=False)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segments for w in (s.words or [])]


def transcribe(samples: np.ndarray, rate: int) -> str:
    """The plain text of a clip (used once, to write the reference text of a voice)."""
    audio = resample_poly(samples, 16000, rate).astype("float32") if rate != 16000 else samples.astype("float32")
    segments, _ = _model().transcribe(audio, language="en", beam_size=5, vad_filter=False)
    return " ".join(s.text.strip() for s in segments).strip()


def map_words(words: List[str], heard: List[Tuple[str, float, float]], duration: float) -> Tuple[List[Dict[str, Any]], float]:
    """Script words with times, and the share of them that were matched to heard words."""
    n = len(words)
    if n == 0:
        return [], 0.0
    a = [norm(w) for w in words]
    b = [norm(h[0]) for h in heard]
    times: List[Optional[Tuple[float, float]]] = [None] * n
    matched = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = (heard[j1 + k][1], heard[j1 + k][2])
                matched += 1
    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        left = times[i - 1][1] if i > 0 else 0.0
        right = max(times[j][0] if j < n else duration, left)
        weights = [max(1, len(a[k])) for k in range(i, j)]
        total, cursor = float(sum(weights)), left
        for k, w in zip(range(i, j), weights):
            span = (right - left) * w / total
            times[k] = (cursor, cursor + span)
            cursor += span
        i = j
    return ([{"text": words[k], "start": round(times[k][0], 3), "end": round(times[k][1], 3)} for k in range(n)],
            matched / n)


def align(text: str, samples: np.ndarray, rate: int, use_whisper: bool = True) -> Dict[str, Any]:
    duration = len(samples) / rate
    words = tokens(text)
    heard: List[Tuple[str, float, float]] = []
    error = ""
    if use_whisper and available():
        try:
            heard = heard_words(samples, rate)
        except Exception as exc:          # a failed recogniser must not stop the video
            error = f"{type(exc).__name__}: {exc}"
    mapped, share = map_words(words, heard, duration)
    method = "whisper" if heard and share >= WEAK_MATCH else "estimated"
    return {"words": mapped, "method": method, "matched": round(share, 2), "error": error}