"""
Kestrel Shorts - Voice engine
=============================

Speaks the Voice Director's plan. For every sentence (unit) of the script it makes one audio file, joins them
with the planned pauses into the voice track, finds the time of every word, and writes `voice_timing`: the
real clock the Frame Supervisor cuts the visuals to (replacing the estimates).

  Provider    F5-TTS is the voice. It runs in a separate process under the Python that has F5-TTS and torch (the
              one that ships with ComfyUI), see f5_worker.py. Kokoro (CPU, in this venv) is only the fallback:
              when F5 cannot start or a sentence fails, so a run still ends with a video. A voice profile can name
              another provider later; each provider is one function that turns jobs into wav files.
  Tone        a profile may have `styles`: {"curious": {"reference_audio": "...", "ref_text": "..."}}; a tone
              without a clip is spoken in the plain voice (the plan already lists these as voice wishes).
  Cache       every sentence is cached under runs/<run>/audio/units, named by a signature of everything that
              shapes it (text, voice, speed, seed, tone clip), so a re-run only speaks what changed. It stays until
              the final stitch (--clear-cache removes it). `--set` changes one sentence (speed, seed, tone) and
              the next --go speaks only that one: the page's voice tuning will write the same overrides file.
  Loudness    every sentence is trimmed of silence at both ends and brought to the same level.

    python -m engines.voice.voice_engine --from-run <shared_context.json>              preview (no GPU)
    python -m engines.voice.voice_engine --from-run <...> --go                         speak, join, align
    python -m engines.voice.voice_engine --from-run <...> --set "beat_2a.1 speed=0.95 seed=7" --go
    python -m engines.voice.voice_engine --selftest f5|kokoro                          one test sentence
    python -m engines.voice.voice_engine --transcribe-refs                             write missing ref_text
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from core import config
from creative.catalogs import asset_catalog as ac
from engines.audio import audio_rules as ar
from engines.voice import alignment

SAMPLE_RATE = 24000
TARGET_RMS_DB, PEAK_LIMIT_DB = -20.0, -1.0       # every sentence is brought to this level
SPEAKER_GAP = 0.35                                # when the next sentence is another character's: a natural turn-taking gap
END_TAIL = 0.8                                    # silence after the last word: the video never ends on a cut-off sound
POST_VERSION = 2                                  # bump when trimming or levelling changes: cached sentences are made again
FFMPEG_DLL_DIR = Path(getattr(config, "FFMPEG_SHARED_DIR", r"D:\AI Tools\FFmpeg_Shared\bin"))   # shared FFmpeg DLLs for TorchCodec
MIN_GAP = 0.15                                    # seconds between sentences even when the plan asks for no pause
FADE_IN, FADE_OUT = 1.0, 1.5                      # music and ambience beds
SPEED_RANGE = (0.6, 1.6)
WORKER = Path(__file__).with_name("f5_worker.py")
COMFY_PYTHON = Path(getattr(config, "COMFYUI_PYTHON", r"D:\AI Tools\ComfyUI_windows_portable\python_embeded\python.exe"))
F5_NAMES = ("f5", "f5-tts", "f5tts")
KOKORO_DEFAULTS = {"female": "af_sarah", "male": "am_adam"}
SELFTEST_TEXT = "This is a short test of the voice engine. How does it sound?"


class VoiceError(RuntimeError):
    pass


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _voices_dir() -> Path:
    return Path(config.SHORTS_ROOT) / "core" / "registry" / "voices"


def _clamp(value: float) -> float:
    return max(SPEED_RANGE[0], min(SPEED_RANGE[1], float(value)))


def _known_ref_text(reference_audio: str) -> str:
    """The transcript of a reference clip, if the F5 folder's own voices_config.json has one for it."""
    path = Path(config.F5_TTS_MODEL_DIR) / "ref_voices" / "voices_config.json"
    if not reference_audio or not path.is_file():
        return ""
    try:
        data = _read_json(path)
    except Exception:
        return ""
    name = Path(reference_audio).name.lower()
    stem = Path(reference_audio).stem.lower()
    keys = ("ref_text", "text", "transcript", "reference_text", "prompt_text")

    def matches(s: str) -> bool:
        return re.split(r"[\\/]", s.lower())[-1] in (name, stem)

    def text_of(d: Dict[str, Any]) -> str:
        return next((v.strip() for k in keys for v in [d.get(k)] if isinstance(v, str) and v.strip()), "")

    def walk(node: Any, label: str = "") -> str:
        if isinstance(node, dict):
            if (label and matches(label)) or any(isinstance(v, str) and matches(v) for v in node.values()):
                found = text_of(node)
                if found:
                    return found
            for k, v in node.items():
                if isinstance(v, str) and matches(k) and len(v.split()) > 2:
                    return v.strip()
                found = walk(v, k)
                if found:
                    return found
        elif isinstance(node, list):
            for v in node:
                found = walk(v, label)
                if found:
                    return found
        return ""

    return walk(data)


def _signature(parts: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]


# ----------------------------------------------------------------------
# Jobs: one per sentence
# ----------------------------------------------------------------------

def _cast_voice(registry: Any, channel_id: str, speaker: str) -> Dict[str, Any]:
    cast = [m["id"] for m in registry.character_menu(channel_id)]
    cid = speaker if speaker in cast else (cast[0] if cast else "")
    if not cid:
        raise VoiceError("The channel has no characters, so there is no voice to speak with.")
    voice_id = registry.get_character(cid).get("voice_profile_id")
    voice = dict(registry.get_voice(voice_id) or {})
    voice.setdefault("voice_profile_id", voice_id)
    return voice


def _reference(voice: Dict[str, Any], tone: str) -> Tuple[str, str, str]:
    """(reference clip, its text, the tone actually used): a tone clip when the profile has one."""
    style = (voice.get("styles") or {}).get(tone) if tone != "neutral" else None
    if isinstance(style, dict) and style.get("reference_audio"):
        clip = style["reference_audio"]
        return clip, style.get("ref_text") or _known_ref_text(clip), tone
    clip = voice.get("reference_audio", "")
    return clip, voice.get("ref_text") or _known_ref_text(clip), "neutral"


def _provider_of(voice: Dict[str, Any], forced: str = "") -> str:
    name = (forced or str(voice.get("provider", "F5-TTS"))).lower()
    return "f5" if name in F5_NAMES else name


def _kokoro_voice(voice: Dict[str, Any]) -> str:
    if voice.get("kokoro_voice"):
        return str(voice["kokoro_voice"])
    return KOKORO_DEFAULTS["female" if "female" in str(voice.get("reference_audio", "")).lower() else "male"]


def build_jobs(plan: Dict[str, Any], script: Any, registry: Any, channel_id: str, overrides: Dict[str, Dict[str, Any]],
               audio_dir: Path, forced_provider: str = "") -> List[Dict[str, Any]]:
    units = {u["unit_id"]: u for u in ar.split_units(script.sections)}
    jobs: List[Dict[str, Any]] = []
    for line in plan["lines"]:
        uid = line["unit_id"]
        text = units[uid]["text"]
        ov = overrides.get(uid, {})
        voice = _cast_voice(registry, channel_id, line["speaker"])
        tone = ov.get("tone", line["tone"])
        clip, ref_text, tone_used = _reference(voice, tone)
        speed = _clamp(ov["speed"]) if "speed" in ov else _clamp(line["speed"] * float(voice.get("speed", 1.0)))
        seed = int(ov["seed"]) if "seed" in ov else int(hashlib.sha1(f"{uid}|{text}".encode()).hexdigest()[:6], 16) % 100000
        provider = _provider_of(voice, forced_provider)
        parts = {"text": text, "provider": provider, "clip": clip, "ref_text": ref_text, "speed": round(speed, 3),
                 "seed": seed, "tone": tone_used, "post": POST_VERSION, "kokoro": _kokoro_voice(voice) if provider == "kokoro" else ""}
        sig = _signature(parts)
        jobs.append({**parts, "unit_id": uid, "voice_profile_id": voice.get("voice_profile_id"), "signature": sig,
                     "path": audio_dir / "units" / f"{uid}_{sig}.wav", "raw": audio_dir / "raw" / f"{uid}_{sig}.wav",
                     "tone_wanted": tone, "kokoro_voice": _kokoro_voice(voice)})
    return jobs


def _sidecar(job: Dict[str, Any]) -> Path:
    return job["path"].with_suffix(".json")


def is_done(job: Dict[str, Any]) -> bool:
    return job["path"].exists() and _sidecar(job).exists()


# ----------------------------------------------------------------------
# Providers: jobs in, raw wav files out; each returns {unit_id: error} for the ones that failed
# ----------------------------------------------------------------------

def run_f5(jobs: List[Dict[str, Any]], work_dir: Path) -> Dict[str, str]:
    if not jobs:
        return {}
    if not COMFY_PYTHON.exists():
        return {j["unit_id"]: f"F5 python not found: {COMFY_PYTHON}" for j in jobs}
    items = []
    for j in jobs:
        ref = Path(config.F5_TTS_MODEL_DIR) / j["clip"]
        items.append({"id": j["unit_id"], "text": j["text"], "ref_audio": str(ref), "ref_text": j["ref_text"],
                      "speed": j["speed"], "seed": j["seed"], "out": str(j["raw"])})
    work_dir.mkdir(parents=True, exist_ok=True)
    job_file = work_dir / "f5_job.json"
    job_file.write_text(json.dumps({"model_dir": str(config.F5_TTS_MODEL_DIR), "model": getattr(config, "F5_MODEL", ""),
                                    "nfe_step": getattr(config, "F5_NFE_STEPS", 32), "ffmpeg_dll_dir": str(FFMPEG_DLL_DIR),
                                    "jobs": items}), encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    failed: Dict[str, str] = {}
    seen: set = set()
    fatal = ""
    try:
        proc = subprocess.Popen([str(COMFY_PYTHON), str(WORKER), str(job_file)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env,
                                cwd=str(WORKER.parent))
    except OSError as exc:
        return {j["unit_id"]: f"could not start F5: {exc}" for j in jobs}
    tail: List[str] = []
    for line in proc.stdout:                                    # type: ignore[union-attr]
        line = line.rstrip()
        if line.startswith("@@RESULT "):
            r = json.loads(line[9:])
            seen.add(r["id"])
            if r.get("ok"):
                print(f"[Voice] F5 {r['id']} spoken in {r['seconds']} s")
            else:
                failed[r["id"]] = r.get("error", "unknown error")
                print(f"[Voice] F5 {r['id']} FAILED: {failed[r['id']]}")
        elif line.startswith("@@FATAL "):
            fatal = json.loads(line[8:]).get("error", "F5 could not start")
            print(f"[Voice] F5 could not start: {fatal}")
        elif line.startswith("@@LOG "):
            print(f"[Voice] F5 {json.loads(line[6:])['message']}")
        elif line.strip():
            tail.append(line)
            tail = tail[-6:]
    proc.wait()
    for j in jobs:
        if j["unit_id"] not in seen and j["unit_id"] not in failed:
            failed[j["unit_id"]] = fatal or ("F5 stopped early: " + " | ".join(tail[-2:]))[:300]
    return failed


def run_kokoro(jobs: List[Dict[str, Any]]) -> Dict[str, str]:
    if not jobs:
        return {}
    try:
        from kokoro_onnx import Kokoro
        folder = Path(config.KOKORO_MODEL_DIR)
        model = next(iter(sorted(folder.glob("*.onnx"))))
        voices = next(iter(sorted(folder.glob("voices*"))))
        kokoro = Kokoro(str(model), str(voices))
    except Exception as exc:
        return {j["unit_id"]: f"Kokoro could not load ({type(exc).__name__}: {exc})" for j in jobs}
    failed: Dict[str, str] = {}
    for j in jobs:
        try:
            samples, rate = kokoro.create(j["text"], voice=j["kokoro_voice"], speed=max(0.5, min(2.0, j["speed"])), lang="en-us")
            j["raw"].parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(j["raw"]), samples, rate)
            print(f"[Voice] Kokoro {j['unit_id']} spoken ({j['kokoro_voice']})")
        except Exception as exc:
            failed[j["unit_id"]] = f"{type(exc).__name__}: {exc}"
    return failed


PROVIDERS: Dict[str, Callable[..., Dict[str, str]]] = {"kokoro": run_kokoro}     # F5 is called with its work folder


# ----------------------------------------------------------------------
# Audio: trim, level, join
# ----------------------------------------------------------------------

def _frames_db(audio: np.ndarray, rate: int, size: float = 0.02) -> np.ndarray:
    n = max(1, int(rate * size))
    count = len(audio) // n
    if count == 0:
        return np.array([-120.0])
    frames = audio[: count * n].reshape(count, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def _trim(audio: np.ndarray, rate: int) -> np.ndarray:
    """Cut silence at both ends (keeping a little air): F5 and Kokoro both add some."""
    db = _frames_db(audio, rate)
    live = np.where(db > max(db.max() - 50.0, -65.0))[0]
    if len(live) == 0:
        return audio
    n = int(rate * 0.02)
    start = max(0, live[0] * n - int(rate * 0.04))
    end = min(len(audio), (live[-1] + 1) * n + int(rate * 0.25))
    return audio[start:end]


def _level(audio: np.ndarray, rate: int) -> np.ndarray:
    """Same loudness for every sentence: RMS of the speech (not the silence) to TARGET_RMS_DB, peaks limited."""
    db = _frames_db(audio, rate)
    live = db > db.max() - 30.0
    n = int(rate * 0.02)
    speech = audio[: len(db) * n].reshape(len(db), n)[live]
    rms = float(np.sqrt((speech ** 2).mean())) if speech.size else float(np.sqrt((audio ** 2).mean()) + 1e-9)
    gain = 10 ** (TARGET_RMS_DB / 20) / max(rms, 1e-9)
    out = audio * gain
    peak = float(np.abs(out).max())
    limit = 10 ** (PEAK_LIMIT_DB / 20)
    return out * (limit / peak) if peak > limit else out


def _read_mono(path: Path) -> Tuple[np.ndarray, int]:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, rate


def postprocess(raw: Path, out: Path) -> float:
    audio, rate = _read_mono(raw)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype("float32")
    audio = _level(_trim(audio, SAMPLE_RATE), SAMPLE_RATE)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
    return len(audio) / SAMPLE_RATE


# ----------------------------------------------------------------------
# Speak what is missing
# ----------------------------------------------------------------------

def speak(jobs: List[Dict[str, Any]], audio_dir: Path, allow_fallback: bool = True) -> Dict[str, str]:
    """Make every missing sentence. Returns the sentences that could not be made at all, with the reason."""
    todo = [j for j in jobs if not is_done(j)]
    if not todo:
        return {}
    for j in todo:                                  # a missing reference text is a setup problem, not a fallback case
        if j["provider"] == "f5" and not j["ref_text"].strip():
            raise VoiceError(f"The voice {j['voice_profile_id']!r} has no ref_text (the words spoken in its reference clip), "
                             "and the F5 folder's voices_config.json has none for it either. Run "
                             "`python -m engines.voice.voice_engine --transcribe-refs` or type the words into the voice file.")
    try:
        from core import llm
        print("[Voice] Ollama unloaded:", llm.unload_model())
    except Exception as exc:
        print("[Voice] Could not ask Ollama to unload:", exc)
    try:
        from engines.image import comfyui_api
        comfyui_api.free()
    except Exception:
        pass

    started = time.monotonic()
    failures: Dict[str, str] = {}
    by_id = {j["unit_id"]: j for j in todo}
    first = run_f5([j for j in todo if j["provider"] == "f5"], audio_dir / "work")
    other: List[Dict[str, Any]] = [j for j in todo if j["provider"] not in ("f5", "kokoro")]
    for j in other:
        failures[j["unit_id"]] = f"unknown provider {j['provider']!r}"
    retry: List[Dict[str, Any]] = []
    for uid, reason in first.items():
        if allow_fallback:
            print(f"[Voice] {uid}: falling back to Kokoro ({reason})")
            by_id[uid]["fallback"] = reason
            retry.append(by_id[uid])
        else:
            failures[uid] = reason
    second = run_kokoro([j for j in todo if j["provider"] == "kokoro"] + retry)
    for uid, reason in second.items():
        failures[uid] = (first.get(uid, "") + " | " if uid in first else "") + reason

    for j in todo:
        if j["unit_id"] in failures or not j["raw"].exists():
            failures.setdefault(j["unit_id"], "no audio was produced")
            continue
        seconds = postprocess(j["raw"], j["path"])
        j["raw"].unlink(missing_ok=True)
        _sidecar(j).write_text(json.dumps({
            "unit_id": j["unit_id"], "signature": j["signature"], "provider_used": "kokoro (fallback)" if "fallback" in j
            else j["provider"], "fallback_reason": j.get("fallback", ""), "tone_used": j["tone"], "speed": j["speed"],
            "seed": j["seed"], "seconds": round(seconds, 2)}, indent=2), encoding="utf-8")
        for old in j["path"].parent.glob(f"{j['unit_id']}_*.wav"):      # keep only the current version on disk
            if old != j["path"]:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
    print(f"[Voice] {len(todo) - len(failures)}/{len(todo)} sentence(s) made in {time.monotonic() - started:.0f} s")
    return failures


# ----------------------------------------------------------------------
# Join, align, write the timeline
# ----------------------------------------------------------------------

def assemble(plan: Dict[str, Any], jobs: List[Dict[str, Any]], script: Any, audio_dir: Path, align_words: bool = True,
             realign: bool = False) -> Dict[str, Any]:
    units = {u["unit_id"]: u for u in ar.split_units(script.sections)}
    by_id = {j["unit_id"]: j for j in jobs}
    pieces: List[np.ndarray] = []
    timeline: List[Dict[str, Any]] = []
    cursor = 0.0
    lines = plan["lines"]
    for index, line in enumerate(lines):
        job = by_id[line["unit_id"]]
        audio, _ = _read_mono(job["path"])
        side = _read_json(_sidecar(job))
        words = side.get("alignment")
        if not words or (align_words and alignment.available() and (
                realign or (words.get("method") == "estimated" and not side.get("alignment_tried")))):
            words = alignment.align(units[line["unit_id"]]["text"], audio, SAMPLE_RATE, use_whisper=align_words)
            side["alignment"], side["alignment_tried"] = words, bool(align_words and alignment.available())
            _sidecar(job).write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
        duration = len(audio) / SAMPLE_RATE
        if index == len(lines) - 1:
            gap = END_TAIL
        else:
            gap = max(line["pause_seconds"], SPEAKER_GAP if lines[index + 1]["speaker"] != line["speaker"] else MIN_GAP)
        pieces += [audio, np.zeros(int(round(gap * SAMPLE_RATE)), dtype="float32")]
        timeline.append({"unit_id": line["unit_id"], "section_id": line["section_id"], "beat_id": line["beat_id"],
                         "speaker": line["speaker"], "start": round(cursor, 3), "end": round(cursor + duration, 3),
                         "gap_after": round(gap, 3), "tone": line["tone"], "tone_used": side.get("tone_used", "neutral"),
                         "speed": side.get("speed"), "provider": side.get("provider_used"),
                         "emphasis": line["emphasis"], "alignment": words["method"], "matched": words.get("matched"),
                         "words": [{"text": w["text"], "start": round(cursor + w["start"], 3), "end": round(cursor + w["end"], 3)}
                                   for w in words["words"]],
                         "file": str(job["path"].relative_to(audio_dir.parent))})
        cursor += duration + gap
    track = np.concatenate(pieces) if pieces else np.zeros(0, dtype="float32")
    total = len(track) / SAMPLE_RATE
    sf.write(str(audio_dir / "voice.wav"), track, SAMPLE_RATE, subtype="PCM_16")

    beats: Dict[str, Dict[str, Any]] = {}
    order = [t["beat_id"] for t in timeline]
    for beat in dict.fromkeys(order):
        mine = [t for t in timeline if t["beat_id"] == beat]
        beats[beat] = {"start": mine[0]["start"], "units": [t["unit_id"] for t in mine]}
    ids = list(beats)
    for i, beat in enumerate(ids):
        beats[beat]["end"] = round(beats[ids[i + 1]]["start"] if i + 1 < len(ids) else total, 3)
        piece = track[int(beats[beat]["start"] * SAMPLE_RATE): int(beats[beat]["end"] * SAMPLE_RATE)]
        (audio_dir / "beats").mkdir(exist_ok=True)
        sf.write(str(audio_dir / "beats" / f"{beat}.wav"), piece, SAMPLE_RATE, subtype="PCM_16")
        beats[beat]["file"] = f"audio/beats/{beat}.wav"

    def at(event: Dict[str, Any]) -> float:
        t = next(x for x in timeline if x["unit_id"] == event["unit_id"])
        if event["anchor"] == "start":
            return t["start"]
        if event["anchor"] == "end":
            return t["end"]
        want = [alignment.norm(w) for w in alignment.tokens(event.get("words", ""))]
        heard = [alignment.norm(w["text"]) for w in t["words"]]
        for k in range(len(heard) - len(want) + 1):
            if want and heard[k:k + len(want)] == want:
                return t["words"][k]["start"]
        return t["start"]

    files = {kind: ac.scan(kind) for kind in ("sfx", "music", "ambience")}

    def bed(kind: str) -> List[Dict[str, Any]]:
        out = []
        for run in plan.get(kind, []):
            out.append({"asset": run["asset"], "file": (files[kind].get(run["asset"]) or {}).get("file", ""),
                        "from_beat": run["from_beat"], "to_beat": run["to_beat"], "start": beats[run["from_beat"]]["start"],
                        "end": beats[run["to_beat"]]["end"], "fade_in": FADE_IN, "fade_out": FADE_OUT})
        return out

    return {"version": 1, "sample_rate": SAMPLE_RATE, "script_hash": plan.get("script_hash"), "duration": round(total, 3),
            "voice_file": "audio/voice.wav", "beats": beats, "units": timeline,
            "events": {"sfx": [{**e, "at": round(at(e), 3), "file": (files["sfx"].get(e["asset"]) or {}).get("file", "")}
                               for e in plan.get("sfx", [])],
                       "dynamics": [{**e, "at": round(at(e), 3)} for e in plan.get("dynamics", [])],
                       "music": bed("music"), "ambience": bed("ambience")},
            "music_db": {l["unit_id"]: l["music_db"] for l in lines}, "mix": plan.get("mix", {})}


# ----------------------------------------------------------------------
# Tools: reference texts and a self-test
# ----------------------------------------------------------------------

def transcribe_refs() -> None:
    """Write the missing ref_text of every voice profile (from voices_config.json, else by listening)."""
    for file in sorted(_voices_dir().glob("*.json")):
        voice = _read_json(file)
        clip = voice.get("reference_audio", "")
        if voice.get("ref_text", "").strip() or not clip:
            continue
        text = _known_ref_text(clip)
        source = "voices_config.json"
        if not text:
            if not alignment.available():
                print(f"[Voice] {file.stem}: no ref_text and faster-whisper is not installed (pip install faster-whisper), "
                      f"or type the words of {clip} into the file by hand.")
                continue
            audio, rate = _read_mono(Path(config.F5_TTS_MODEL_DIR) / clip)
            text, source = alignment.transcribe(audio, rate), "faster-whisper"
        voice["ref_text"] = text
        file.write_text(json.dumps(voice, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Voice] {file.stem}: ref_text from {source}:\n         {text}\n         (listen to {clip} and fix the file if a word is wrong)")


def audition(names: List[str]) -> None:
    """Speak the same test sentence with several reference clips (one model load) to choose voices by ear."""
    refs = sorted((Path(config.F5_TTS_MODEL_DIR) / "ref_voices").glob("*.wav"))
    folder = Path(config.SHORTS_ROOT) / "runtime" / "voice_selftest"
    jobs: List[Dict[str, Any]] = []
    for name in names:
        match = next((r for r in refs if name.lower() in r.name.lower()), None)
        if match is None:
            print(f"[Voice] no reference clip matches {name!r}. Clips: {', '.join(r.stem for r in refs)}")
            continue
        rel = f"ref_voices/{match.name}"
        text = _known_ref_text(rel)
        if not text and alignment.available():
            audio, rate = _read_mono(match)
            text = alignment.transcribe(audio, rate)
            print(f"[Voice] {match.stem}: words of the clip, by listening: {text}")
        if not text:
            print(f"[Voice] {match.stem}: no ref_text known and faster-whisper is not installed, skipped")
            continue
        jobs.append({"unit_id": match.stem, "text": SELFTEST_TEXT, "clip": rel, "ref_text": text, "speed": 1.0, "seed": 1,
                     "raw": folder / f"audition_{match.stem}.wav", "voice_profile_id": match.stem})
    failed = run_f5(jobs, folder)
    for job in jobs:
        if job["unit_id"] in failed:
            print(f"[Voice] {job['unit_id']} FAILED: {failed[job['unit_id']]}")
            continue
        postprocess(job["raw"], folder / f"audition_{job['unit_id']}_final.wav")
        print(f"[Voice] {job['unit_id']} -> {folder / ('audition_' + job['unit_id'] + '_final.wav')}")


def selftest(provider: str) -> None:
    files = sorted(_voices_dir().glob("*.json"))
    if not files:
        raise VoiceError(f"No voice profiles in {_voices_dir()}")
    voice = _read_json(files[0])
    clip, ref_text, _ = _reference(voice, "neutral")
    folder = Path(config.SHORTS_ROOT) / "runtime" / "voice_selftest"
    job = {"unit_id": "selftest", "text": SELFTEST_TEXT, "clip": clip, "ref_text": ref_text, "speed": 1.0, "seed": 1,
           "kokoro_voice": _kokoro_voice(voice), "raw": folder / f"selftest_{provider}.wav", "voice_profile_id": voice.get("voice_profile_id")}
    if provider == "f5" and not ref_text.strip():
        raise VoiceError("The first voice has no ref_text yet: run --transcribe-refs first.")
    started = time.monotonic()
    failed = run_f5([job], folder) if provider == "f5" else run_kokoro([job])
    if failed:
        print(f"[Voice] {provider} self-test FAILED: {failed['selftest']}")
        return
    seconds = postprocess(job["raw"], folder / f"selftest_{provider}_final.wav")
    print(f"[Voice] {provider} self-test ok: {seconds:.1f} s of speech in {time.monotonic() - started:.0f} s -> {folder}")


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def _parse_set(text: str) -> Tuple[str, Dict[str, Any]]:
    unit, *pairs = text.split()
    values: Dict[str, Any] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key in ("speed",):
            values[key] = float(value)
        elif key == "seed":
            values[key] = int(value)
        elif key == "tone":
            values[key] = value
        else:
            raise VoiceError(f"--set understands speed, seed and tone, not {key!r}")
    return unit, values


def main() -> None:
    from agents._common import read_brief
    from core.models.script import Script
    from core.registry_loader import RegistryLoader
    from core.run_context import RunContext

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-run")
    parser.add_argument("--go", action="store_true", help="speak what is missing, join and align (default: preview only)")
    parser.add_argument("--set", action="append", default=[], metavar='"UNIT speed=0.95 seed=7 tone=warm"')
    parser.add_argument("--provider", default="", help="force one provider for every sentence (f5 or kokoro)")
    parser.add_argument("--no-fallback", action="store_true", help="do not fall back to Kokoro when F5 fails")
    parser.add_argument("--no-align", action="store_true", help="estimate word times instead of listening")
    parser.add_argument("--realign", action="store_true", help="listen again to every sentence for word times")
    parser.add_argument("--audition", nargs="+", metavar="CLIP", help="speak a test sentence with reference clips matching these names")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--transcribe-refs", action="store_true")
    parser.add_argument("--selftest", choices=["f5", "kokoro"])
    args = parser.parse_args()

    if args.transcribe_refs:
        return transcribe_refs()
    if args.selftest:
        return selftest(args.selftest)
    if args.audition:
        return audition(args.audition)
    if not args.from_run:
        parser.error("--from-run is required")

    ctx = RunContext.load(args.from_run)
    audio_dir = Path(ctx.run_directory) / "audio"
    if args.clear_cache:
        import shutil
        shutil.rmtree(audio_dir, ignore_errors=True)
        print(f"[Voice] Cache cleared: {audio_dir}")
        return
    plan = ctx.read_stage_output("voice_plan")
    if not plan:
        raise VoiceError("This run has no voice_plan yet: run the Voice Director first.")
    script = Script(**ctx.read_stage_output("story_director"))
    if plan.get("script_hash") != ar.script_hash(script.sections):
        raise VoiceError("The script changed after the voice plan was made: run the Voice Director again.")
    info = read_brief(ctx)

    overrides_file = audio_dir / "overrides.json"
    overrides = _read_json(overrides_file) if overrides_file.exists() else {}
    for text in args.set:
        unit, values = _parse_set(text)
        overrides.setdefault(unit, {}).update(values)
    if args.set:
        audio_dir.mkdir(parents=True, exist_ok=True)
        overrides_file.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    jobs = build_jobs(plan, script, RegistryLoader(), info["channel_id"], overrides, audio_dir, args.provider)
    todo = [j for j in jobs if not is_done(j)]
    print(f"[Voice] {len(jobs)} sentence(s), {len(todo)} to speak, {len(jobs) - len(todo)} already cached.")
    for j in jobs:
        note = "" if j["tone"] == j["tone_wanted"] or j["tone_wanted"] == "neutral" else f" (wanted {j['tone_wanted']}: no clip yet)"
        print(f"  {j['unit_id']:<11}{j['provider']:<8}speed {j['speed']:<5} tone {j['tone']:<9}{'cached' if is_done(j) else 'to speak'}{note}")
    if not args.go:
        print("Preview only. Add --go to speak.")
        return
    failures = speak(jobs, audio_dir, allow_fallback=not args.no_fallback)
    if failures:
        raise VoiceError("These sentences could not be spoken: " + "; ".join(f"{u}: {r}" for u, r in failures.items()))
    timing = assemble(plan, jobs, script, audio_dir, align_words=not args.no_align, realign=args.realign)
    ctx.write_stage_output("voice_timing", timing)
    ctx.save()
    methods = sorted({u["alignment"] for u in timing["units"]})
    weak = [u["unit_id"] for u in timing["units"] if u["alignment"] == "estimated"]
    if weak:
        print(f"[Voice] Word times only estimated for: {', '.join(weak)}"
              + ("" if alignment.available() else " (faster-whisper is not installed)"))
    print(f"[Voice] Voice track {timing['duration']:.1f} s (word times: {', '.join(methods)}) -> {audio_dir / 'voice.wav'}")
    print(f"[Voice] Beats: " + ", ".join(f"{b} {v['start']:.1f}-{v['end']:.1f}s" for b, v in timing["beats"].items()))


if __name__ == "__main__":
    main()