"""
Kestrel Shorts - Core Configuration
===================================

Stable runtime/platform settings for the Shorts application (vertical,
short-form). Studio is a separate clone of this project and is not
configured here.

Nothing in this file touches the disk on import. Call
`ensure_directories()` once at application startup.

Not in this file: creative decisions, director logic, typography, voice
profiles, ComfyUI workflows, LoRA routing, composition rules.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# ============================================================
# 0. PLACE AND TIME
# ============================================================
#
# Stamped into every run and every search so agents can judge how fresh
# information is (e.g. "news from the last 24 hours"). Edit LOCATION.

LOCATION = "Chennai, India"


def now_stamp() -> dict:
    """Current local date, time, timezone and location."""
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "time": now.strftime("%H:%M"),
        "timezone": now.tzname() or "",
        "utc_offset": now.strftime("%z"),
        "location": LOCATION,
    }


def describe_stamp(stamp: dict) -> str:
    """One line for prompts, e.g. 'Saturday 2026-10-17 19:05 (UTC+0530), Chennai, India'."""
    return (
        f"{stamp.get('weekday', '')} {stamp.get('date', '')} {stamp.get('time', '')} "
        f"(UTC{stamp.get('utc_offset', '')}), {stamp.get('location', '')}"
    ).strip()


# ============================================================
# 1. PATHS
# ============================================================

# The project folder, derived from this file so moving it never breaks paths.
SHORTS_ROOT = Path(__file__).resolve().parent.parent

# Shared storage (models, outputs). Override with KESTREL_BASE_DIR if needed.
BASE_DIR = Path(os.environ.get("KESTREL_BASE_DIR", r"C:\AI_Studio"))
MODELS_DIR = BASE_DIR / "Models"
OUTPUTS_ROOT = BASE_DIR / "Outputs"

SHORTS_OUTPUTS_DIR = OUTPUTS_ROOT / "Shorts"
SHORTS_IMAGES_DIR = SHORTS_OUTPUTS_DIR / "Images"
SHORTS_VIDEOS_DIR = SHORTS_OUTPUTS_DIR / "Videos"
SHORTS_TEMP_DIR = SHORTS_OUTPUTS_DIR / "Temp"
SHORTS_AUDIO_DIR = SHORTS_OUTPUTS_DIR / "Audio"
SHORTS_CAPTIONS_DIR = SHORTS_OUTPUTS_DIR / "Captions"
SHORTS_PROJECTS_DIR = SHORTS_OUTPUTS_DIR / "Projects"

# Working folders for runs: context, stage outputs, images, audio, reports.
RUNS_DIR = SHORTS_ROOT / "runtime" / "runs"

# Shared voice/model resources (not owned by Shorts).
KOKORO_MODEL_DIR = MODELS_DIR / "Kokoro"
F5_TTS_MODEL_DIR = MODELS_DIR / "F5-TTS"

# ============================================================
# 2. SERVICES
# ============================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_URL = OLLAMA_BASE_URL + "/api/generate"   # legacy name, kept
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT = 300                              # legacy; see LLM_IDLE_TIMEOUT

COMFYUI_URL = "http://127.0.0.1:8188"
SEARCH_SERVICE_URL = "http://127.0.0.1:8000/search"

# The shared search service (also used by Studio and Open WebUI). If it is
# not running, core/search.py can start it.
SEARCH_SERVICE_SCRIPT = BASE_DIR / "Projects" / "ResearchAssistant" / "search_service.py"
SEARCH_AUTOSTART = True
SEARCH_RESULTS_PER_QUERY = 5

# ============================================================
# 3. LLM SETTINGS (read by core/llm.py)
# ============================================================
#
# One table so every agent is tuned in one place. These are starting
# values for a 6 GB GPU; adjust from the token logs. num_ctx is the
# window shared by prompt AND output, so keep num_ctx comfortably above
# (expected prompt tokens + num_predict).

LLM_STAGES = {
    "orchestrator":    {"num_ctx": 4096, "num_predict": 1200, "temperature": 0.2},
    "researcher":      {"num_ctx": 8192, "num_predict": 1500, "temperature": 0.2},
    "architect":       {"num_ctx": 8192, "num_predict": 1500, "temperature": 0.5},
    "script_refiner":  {"num_ctx": 8192, "num_predict": 2000, "temperature": 0.6},
    "story_director":  {"num_ctx": 8192, "num_predict": 3000, "temperature": 0.45},
    "visual_director": {"num_ctx": 8192, "num_predict": 1500, "temperature": 0.25},
    "reviewer":        {"num_ctx": 6144, "num_predict": 1000, "temperature": 0.2},
    "image_prompter":  {"num_ctx": 6144, "num_predict": 1600, "temperature": 0.35},
    "voice_director":   {"num_ctx": 6144, "num_predict": 1400, "temperature": 0.4},
    "publisher":       {"num_ctx": 4096, "num_predict": 600,  "temperature": 0.8},
}

# With streaming, this is the longest silence allowed between tokens,
# not the total generation time.
LLM_IDLE_TIMEOUT = 120
LLM_KEEP_ALIVE = "30m"      # keep the model loaded between LLM stages
LLM_MAX_RETRIES = 2

# Director/reviewer back-and-forth is capped; the runner enforces this.
MAX_NEGOTIATION_ROUNDS = 2

# ============================================================
# 4. VIDEO FORMAT (Shorts: vertical)
# ============================================================

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
ASPECT_RATIO = "9:16"
DEFAULT_FPS = 30

# ============================================================
# 5. DURATION
# ============================================================
#
# Length is decided per video (user request, else the Architect).
# These are only the outer limits and the words-per-second rate used
# to turn a target length into a word budget.
#
# Ceiling: YouTube Shorts allows up to 3 minutes. Other platforms vary
# by account, so 180 s is the safe common ceiling.

MIN_VIDEO_SECONDS = 8.0
MAX_VIDEO_SECONDS = 180.0

# Placeholder rate (~145 wpm). Recalibrate from real TTS timings.
WORDS_PER_SECOND = 2.4

# How far the written script may drift from its word budget.
WORD_BUDGET_TOLERANCE = 0.15


def clamp_duration(seconds: float) -> float:
    """Clamp a requested length to the allowed range."""
    return float(min(MAX_VIDEO_SECONDS, max(MIN_VIDEO_SECONDS, float(seconds))))


def word_budget(seconds: float) -> int:
    """Spoken-word budget for a target length."""
    return int(round(clamp_duration(seconds) * WORDS_PER_SECOND))


# ============================================================
# 5b. VISUAL PLANNING LIMITS
# ============================================================

MAX_GENERATED_IMAGES = 100      # new or image-to-image images per video; beyond this, reuse
MAX_SHOTS_PER_BEAT = 8          # safety cap; the Visual Director decides the real number
MIN_WORDS_PER_SHOT = 6          # a shot needs about this much narration (~2.5 s)

# ============================================================
# 6. ENCODING
# ============================================================

VIDEO_CODEC = "h264_nvenc"
VIDEO_PRESET = "p4"
VIDEO_CPU_CODEC = "libx264"        # fallback when NVENC is unavailable
VIDEO_CPU_PRESET = "medium"

# ============================================================
# 7. TIMING DEFAULTS
# ============================================================

MIN_BEAT_DURATION = 0.5
MAX_BEAT_DURATION = 8.0
MIN_SCENE_DURATION = 1.0

CAPTION_MIN_DURATION = 0.75
CAPTION_TARGET_DURATION = 2.5
CAPTION_MAX_DURATION = 4.0
CAPTION_MAX_WORDS = 6
CAPTION_MAX_CHARACTERS = 32

# ============================================================
# 8. RENDERING / APPLICATION DEFAULTS
# ============================================================

AVATAR_ALPHA_THRESHOLD = 1
DEFAULT_BACKGROUND = "black"

DEFAULT_MODE = "shorts"
DEFAULT_CONTENT_RATING = "safe"

SCRIPT_REFINER_ENABLED = True
MAX_SCRIPT_REFINEMENT_RETRIES = 2

# ============================================================
# 9. DIRECTORY INITIALIZATION (call once at startup)
# ============================================================

SHORTS_DIRECTORIES = (
    SHORTS_OUTPUTS_DIR,
    SHORTS_IMAGES_DIR,
    SHORTS_VIDEOS_DIR,
    SHORTS_TEMP_DIR,
    SHORTS_AUDIO_DIR,
    SHORTS_CAPTIONS_DIR,
    SHORTS_PROJECTS_DIR,
    RUNS_DIR,
)


def ensure_directories() -> None:
    """Create the output and run folders if they do not exist."""
    for directory in SHORTS_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)