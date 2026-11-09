"""
Voice-direction options the Story Director may choose from.

Trimmed to what the voice engines can really act on:
  - pace        -> a speaking-speed setting
  - pause_after -> silence inserted after the line

Emotion and intensity are not offered: F5-TTS and Kokoro have no such
controls. (Emotion could later come from different reference clips per
character.) This file only names the options; the voice engine decides what
each one means. ESTIMATED_PAUSE_SECONDS is used only to estimate video
length before audio exists; real timing always comes from the audio.
"""
from typing import Dict, List

VOICE_CATALOG: Dict[str, Dict[str, str]] = {
    "pace": {
        "slow": "Unhurried; for weighty or reflective lines.",
        "natural": "Normal conversational speed.",
        "brisk": "Quicker; for lively or excited lines.",
    },
    "pause_after": {
        "none": "No extra pause.",
        "short": "A brief beat.",
        "medium": "A clear pause.",
        "long": "A long pause; for a key moment.",
    },
}

DEFAULTS = {"pace": "natural", "pause_after": "none"}

ESTIMATED_PAUSE_SECONDS = {"none": 0.0, "short": 0.3, "medium": 0.6, "long": 1.0}


def get_voice_catalog() -> Dict[str, Dict[str, str]]:
    return {c: dict(o) for c, o in VOICE_CATALOG.items()}


def get_voice_options(category: str) -> Dict[str, str]:
    return dict(VOICE_CATALOG.get(category, {}))


def get_voice_option_names(category: str) -> List[str]:
    return list(VOICE_CATALOG.get(category, {}))


def is_valid_voice_option(category: str, name: str) -> bool:
    return name in VOICE_CATALOG.get(category, {})