"""
Kestrel Shorts - Audio catalog
==============================

The vocabulary the Voice Director chooses from, and the numbers code attaches to each word. The agent picks
WORDS (a pace, a tone, an energy, a level); every number (speed, seconds, decibels, how many effects fit
per minute) lives here, so one change retunes the whole system.

  Per line     pace, tone, pause_after, emphasis
  Per beat     energy (the sound arc of the video) and tempo; a "moment" can shift both inside a beat
  Music        one bed per run of beats; its loudness follows the energy of each line
  Sound        effects anchored to a line (start, end, or exact words), and dynamics (dip, drop_out, swell)

The voice engines (F5-TTS, Kokoro) can change speed and insert silence. Tone needs a reference clip per
tone (a `styles` map in the character's voice profile); until a clip exists the engine speaks the plain
voice and the plan records the wish.
"""
from __future__ import annotations

import re
from typing import Dict, List

PACES: Dict[str, str] = {
    "slow": "Unhurried; for weighty, reflective or tense lines.",
    "natural": "Normal conversational speed.",
    "brisk": "Quicker; for lively or excited lines.",
    "rushed": "Fast and urgent; for a peak or a burst of excitement. Use rarely.",
}
PACE_SPEED: Dict[str, float] = {"slow": 0.88, "natural": 1.0, "brisk": 1.12, "rushed": 1.22}

TONES: Dict[str, str] = {
    "neutral": "Plain, even delivery.",
    "warm": "Friendly and relaxed, a smile in the voice.",
    "curious": "Questioning, rising at the end; for real questions and wonder.",
    "excited": "Bright and energetic; for discoveries and peaks.",
    "serious": "Low, steady, weighty; for facts that matter.",
    "hushed": "Soft and close, almost a whisper; for secrets, tension and awe.",
    "amused": "Light and playful; a smile or a small chuckle.",
    "dramatic": "Slow, deliberate, full of weight; for a reveal.",
}
# A long pause is part of the expression: it is only kept after a line that earns it.
EXPRESSIVE_TONES = {"curious", "excited", "serious", "hushed", "amused", "dramatic"}

PAUSES: Dict[str, str] = {
    "none": "No extra pause.",
    "short": "A brief beat.",
    "medium": "A clear pause.",
    "long": "A long pause; lets a question or a heavy fact land.",
    "hold": "A held silence; the beat before a reveal.",
}
PAUSE_SECONDS: Dict[str, float] = {"none": 0.0, "short": 0.3, "medium": 0.6, "long": 1.0, "hold": 1.6}
BEAT_END_PAUSE: Dict[str, str] = {"hook": "short", "question": "medium", "conclusion": "short"}   # other roles: short

ENERGIES: Dict[str, str] = {
    "hush": "Very quiet and tense: the moment before something happens.",
    "calm": "Gentle and reflective.",
    "steady": "Even and informative.",
    "rising": "Interest or tension building.",
    "intense": "Fast and strong: the push toward a peak.",
    "peak": "The climax or the biggest surprise.",
}
ENERGY_TEMPO: Dict[str, str] = {"hush": "slow", "calm": "natural", "steady": "natural", "rising": "natural",
                                "intense": "brisk", "peak": "brisk"}
ROLE_ENERGY: Dict[str, str] = {"hook": "rising", "explanation": "steady", "question": "rising",
                               "conclusion": "calm", "coda": "calm"}

LEVELS: Dict[str, str] = {"quiet": "Barely there.", "low": "Under the voice.", "medium": "Present.",
                          "high": "Prominent."}
LEVEL_DB: Dict[str, float] = {"quiet": -30.0, "low": -24.0, "medium": -18.0, "high": -12.0}   # relative to full scale
ENERGY_LEVEL: Dict[str, str] = {"hush": "quiet", "calm": "low", "steady": "low", "rising": "medium",
                                "intense": "medium", "peak": "high"}                          # music level per energy
VOICE_DB = -3.0
DUCK_DB = -9.0                       # music and ambience under speech

DYNAMICS: Dict[str, str] = {
    "dip": "Music and ambience fall for a moment: tension before a reveal.",
    "drop_out": "Everything but the voice goes silent for a moment (use rarely).",
    "swell": "Music rises for a moment: a payoff.",
}
DYNAMIC_SECONDS: Dict[str, float] = {"short": 0.8, "medium": 1.5, "long": 3.0}
DYNAMIC_DB: Dict[str, float] = {"dip": -10.0, "swell": 6.0, "drop_out": -60.0}
MAX_DYNAMICS_PER_BEAT = 1
MAX_DROP_OUTS = 2
SECONDS_PER_DYNAMIC = 25             # at most one dynamic per this many seconds (and at least one per video)

ANCHORS: List[str] = ["start", "end", "words"]
SFX_INTERVAL_SECONDS: Dict[str, int] = {"low": 30, "medium": 15, "high": 8}     # one effect per this many seconds
SFX_PER_BEAT = 2


def names(options: Dict[str, str]) -> List[str]:
    return list(options)


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:limit]


def help_text(options: Dict[str, str]) -> str:
    return "; ".join(f"{k} ({v.rstrip('.')})" for k, v in options.items())