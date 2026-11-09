from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class VoiceProfile:
    """
    A character's persistent voice. Mirrors registry/voices/*.json.

    `reference_audio` is relative to F5_TTS_MODEL_DIR. `ref_text` is the
    transcript of the reference clip; empty means the engine auto-transcribes.
    """

    voice_profile_id: str
    character_id: str
    provider: str
    reference_audio: str

    ref_text: str = ""
    language: str = "en"
    voice_description: str = ""
    generation_settings: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)