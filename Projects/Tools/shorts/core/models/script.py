from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import config

_PROMOTED = (
    "beat_title", "narrative_purpose", "setting_context", "atmosphere",
    "continuity_from_previous", "continuity_to_next",
)


@dataclass
class ScriptSection:
    """
    One spoken line by one speaker. Sections sharing a beat_id form one
    beat, which is how several characters speak within a single beat.
    """

    section_id: str
    text: str
    beat_id: Optional[str] = None
    beat_title: str = ""
    beat_role: str = ""            # hook | explanation | question | ... (beat_role catalog)
    speaker_character_id: Optional[str] = None
    emphasis: Optional[str] = None
    performance_intent: Optional[str] = None
    voice_direction: Dict[str, str] = field(default_factory=dict)

    narrative_purpose: str = ""
    setting_context: str = ""
    atmosphere: str = ""
    continuity_from_previous: str = ""
    continuity_to_next: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Older output kept these inside metadata; promote them.
        for key in _PROMOTED:
            if not getattr(self, key) and isinstance(self.metadata.get(key), str):
                setattr(self, key, self.metadata[key])
        if not self.voice_direction and isinstance(self.metadata.get("voice_direction"), dict):
            self.voice_direction = dict(self.metadata["voice_direction"])

    @property
    def beat_key(self) -> str:
        """The beat this section belongs to (its own ID if it has no beat_id)."""
        return self.beat_id or self.section_id

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ScriptBeat:
    """A computed view: consecutive sections sharing a beat. Not stored; has no timing."""

    beat_id: str
    title: str
    sections: List[ScriptSection]

    @property
    def role(self) -> str:
        return self.sections[0].beat_role if self.sections else ""

    @property
    def speakers(self) -> List[str]:
        seen: List[str] = []
        for s in self.sections:
            if s.speaker_character_id and s.speaker_character_id not in seen:
                seen.append(s.speaker_character_id)
        return seen

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sections)


@dataclass
class Script:
    """
    Spoken script. The Script Refiner writes a sketch (no speakers); the
    Story Director writes the final version and owns its length:
    `target_seconds` is the length it settled on, which may differ from the
    Architect's proposal. Real timing comes later from TTS.
    """

    title: str
    sections: List[ScriptSection] = field(default_factory=list)
    target_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sections = [
            ScriptSection(**s) if isinstance(s, dict) else s for s in self.sections
        ]

    def add_section(self, section: ScriptSection) -> None:
        self.sections.append(section)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def full_text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text.strip())

    @property
    def word_count(self) -> int:
        return sum(s.word_count for s in self.sections)

    def estimated_seconds(self, words_per_second: Optional[float] = None) -> float:
        rate = words_per_second or config.WORDS_PER_SECOND
        return self.word_count / rate

    @property
    def beats(self) -> List[ScriptBeat]:
        result: List[ScriptBeat] = []
        for section in self.sections:
            if result and result[-1].beat_id == section.beat_key:
                result[-1].sections.append(section)
            else:
                result.append(ScriptBeat(section.beat_key, section.beat_title, [section]))
        return result