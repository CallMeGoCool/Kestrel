from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Beat:
    """
    Represents the smallest meaningful timed storytelling unit of a Short.

    A Beat may contain narration/dialogue, visual segments, text events,
    and character staging. A Beat is not tied to a specific rendered image,
    video clip, asset, coordinate system, or rendering engine.
    """

    beat_id: str

    start_time: float
    end_time: float

    narration: str = ""

    speaker_character_id: Optional[str] = None

    visual_segments: List["VisualSegment"] = field(default_factory=list)

    text_events: List["TextEvent"] = field(default_factory=list)

    character_staging: List["CharacterStaging"] = field(default_factory=list)

    transition: Optional[str] = None

    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def add_visual_segment(self, segment: "VisualSegment") -> None:
        self.visual_segments.append(segment)

    def add_text_event(self, event: "TextEvent") -> None:
        self.text_events.append(event)

    def add_character_staging(
        self,
        staging: "CharacterStaging",
    ) -> None:
        self.character_staging.append(staging)