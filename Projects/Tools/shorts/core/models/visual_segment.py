from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VisualSegment:
    """
    Represents one visual portion of a Beat.

    A VisualSegment describes visual intent and timing. It does not contain
    rendered images, filesystem paths, coordinates, or rendering settings.
    """

    segment_id: str

    start_time: float
    end_time: float

    visual_intent: str = ""

    asset_id: Optional[str] = None

    generation_required: bool = False

    transition: Optional[str] = None

    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)