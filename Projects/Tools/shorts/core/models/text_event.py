from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextEvent:
    """
    Represents one semantic text event within a Beat.

    A TextEvent describes textual content, timing, emphasis, and the
    intended presentation treatment. Typography, positioning, styling,
    timing refinement, and rendering are handled downstream.

    Presentation treatment is intentionally semantic rather than tied
    to a specific rendering implementation. For example, a TextEvent
    may request "kinetic emphasis" without specifying fonts, colors,
    size, position, animation, or other rendering details.
    """

    event_id: str

    text: str

    start_time: float
    end_time: float

    emphasis: Optional[str] = None

    presentation_intent: Optional[str] = None

    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)