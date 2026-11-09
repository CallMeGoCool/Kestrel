from dataclasses import dataclass, field
from typing import List


@dataclass
class Scene:
    """
    Represents one continuous narrative section of a Short.

    A Scene groups related beats and provides a higher-level container for
    narrative, visual, character, and continuity context.

    It does not perform rendering, composition, asset generation, or audio
    generation.
    """

    scene_id: str

    start_time: float = 0.0
    end_time: float = 0.0

    beats: List["Beat"] = field(default_factory=list)

    description: str = ""
    visual_context: str = ""

    character_ids: List[str] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def add_beat(self, beat: "Beat") -> None:
        self.beats.append(beat)

    def add_character(self, character_id: str) -> None:
        if character_id not in self.character_ids:
            self.character_ids.append(character_id)

    @property
    def beat_count(self) -> int:
        return len(self.beats)