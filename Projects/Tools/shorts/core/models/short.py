from dataclasses import dataclass, field
from typing import List


@dataclass
class Short:
    """
    Represents one complete Kestrel Shorts production.

    A Short belongs to a channel and contains the scenes that make up the
    narrative timeline. It stores production-level information, but does not
    perform story generation, rendering, audio generation, or video encoding.
    """

    project_id: str
    channel_id: str

    title: str = ""

    target_duration: float = 120.0
    min_duration: float = 90.0
    max_duration: float = 120.0

    width: int = 1080
    height: int = 1920
    fps: int = 30

    scenes: List["Scene"] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    def add_scene(self, scene: "Scene") -> None:
        self.scenes.append(scene)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    @property
    def duration(self) -> float:
        if not self.scenes:
            return 0.0

        return max(scene.end_time for scene in self.scenes)