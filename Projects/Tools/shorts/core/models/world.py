from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from core.models._serde import coerce_fields


@dataclass
class Environment:
    setting: str = ""
    time_of_day: str = ""
    atmosphere: List[str] = field(default_factory=list)
    background_theme: str = ""
    environmental_conventions: List[str] = field(default_factory=list)


@dataclass
class World:
    """
    A channel's sub-genre / direction. Mirrors registry/worlds/*.json,
    including the nested `environment` block.
    """

    world_id: str
    name: str
    description: str = ""
    visual_identity: str = ""   # the look and mood of images in this world (a few sentences)

    environment: Environment = field(default_factory=Environment)
    storytelling_conventions: List[str] = field(default_factory=list)
    image_generation_guidance: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerce_fields(self)

    # Compatibility accessors for agents written against the old flat model.
    @property
    def atmosphere(self) -> str:
        return ", ".join(self.environment.atmosphere)

    @property
    def visual_aesthetic(self) -> str:
        return self.environment.background_theme

    @property
    def environmental_conventions(self) -> Tuple[str, ...]:
        return tuple(self.environment.environmental_conventions)