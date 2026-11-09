from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Channel:
    """
    A channel's identity. Mirrors registry/channels/*.json.

    It only lists IDs of its worlds and characters; their details live in
    their own files.
    """

    channel_id: str
    name: str
    family_id: str

    description: str = ""
    world_ids: List[str] = field(default_factory=list)
    available_character_ids: List[str] = field(default_factory=list)
    storytelling_style: List[str] = field(default_factory=list)

    # Who the videos are for, and how often avatars appear (none | cameo | frequent | constant).
    audience: str = ""
    reading_grade: int = 0
    avatar_presence: str = ""
    visual_identity: str = ""           # the look and mood every image of this channel keeps (a few sentences)
    theme_id: str = ""                  # the channel's presentation kit in registry/themes
    default_visual_style: str = ""

    # Optional; kept so agents written against the old model still run.
    topic_focus: List[str] = field(default_factory=list)
    visual_style: Any = None

    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)