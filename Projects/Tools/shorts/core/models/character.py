from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models._serde import coerce_fields
from core.models.avatar import Avatar
from core.models.voice_profile import VoiceProfile


@dataclass
class Character:
    """
    A character's fixed identity. Mirrors registry/characters/*.json.

    `voice` and `avatar` are filled by the loader when resolving a channel;
    they are not part of the character file.
    """

    character_id: str
    display_name: str
    description: str = ""

    personality: Dict[str, Any] = field(default_factory=dict)
    behavior: List[str] = field(default_factory=list)
    humanizing_biases: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    voice_profile_id: str = ""
    avatar_id: str = ""

    voice: Optional[VoiceProfile] = None
    avatar: Optional[Avatar] = None

    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerce_fields(self)

    @property
    def communication_style(self) -> str:
        return str(self.personality.get("communication_style", ""))