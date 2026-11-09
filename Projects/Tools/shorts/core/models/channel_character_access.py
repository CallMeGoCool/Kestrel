from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class CharacterAccessType(str, Enum):
    """
    Defines how a character is allowed to appear within a channel.
    """

    RESIDENT = "resident"
    GUEST = "guest"


@dataclass
class ChannelCharacterAccess:
    """
    Defines a character's relationship with a specific channel.

    Character identity remains global. This model only describes how that
    character may be used within a particular channel.
    """

    channel_id: str
    character_id: str

    access_type: CharacterAccessType = CharacterAccessType.RESIDENT

    restrictions: Tuple[str, ...] = ()

    metadata: dict = field(default_factory=dict)