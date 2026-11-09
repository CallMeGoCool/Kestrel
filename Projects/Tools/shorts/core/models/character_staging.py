from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CharacterStageAction(str, Enum):
    """
    Defines the semantic action a character performs within a beat.
    """

    PRESENT = "present"
    ENTER = "enter"
    EXIT = "exit"
    REACT = "react"
    SPEAK = "speak"
    LISTEN = "listen"
    EMPHASIZE = "emphasize"


@dataclass
class CharacterStaging:
    """
    Describes how a character participates visually within a Beat.

    This is semantic staging intent. It does not define pixel coordinates,
    pose filenames, scaling values, or rendering operations.
    """

    character_id: str

    action: CharacterStageAction = CharacterStageAction.PRESENT

    pose_id: Optional[str] = None

    visual_role: Optional[str] = None

    position_hint: Optional[str] = None

    emphasis: Optional[str] = None

    metadata: dict = field(default_factory=dict)