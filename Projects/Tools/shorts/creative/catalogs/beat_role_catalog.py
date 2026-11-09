"""
Catalog of beat roles: what a beat DOES in the video.

The Story Director labels each beat with one of these. Later stages use the
role to decide the visuals, for example how likely an avatar is to appear
(`avatar_fit`). The catalog holds names and hints only; code decides what to
do with them.
"""
from typing import Any, Dict, List, Optional

BEAT_ROLE_CATALOG: Dict[str, Dict[str, Any]] = {
    "hook": {
        "description": "The opening: makes the viewer curious.",
        "avatar_fit": "high",
    },
    "explanation": {
        "description": "Explains an idea or how something works.",
        "avatar_fit": "low",
    },
    "question": {
        "description": "A character asks or challenges something.",
        "avatar_fit": "high",
    },
    "reaction": {
        "description": "A character reacts to what was just said.",
        "avatar_fit": "high",
    },
    "emphasis": {
        "description": "A key moment the viewer should not miss.",
        "avatar_fit": "medium",
    },
    "example": {
        "description": "A concrete example, story or illustration of the idea.",
        "avatar_fit": "low",
    },
    "data": {
        "description": "Numbers, lists, timelines or comparisons.",
        "avatar_fit": "low",
    },
    "conclusion": {
        "description": "Where the main takeaway lands. It need not be the last beat.",
        "avatar_fit": "high",
    },
    "coda": {
        "description": "A light closing note after the conclusion (a callback, a question, a joke). Optional.",
        "avatar_fit": "high",
    },
}


def get_beat_role_catalog() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in BEAT_ROLE_CATALOG.items()}


def get_beat_role_names() -> List[str]:
    return list(BEAT_ROLE_CATALOG)


def is_valid_beat_role(role: Optional[str]) -> bool:
    return isinstance(role, str) and role in BEAT_ROLE_CATALOG