from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Theme:
    """
    A channel's presentation kit. Mirrors registry/themes/*.json.

    `format` is the presentation recipe (the Remotion template it is keyed by); `format_hints`
    tells the directors in a few sentences what that recipe is like. Motion, transition and
    overlay names come from creative/catalogs/visual_catalog.py.
    """

    theme_id: str
    name: str
    format: str

    description: str = ""
    format_hints: str = ""
    palette: Dict[str, str] = field(default_factory=dict)
    typography: Dict[str, Any] = field(default_factory=dict)
    motion: Dict[str, Any] = field(default_factory=dict)
    transitions: List[str] = field(default_factory=list)
    overlays: List[str] = field(default_factory=list)
    image_treatment: Dict[str, Any] = field(default_factory=dict)
    audio: Dict[str, Any] = field(default_factory=dict)
    subtitles: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)