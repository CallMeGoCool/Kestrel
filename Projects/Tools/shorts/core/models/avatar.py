from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models._serde import coerce_fields


@dataclass
class Pose:
    """One scene-ready image of an avatar."""

    pose_id: str
    file: str
    description: str = ""


@dataclass
class AvatarAssets:
    canonical_image: str   # identity reference only; scenes use poses
    poses_directory: str   # relative to the project root


@dataclass
class Avatar:
    """
    A character's visual identity. Mirrors registry/avatars/*.json.

    The default pose is the twin of the canonical image.
    """

    avatar_id: str
    character_id: str
    assets: AvatarAssets
    default_pose: str
    poses: List[Pose] = field(default_factory=list)

    display_name: str = ""
    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerce_fields(self)

    @property
    def pose_ids(self) -> List[str]:
        return [pose.pose_id for pose in self.poses]

    def get_pose(self, pose_id: str) -> Optional[Pose]:
        return next((p for p in self.poses if p.pose_id == pose_id), None)