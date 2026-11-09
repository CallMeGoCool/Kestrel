"""
Kestrel Shorts — Avatar Pose System

Purpose:
    Provide a semantic interface for discovering and resolving the
    available visual poses of an avatar.

Architecture:

        AssetManager
              ↓
        ResolvedAsset
              ↓
      AvatarPoseSystem
              ↓
      Composition / Director

This module is intentionally AI-independent.

It does NOT:
    - call an LLM
    - call ComfyUI
    - generate poses
    - render images
    - calculate geometry
    - position avatars
    - make composition decisions

The AssetManager remains responsible for:
    - filesystem discovery
    - asset registration
    - asset resolution
    - asset paths

The AvatarPoseSystem is responsible only for:
    - understanding avatar → pose relationships
    - listing available poses
    - checking whether a pose exists
    - resolving an avatar pose through AssetManager
"""

from __future__ import annotations

from typing import Dict, List, Optional

from shorts.assets.asset_manager import (
    ASSET_AVATAR_POSE,
    AssetManager,
    ResolvedAsset,
)


# ---------------------------------------------------------------------------
# Avatar Pose System
# ---------------------------------------------------------------------------

class AvatarPoseSystem:
    """
    Semantic interface for avatar poses.

    The AvatarPoseSystem does not maintain a separate filesystem registry.

    AssetManager remains the source of truth for discovered assets.

    Example:

        manager = AssetManager()
        manager.discover_all()

        poses = AvatarPoseSystem(manager)

        poses.list_poses("scientist_male_2")

        poses.get_pose(
            "scientist_male_2",
            "head_sideways",
        )
    """

    def __init__(
        self,
        asset_manager: AssetManager,
    ):
        """
        Initialize the AvatarPoseSystem.

        Parameters:
            asset_manager:
                Existing AssetManager instance used for discovery
                and asset resolution.
        """

        if not isinstance(asset_manager, AssetManager):
            raise TypeError(
                "asset_manager must be an instance of AssetManager."
            )

        self.asset_manager = asset_manager

    # ------------------------------------------------------------------
    # Discovery / Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> List[ResolvedAsset]:
        """
        Refresh avatar pose information from the filesystem.

        AssetManager remains responsible for the actual discovery.

        Returns:
            List of currently discovered avatar pose assets.
        """

        return self.asset_manager.discover_avatar_poses()

    # ------------------------------------------------------------------
    # Avatar Listing
    # ------------------------------------------------------------------

    def list_avatars(self) -> List[str]:
        """
        Return avatar IDs that currently have at least one discovered pose.

        Example:

            [
                "genius_female",
                "scientist_female_1",
                "scientist_male_2",
            ]
        """

        poses = self.asset_manager.list_assets(
            ASSET_AVATAR_POSE
        )

        avatar_ids = set()

        for asset in poses:
            avatar_id = self._get_avatar_id(asset)

            if avatar_id:
                avatar_ids.add(avatar_id)

        return sorted(avatar_ids)

    # ------------------------------------------------------------------
    # Pose Listing
    # ------------------------------------------------------------------

    def list_poses(
        self,
        avatar_id: str,
    ) -> List[str]:
        """
        Return all currently available pose names for an avatar.

        Example:

            list_poses("scientist_male_2")

            -> ["default", "head_sideways"]
        """

        avatar_id = self._normalize_avatar_id(avatar_id)

        poses = self.asset_manager.list_assets(
            ASSET_AVATAR_POSE
        )

        pose_names = set()

        for asset in poses:
            asset_avatar_id = self._get_avatar_id(asset)

            if asset_avatar_id != avatar_id:
                continue

            if asset.pose:
                pose_names.add(asset.pose)

        return sorted(pose_names)

    # ------------------------------------------------------------------
    # Pose Existence
    # ------------------------------------------------------------------

    def has_pose(
        self,
        avatar_id: str,
        pose_name: str,
    ) -> bool:
        """
        Return True if the requested avatar pose exists.
        """

        avatar_id = self._normalize_avatar_id(avatar_id)
        pose_name = self._normalize_pose_name(pose_name)

        if not avatar_id or not pose_name:
            return False

        asset_id = self._build_asset_id(
            avatar_id,
            pose_name,
        )

        try:
            self.asset_manager.resolve(
                asset_id,
                asset_type=ASSET_AVATAR_POSE,
            )
            return True

        except (FileNotFoundError, RuntimeError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Pose Resolution
    # ------------------------------------------------------------------

    def get_pose(
        self,
        avatar_id: str,
        pose_name: str,
    ) -> ResolvedAsset:
        """
        Resolve a specific avatar pose.

        Example:

            get_pose(
                "scientist_male_2",
                "head_sideways",
            )

        Returns:
            ResolvedAsset for:

                scientist_male_2:head_sideways
        """

        avatar_id = self._normalize_avatar_id(avatar_id)
        pose_name = self._normalize_pose_name(pose_name)

        if not avatar_id:
            raise ValueError("avatar_id cannot be empty.")

        if not pose_name:
            raise ValueError("pose_name cannot be empty.")

        asset_id = self._build_asset_id(
            avatar_id,
            pose_name,
        )

        return self.asset_manager.resolve(
            asset_id,
            asset_type=ASSET_AVATAR_POSE,
        )

    # ------------------------------------------------------------------
    # Pose Map
    # ------------------------------------------------------------------

    def get_pose_map(
        self,
        avatar_id: str,
    ) -> Dict[str, ResolvedAsset]:
        """
        Return all available poses for an avatar.

        Example:

            {
                "default": ResolvedAsset(...),
                "head_sideways": ResolvedAsset(...),
            }
        """

        avatar_id = self._normalize_avatar_id(avatar_id)

        if not avatar_id:
            return {}

        poses = self.asset_manager.list_assets(
            ASSET_AVATAR_POSE
        )

        result: Dict[str, ResolvedAsset] = {}

        for asset in poses:
            asset_avatar_id = self._get_avatar_id(asset)

            if asset_avatar_id != avatar_id:
                continue

            if not asset.pose:
                continue

            result[asset.pose] = asset

        return dict(
            sorted(
                result.items(),
                key=lambda item: item[0].lower(),
            )
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_avatar_id(
        avatar_id: Optional[str],
    ) -> str:
        """
        Normalize an avatar ID.
        """

        if avatar_id is None:
            return ""

        return avatar_id.strip()

    @staticmethod
    def _normalize_pose_name(
        pose_name: Optional[str],
    ) -> str:
        """
        Normalize a pose name.
        """

        if pose_name is None:
            return ""

        return pose_name.strip()

    @staticmethod
    def _build_asset_id(
        avatar_id: str,
        pose_name: str,
    ) -> str:
        """
        Build the AssetManager pose ID.

        Existing Kestrel convention:

            avatar_id:pose_name
        """

        return f"{avatar_id}:{pose_name}"

    @staticmethod
    def _get_avatar_id(
        asset: ResolvedAsset,
    ) -> Optional[str]:
        """
        Extract the avatar ID from a discovered pose asset.

        AssetManager stores this in:

            asset.metadata["avatar_id"]
        """

        if not asset.metadata:
            return None

        avatar_id = asset.metadata.get("avatar_id")

        if not isinstance(avatar_id, str):
            return None

        avatar_id = avatar_id.strip()

        return avatar_id or None

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def print_poses(
        self,
        avatar_id: Optional[str] = None,
    ) -> None:
        """
        Print available avatar poses.

        If avatar_id is provided, only that avatar's poses are shown.
        """

        if avatar_id is not None:
            avatar_id = self._normalize_avatar_id(avatar_id)

            print(f"Avatar: {avatar_id}")

            poses = self.get_pose_map(avatar_id)

            if not poses:
                print("  (no poses found)")
                return

            for pose_name, asset in poses.items():
                print(
                    f"  {pose_name}"
                    f" -> {asset.path}"
                )

            return

        print("Avatar poses:")

        avatars = self.list_avatars()

        if not avatars:
            print("  (none)")
            return

        for current_avatar in avatars:
            poses = self.list_poses(current_avatar)

            print(f"  {current_avatar}")

            for pose_name in poses:
                print(f"    - {pose_name}")


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("KESTREL SHORTS — AVATAR POSE SYSTEM")
    print("=" * 70)

    manager = AssetManager()

    print("\nAsset Manager initialized.")
    print(f"Shorts root:")
    print(f"  {manager.get_root()}")

    print("\nDiscovering assets...")

    manager.discover_all()

    pose_system = AvatarPoseSystem(manager)

    print("\nAvatar Pose System initialized.")

    print("\nAvailable avatars with poses:")

    avatars = pose_system.list_avatars()

    if not avatars:
        print("  (none)")
    else:
        for avatar_id in avatars:
            print(f"  {avatar_id}")

    print("\nAvailable poses:")

    for avatar_id in avatars:
        poses = pose_system.list_poses(avatar_id)

        print(f"  {avatar_id}:")

        for pose_name in poses:
            print(f"    - {pose_name}")

    print("\n" + "=" * 70)
    print("AVATAR POSE SYSTEM READY")
    print("=" * 70)