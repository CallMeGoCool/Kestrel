"""
Kestrel Shorts — Asset Manager / Resolver

Purpose:
    Resolve semantic asset IDs into actual filesystem assets.

Architecture:

    Scene Plan / Director
            ↓
        asset_id
            ↓
      AssetManager
            ↓
      ResolvedAsset
            ↓
    AssetGeometry / Composition / Renderer

This module is intentionally AI-independent.

It does NOT:
    - call an LLM
    - call ComfyUI
    - generate assets
    - render assets
    - decide scene composition

It only manages and resolves assets that already exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Asset Types
# ---------------------------------------------------------------------------

ASSET_AVATAR = "avatar"
ASSET_AVATAR_POSE = "avatar_pose"
ASSET_VISUAL = "visual"
ASSET_BACKGROUND = "background"
ASSET_DIAGRAM = "diagram"
ASSET_ILLUSTRATION = "illustration"
ASSET_ICON = "icon"
ASSET_UNKNOWN = "unknown"


SUPPORTED_ASSET_TYPES = {
    ASSET_AVATAR,
    ASSET_AVATAR_POSE,
    ASSET_VISUAL,
    ASSET_BACKGROUND,
    ASSET_DIAGRAM,
    ASSET_ILLUSTRATION,
    ASSET_ICON,
}


# ---------------------------------------------------------------------------
# Resolved Asset
# ---------------------------------------------------------------------------

@dataclass
class ResolvedAsset:
    """
    Represents an asset after successful resolution.

    AssetManager owns identity and filesystem resolution.
    Geometry and composition systems operate on the resolved asset
    after this stage.
    """

    asset_id: str
    path: Path

    asset_type: str = ASSET_UNKNOWN

    category: Optional[str] = None
    pose: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        """Return True if the resolved filesystem path exists."""

        return self.path.exists()

    @property
    def filename(self) -> str:
        """Return the asset filename."""

        return self.path.name

    @property
    def extension(self) -> str:
        """Return the lowercase file extension."""

        return self.path.suffix.lower()

    def __str__(self) -> str:
        return str(self.path)


# ---------------------------------------------------------------------------
# Asset Manager
# ---------------------------------------------------------------------------

class AssetManager:
    """
    Resolve semantic asset IDs into actual filesystem assets.

    Responsibilities:

        - discover existing avatar assets
        - resolve canonical avatars
        - resolve avatar poses
        - resolve custom/generated visual assets
        - validate asset existence
        - expose semantic asset metadata
        - maintain an explicit runtime registry

    Non-responsibilities:

        - AI inference
        - asset generation
        - ComfyUI execution
        - composition decisions
        - geometry analysis
        - rendering
        - video creation
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
    }

    def __init__(
        self,
        shorts_root: Optional[Path | str] = None,
        avatar_root: Optional[Path | str] = None,
        custom_images_root: Optional[Path | str] = None,
    ):
        """
        Initialize the AssetManager.

        Parameters
        ----------
        shorts_root:
            Root directory of the Shorts application.

            Defaults to the directory containing this file.

        avatar_root:
            Optional explicit avatar root.

            Defaults to:
                <shorts_root>/avatars

        custom_images_root:
            Optional explicit custom/generated image root.

            Defaults to:
                <shorts_root>/custom_images

        The explicit root parameters make the manager easy to connect
        to the final Shorts configuration later without hard-coding
        application paths inside the asset subsystem.
        """

        if shorts_root is None:
            shorts_root = Path(__file__).resolve().parent.parent

        self.shorts_root = Path(shorts_root).resolve()

        if avatar_root is None:
            avatar_root = self.shorts_root / "avatars"

        if custom_images_root is None:
            custom_images_root = self.shorts_root / "custom_images"

        self.avatar_root = Path(avatar_root).resolve()
        self.custom_images_root = Path(custom_images_root).resolve()

        # Explicit runtime registry.
        #
        # Discovery and resolution may populate this registry.
        # The registry is NOT the filesystem source of truth.
        self._registry: Dict[str, ResolvedAsset] = {}

    # ------------------------------------------------------------------
    # Basic Properties
    # ------------------------------------------------------------------

    def get_root(self) -> Path:
        """Return the Shorts root directory."""

        return self.shorts_root

    def get_avatar_root(self) -> Path:
        """Return the avatar asset root."""

        return self.avatar_root

    def get_custom_images_root(self) -> Path:
        """Return the custom/generated image root."""

        return self.custom_images_root

    # ------------------------------------------------------------------
    # Internal Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_asset_type(asset_type: str) -> None:
        """
        Validate an asset type.

        ASSET_UNKNOWN is allowed because explicit registration may
        temporarily represent an asset whose semantic type is not
        known yet.
        """

        if asset_type == ASSET_UNKNOWN:
            return

        if asset_type not in SUPPORTED_ASSET_TYPES:
            raise ValueError(
                f"Unsupported asset type: '{asset_type}'. "
                f"Supported types: {sorted(SUPPORTED_ASSET_TYPES)}"
            )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        asset_id: str,
        path: Path | str,
        asset_type: str = ASSET_UNKNOWN,
        category: Optional[str] = None,
        pose: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResolvedAsset:
        """
        Register an asset explicitly.

        Registration does not require the file to exist.

        This is intentional because generated assets may be registered
        before a later stage validates or consumes them.
        """

        asset_id = asset_id.strip()

        if not asset_id:
            raise ValueError("asset_id cannot be empty.")

        self._validate_asset_type(asset_type)

        resolved_path = Path(path).resolve()

        asset = ResolvedAsset(
            asset_id=asset_id,
            path=resolved_path,
            asset_type=asset_type,
            category=category,
            pose=pose,
            metadata=dict(metadata or {}),
        )

        existing = self._registry.get(asset_id)

        if existing is not None:
            if existing.path != asset.path:
                raise RuntimeError(
                    f"Asset ID collision for '{asset_id}'.\n"
                    f"Existing: {existing.path}\n"
                    f"New:      {asset.path}"
                )

        self._registry[asset_id] = asset

        return asset

    def unregister(self, asset_id: str) -> bool:
        """
        Remove an asset from the runtime registry.

        Returns True if the asset existed in the registry.
        """

        return self._registry.pop(asset_id, None) is not None

    def clear_registry(self) -> None:
        """
        Clear the runtime registry.

        This does not delete any filesystem assets.
        """

        self._registry.clear()

    # ------------------------------------------------------------------
    # Registry Lookup
    # ------------------------------------------------------------------

    def get_registered(
        self,
        asset_id: str,
    ) -> Optional[ResolvedAsset]:
        """Return a registered asset if present."""

        return self._registry.get(asset_id)

    def registered_ids(self) -> List[str]:
        """Return registered asset IDs."""

        return sorted(self._registry.keys())

    # ------------------------------------------------------------------
    # Filesystem Discovery
    # ------------------------------------------------------------------

    def discover_images(
        self,
        root: Path | str,
    ) -> List[Path]:
        """
        Recursively discover image files under a directory.
        """

        root = Path(root).resolve()

        if not root.exists():
            return []

        if not root.is_dir():
            return []

        images: List[Path] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue

            images.append(path.resolve())

        return sorted(images)

    # ------------------------------------------------------------------
    # Avatar Discovery
    # ------------------------------------------------------------------

    def discover_avatars(self) -> List[ResolvedAsset]:
        """
        Discover canonical avatar assets.

        Only files inside a 'canonical' directory are considered
        canonical avatars.

        Example:

            avatars/
                science/
                    canonical/
                        scientist_female_1.png
        """

        if not self.avatar_root.exists():
            return []

        results: List[ResolvedAsset] = []

        for canonical_dir in sorted(
            self.avatar_root.rglob("canonical")
        ):
            if not canonical_dir.is_dir():
                continue

            category = canonical_dir.parent.name

            for path in sorted(canonical_dir.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                asset_id = path.stem

                asset = self.register(
                    asset_id=asset_id,
                    path=path,
                    asset_type=ASSET_AVATAR,
                    category=category,
                    metadata={
                        "source": "avatar_canonical",
                    },
                )

                results.append(asset)

        return results

    # ------------------------------------------------------------------
    # Avatar Pose Discovery
    # ------------------------------------------------------------------

    def discover_avatar_poses(self) -> List[ResolvedAsset]:
        """
        Discover avatar pose assets.

        Expected structure:

            avatars/
                science/
                    poses/
                        scientist_female_1/
                            scientist_female_1_default.png

        Semantic asset ID:

            scientist_female_1:default
        """

        if not self.avatar_root.exists():
            return []

        results: List[ResolvedAsset] = []

        for poses_dir in sorted(
            self.avatar_root.rglob("poses")
        ):
            if not poses_dir.is_dir():
                continue

            category = poses_dir.parent.name

            for avatar_dir in sorted(poses_dir.iterdir()):
                if not avatar_dir.is_dir():
                    continue

                avatar_id = avatar_dir.name

                for path in sorted(avatar_dir.iterdir()):
                    if not path.is_file():
                        continue

                    if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                        continue

                    pose_name = path.stem

                    # Remove the avatar prefix when possible.
                    prefix = f"{avatar_id}_"

                    if pose_name.startswith(prefix):
                        pose_name = pose_name[len(prefix):]

                    asset_id = f"{avatar_id}:{pose_name}"

                    asset = self.register(
                        asset_id=asset_id,
                        path=path,
                        asset_type=ASSET_AVATAR_POSE,
                        category=category,
                        pose=pose_name,
                        metadata={
                            "avatar_id": avatar_id,
                            "source": "avatar_pose",
                        },
                    )

                    results.append(asset)

        return results

    # ------------------------------------------------------------------
    # Custom Image Discovery
    # ------------------------------------------------------------------

    def discover_custom_images(self) -> List[ResolvedAsset]:
        """
        Discover custom/generated images.

        Expected structure:

            shorts/
                custom_images/
                    earth_visual.png
                    roman_empire_map.png
        """

        if not self.custom_images_root.exists():
            return []

        results: List[ResolvedAsset] = []

        for path in self.discover_images(
            self.custom_images_root
        ):
            asset_id = path.stem

            asset = self.register(
                asset_id=asset_id,
                path=path,
                asset_type=ASSET_VISUAL,
                category="custom",
                metadata={
                    "source": "custom_images",
                },
            )

            results.append(asset)

        return results

    # ------------------------------------------------------------------
    # Full Discovery
    # ------------------------------------------------------------------

    def discover_all(
        self,
    ) -> Dict[str, List[ResolvedAsset]]:
        """
        Discover all currently supported filesystem assets.

        The returned dictionary groups assets by semantic asset type.
        """

        avatars = self.discover_avatars()
        poses = self.discover_avatar_poses()
        custom_images = self.discover_custom_images()

        return {
            ASSET_AVATAR: avatars,
            ASSET_AVATAR_POSE: poses,
            ASSET_VISUAL: custom_images,
        }

    # ------------------------------------------------------------------
    # Direct Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        asset_id: str,
        asset_type: Optional[str] = None,
    ) -> ResolvedAsset:
        """
        Resolve an asset ID.

        Resolution order:

            1. Explicit runtime registry
            2. Canonical avatar
            3. Avatar pose
            4. Custom image

        Raises FileNotFoundError if no matching asset exists.
        """

        asset_id = asset_id.strip()

        if not asset_id:
            raise ValueError("asset_id cannot be empty.")

        if asset_type is not None:
            self._validate_asset_type(asset_type)

        # --------------------------------------------------------------
        # 1. Runtime registry
        # --------------------------------------------------------------

        registered = self._registry.get(asset_id)

        if registered is not None:
            type_matches = (
                asset_type is None
                or registered.asset_type == asset_type
            )

            if type_matches and registered.exists:
                return registered

        # --------------------------------------------------------------
        # 2. Canonical avatar
        # --------------------------------------------------------------

        if asset_type in (None, ASSET_AVATAR):
            avatar = self._resolve_avatar(asset_id)

            if avatar is not None:
                return avatar

        # --------------------------------------------------------------
        # 3. Avatar pose
        # --------------------------------------------------------------

        if asset_type in (None, ASSET_AVATAR_POSE):
            pose = self._resolve_pose(asset_id)

            if pose is not None:
                return pose

        # --------------------------------------------------------------
        # 4. Custom image
        # --------------------------------------------------------------

        if asset_type in (
            None,
            ASSET_VISUAL,
            ASSET_BACKGROUND,
            ASSET_DIAGRAM,
            ASSET_ILLUSTRATION,
            ASSET_ICON,
        ):
            custom = self._resolve_custom_image(
                asset_id,
                requested_type=asset_type,
            )

            if custom is not None:
                return custom

        raise FileNotFoundError(
            f"Asset not found: '{asset_id}'"
            + (
                f" (requested type: '{asset_type}')"
                if asset_type
                else ""
            )
        )

    # ------------------------------------------------------------------
    # Avatar Resolution
    # ------------------------------------------------------------------

    def _resolve_avatar(
        self,
        asset_id: str,
    ) -> Optional[ResolvedAsset]:
        """
        Resolve a canonical avatar by asset ID.
        """

        if not self.avatar_root.exists():
            return None

        matches: List[ResolvedAsset] = []

        for canonical_dir in sorted(
            self.avatar_root.rglob("canonical")
        ):
            if not canonical_dir.is_dir():
                continue

            for extension in self.IMAGE_EXTENSIONS:
                candidate = canonical_dir / f"{asset_id}{extension}"

                if not candidate.exists():
                    continue

                if not candidate.is_file():
                    continue

                category = canonical_dir.parent.name

                matches.append(
                    self.register(
                        asset_id=asset_id,
                        path=candidate,
                        asset_type=ASSET_AVATAR,
                        category=category,
                        metadata={
                            "source": "avatar_canonical",
                        },
                    )
                )

        if not matches:
            return None

        unique_paths = {
            asset.path
            for asset in matches
        }

        if len(unique_paths) > 1:
            raise RuntimeError(
                f"Multiple canonical avatar assets found for "
                f"'{asset_id}':\n"
                + "\n".join(
                    str(path)
                    for path in sorted(unique_paths)
                )
            )

        return matches[0]

    # ------------------------------------------------------------------
    # Pose Resolution
    # ------------------------------------------------------------------

    def _resolve_pose(
        self,
        asset_id: str,
    ) -> Optional[ResolvedAsset]:
        """
        Resolve an avatar pose.

        Expected ID:

            scientist_female_1:default
        """

        if ":" not in asset_id:
            return None

        avatar_id, pose_name = asset_id.split(":", 1)

        avatar_id = avatar_id.strip()
        pose_name = pose_name.strip()

        if not avatar_id or not pose_name:
            return None

        if not self.avatar_root.exists():
            return None

        matches: List[ResolvedAsset] = []

        for poses_dir in sorted(
            self.avatar_root.rglob("poses")
        ):
            if not poses_dir.is_dir():
                continue

            avatar_dir = poses_dir / avatar_id

            if not avatar_dir.exists():
                continue

            if not avatar_dir.is_dir():
                continue

            category = poses_dir.parent.name

            # ----------------------------------------------------------
            # Preferred filename:
            #
            # scientist_female_1_default.png
            # ----------------------------------------------------------

            preferred_matches = []

            for extension in self.IMAGE_EXTENSIONS:
                preferred = (
                    avatar_dir
                    / f"{avatar_id}_{pose_name}{extension}"
                )

                if preferred.exists() and preferred.is_file():
                    preferred_matches.append(preferred)

            for preferred in preferred_matches:
                matches.append(
                    self.register(
                        asset_id=asset_id,
                        path=preferred,
                        asset_type=ASSET_AVATAR_POSE,
                        category=category,
                        pose=pose_name,
                        metadata={
                            "avatar_id": avatar_id,
                            "source": "avatar_pose",
                        },
                    )
                )

            if preferred_matches:
                continue

            # ----------------------------------------------------------
            # Fallback:
            #
            # Search for a file whose stem is exactly the pose name.
            # ----------------------------------------------------------

            for path in sorted(avatar_dir.iterdir()):
                if not path.is_file():
                    continue

                if path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                if path.stem != pose_name:
                    continue

                matches.append(
                    self.register(
                        asset_id=asset_id,
                        path=path,
                        asset_type=ASSET_AVATAR_POSE,
                        category=category,
                        pose=pose_name,
                        metadata={
                            "avatar_id": avatar_id,
                            "source": "avatar_pose",
                        },
                    )
                )

        if not matches:
            return None

        unique_paths = {
            asset.path
            for asset in matches
        }

        if len(unique_paths) > 1:
            raise RuntimeError(
                f"Multiple avatar poses found for '{asset_id}':\n"
                + "\n".join(
                    str(path)
                    for path in sorted(unique_paths)
                )
            )

        return matches[0]

    # ------------------------------------------------------------------
    # Custom Image Resolution
    # ------------------------------------------------------------------

    def _resolve_custom_image(
        self,
        asset_id: str,
        requested_type: Optional[str] = None,
    ) -> Optional[ResolvedAsset]:
        """
        Resolve a custom/generated image by asset ID.

        If a semantic type is explicitly requested, preserve that
        requested type in the resulting ResolvedAsset.
        """

        if not self.custom_images_root.exists():
            return None

        matches: List[Path] = []

        for extension in self.IMAGE_EXTENSIONS:
            candidate = (
                self.custom_images_root
                / f"{asset_id}{extension}"
            )

            if candidate.exists() and candidate.is_file():
                matches.append(candidate)

        if not matches:
            return None

        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple custom images found for '{asset_id}':\n"
                + "\n".join(
                    str(path)
                    for path in sorted(matches)
                )
            )

        resolved_type = (
            requested_type
            if requested_type is not None
            else ASSET_VISUAL
        )

        return self.register(
            asset_id=asset_id,
            path=matches[0],
            asset_type=resolved_type,
            category="custom",
            metadata={
                "source": "custom_images",
            },
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        asset_id: str,
        asset_type: Optional[str] = None,
    ) -> bool:
        """
        Return True if an asset can be resolved successfully.
        """

        try:
            self.resolve(
                asset_id,
                asset_type=asset_type,
            )
            return True

        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ):
            return False

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_assets(
        self,
        asset_type: Optional[str] = None,
    ) -> List[ResolvedAsset]:
        """
        Return all currently registered assets.

        If asset_type is provided, filter by type.
        """

        if asset_type is not None:
            self._validate_asset_type(asset_type)

        assets = list(self._registry.values())

        if asset_type is not None:
            assets = [
                asset
                for asset in assets
                if asset.asset_type == asset_type
            ]

        return sorted(
            assets,
            key=lambda asset: asset.asset_id.lower(),
        )

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------

    def print_assets(
        self,
        asset_type: Optional[str] = None,
    ) -> None:
        """
        Print currently registered assets.
        """

        assets = self.list_assets(asset_type)

        print("Registered assets:")

        if not assets:
            print("  (none)")
            return

        for asset in assets:
            print(
                f"  {asset.asset_id}"
                f" [{asset.asset_type}]"
                f" -> {asset.path}"
            )


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("KESTREL SHORTS — ASSET MANAGER")
    print("=" * 70)

    manager = AssetManager()

    print("\nAsset Manager initialized.")

    print("\nShorts root:")
    print(f"  {manager.get_root()}")

    print("\nAvatar root:")
    print(f"  {manager.get_avatar_root()}")

    print("\nCustom image root:")
    print(f"  {manager.get_custom_images_root()}")

    print("\nDiscovering assets...")

    discovered = manager.discover_all()

    for asset_type, assets in discovered.items():
        print(f"\n{asset_type}: {len(assets)}")

        for asset in assets:
            print(f"  {asset.asset_id}")
            print(f"    {asset.path}")

    print("\nRegistered asset count:")
    print(f"  {len(manager.list_assets())}")

    print("\n" + "=" * 70)
    print("ASSET MANAGER READY")
    print("=" * 70)