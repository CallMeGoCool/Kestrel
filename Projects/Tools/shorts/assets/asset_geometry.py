"""
Kestrel Shorts — Asset Geometry

Purpose:
    Analyze resolved image assets and produce geometry information that
    can be consumed by the LayoutEngine.

Architecture:

    AssetManager
          ↓
    ResolvedAsset
          ↓
    AssetGeometryAnalyzer
          ↓
    AssetGeometry
          ↓
    LayoutEngine

This module is deterministic.

It does NOT:
    - call an LLM
    - call ComfyUI
    - generate assets
    - decide composition
    - render video
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from shorts.assets.asset_manager import (
    AssetManager,
    ResolvedAsset,
    ASSET_AVATAR,
    ASSET_AVATAR_POSE,
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class GeometryRect:
    """
    Simple rectangle used for asset geometry measurements.
    """

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def __str__(self) -> str:
        return (
            f"Rect("
            f"x={self.x:.1f}, "
            f"y={self.y:.1f}, "
            f"w={self.width:.1f}, "
            f"h={self.height:.1f}"
            f")"
        )


# ---------------------------------------------------------------------------
# Asset Geometry
# ---------------------------------------------------------------------------

@dataclass
class AssetGeometry:
    """
    Complete geometry information for an image asset.

    canvas_width / canvas_height:
        Dimensions of the actual image file.

    content_bounds:
        Bounding box of visible/non-transparent content inside the image.

    The distinction is important because an avatar may contain significant
    transparent padding around the visible character.
    """

    asset_id: str
    path: Path

    canvas_width: int
    canvas_height: int

    content_bounds: GeometryRect

    asset_type: str = "unknown"
    category: Optional[str] = None
    pose: Optional[str] = None

    has_alpha: bool = False
    alpha_threshold: int = 1

    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    @property
    def canvas_area(self) -> int:
        return self.canvas_width * self.canvas_height

    @property
    def content_area(self) -> float:
        return self.content_bounds.area

    @property
    def content_width(self) -> float:
        return self.content_bounds.width

    @property
    def content_height(self) -> float:
        return self.content_bounds.height

    @property
    def content_ratio(self) -> float:
        if self.canvas_area == 0:
            return 0.0

        return self.content_area / self.canvas_area

    @property
    def content_rect(self) -> GeometryRect:
        return self.content_bounds

    @property
    def canvas_rect(self) -> GeometryRect:
        return GeometryRect(
            x=0,
            y=0,
            width=self.canvas_width,
            height=self.canvas_height,
        )

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_for_content_height(
        self,
        desired_height: float,
    ) -> float:
        """
        Return the scale required to make the visible content
        reach the requested height.
        """

        if self.content_height <= 0:
            raise ValueError(
                f"Asset '{self.asset_id}' has no usable content height."
            )

        if desired_height <= 0:
            raise ValueError(
                "desired_height must be greater than zero."
            )

        return desired_height / self.content_height

    def scaled_content_rect(
        self,
        scale: float,
    ) -> GeometryRect:
        """
        Return the visible content bounds after applying scale.
        """

        if scale <= 0:
            raise ValueError(
                "scale must be greater than zero."
            )

        return GeometryRect(
            x=self.content_bounds.x * scale,
            y=self.content_bounds.y * scale,
            width=self.content_bounds.width * scale,
            height=self.content_bounds.height * scale,
        )

    def rendered_canvas_size(
        self,
        scale: float,
    ) -> tuple[float, float]:
        """
        Return the full image canvas dimensions after applying scale.

        This includes transparent padding.
        """

        if scale <= 0:
            raise ValueError(
                "scale must be greater than zero."
            )

        return (
            self.canvas_width * scale,
            self.canvas_height * scale,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> str:
        return (
            f"AssetGeometry("
            f"id={self.asset_id}, "
            f"canvas={self.canvas_width}x{self.canvas_height}, "
            f"content={self.content_bounds}, "
            f"type={self.asset_type}"
            f")"
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class AssetGeometryAnalyzer:
    """
    Analyze image files and return AssetGeometry objects.
    """

    def __init__(
        self,
        alpha_threshold: int = 1,
    ):
        """
        alpha_threshold controls which alpha values count as visible.

        Default:
            alpha > 0 = visible
        """

        if not 0 <= alpha_threshold <= 255:
            raise ValueError(
                "alpha_threshold must be between 0 and 255."
            )

        self.alpha_threshold = alpha_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        asset: ResolvedAsset,
    ) -> AssetGeometry:
        """
        Analyze a resolved asset.
        """

        if not asset.path.exists():
            raise FileNotFoundError(
                f"Asset file does not exist: {asset.path}"
            )

        if not asset.path.is_file():
            raise FileNotFoundError(
                f"Asset path is not a file: {asset.path}"
            )

        with Image.open(asset.path) as image:
            return self._analyze_image(
                image=image,
                asset=asset,
            )

    def analyze_path(
        self,
        path: Path | str,
        asset_id: Optional[str] = None,
        asset_type: str = "unknown",
        category: Optional[str] = None,
        pose: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssetGeometry:
        """
        Analyze an image directly from a filesystem path.

        Useful for testing or for assets that have not yet been registered
        with AssetManager.
        """

        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Image path is not a file: {path}"
            )

        if asset_id is None:
            asset_id = path.stem

        with Image.open(path) as image:
            asset = ResolvedAsset(
                asset_id=asset_id,
                path=path,
                asset_type=asset_type,
                category=category,
                pose=pose,
                metadata=metadata or {},
            )

            return self._analyze_image(
                image=image,
                asset=asset,
            )

    # ------------------------------------------------------------------
    # Image Analysis
    # ------------------------------------------------------------------

    def _analyze_image(
        self,
        image: Image.Image,
        asset: ResolvedAsset,
    ) -> AssetGeometry:
        """
        Perform the actual image measurement.
        """

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Asset '{asset.asset_id}' has invalid image dimensions: "
                f"{width}x{height}"
            )

        has_alpha = "A" in image.getbands()

        if has_alpha:
            content_bounds = self._find_alpha_bounds(
                image=image,
            )
        else:
            # Opaque image:
            #
            # The entire image is considered visible content.
            content_bounds = GeometryRect(
                x=0,
                y=0,
                width=float(width),
                height=float(height),
            )

        return AssetGeometry(
            asset_id=asset.asset_id,
            path=asset.path,
            canvas_width=width,
            canvas_height=height,
            content_bounds=content_bounds,
            asset_type=asset.asset_type,
            category=asset.category,
            pose=asset.pose,
            has_alpha=has_alpha,
            alpha_threshold=self.alpha_threshold,
            metadata=dict(asset.metadata),
        )

    # ------------------------------------------------------------------
    # Alpha Bounds
    # ------------------------------------------------------------------

    def _find_alpha_bounds(
        self,
        image: Image.Image,
    ) -> GeometryRect:
        """
        Find the smallest rectangle containing all pixels whose alpha
        value is greater than or equal to alpha_threshold.
        """

        alpha = image.getchannel("A")

        if self.alpha_threshold <= 1:
            bbox = alpha.getbbox()

        else:
            alpha = alpha.point(
                lambda value: 255
                if value >= self.alpha_threshold
                else 0
            )

            bbox = alpha.getbbox()

        if bbox is None:
            # Entire image is transparent.
            return GeometryRect(
                x=0,
                y=0,
                width=0,
                height=0,
            )

        left, top, right, bottom = bbox

        return GeometryRect(
            x=float(left),
            y=float(top),
            width=float(right - left),
            height=float(bottom - top),
        )


# ---------------------------------------------------------------------------
# Asset Geometry Resolver
# ---------------------------------------------------------------------------

class AssetGeometryResolver:
    """
    Convenience layer combining AssetManager + AssetGeometryAnalyzer.

    This lets callers ask:

        resolver.resolve("scientist_female_1")

    and receive measured geometry for the asset.
    """

    def __init__(
        self,
        asset_manager: Optional[AssetManager] = None,
        analyzer: Optional[AssetGeometryAnalyzer] = None,
    ):
        self.asset_manager = asset_manager or AssetManager()
        self.analyzer = analyzer or AssetGeometryAnalyzer()

    def resolve(
        self,
        asset_id: str,
        asset_type: Optional[str] = None,
    ) -> AssetGeometry:
        """
        Resolve an asset and analyze its geometry.
        """

        asset = self.asset_manager.resolve(
            asset_id=asset_id,
            asset_type=asset_type,
        )

        return self.analyzer.analyze(asset)

    def resolve_path(
        self,
        path: Path | str,
        asset_id: Optional[str] = None,
        asset_type: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssetGeometry:
        """
        Analyze a direct filesystem path.
        """

        return self.analyzer.analyze_path(
            path=path,
            asset_id=asset_id,
            asset_type=asset_type,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Standalone Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("KESTREL SHORTS — ASSET GEOMETRY")
    print("=" * 70)

    manager = AssetManager()
    analyzer = AssetGeometryAnalyzer()
    resolver = AssetGeometryResolver(
        asset_manager=manager,
        analyzer=analyzer,
    )

    print("\nAsset Geometry system initialized.")

    # --------------------------------------------------------------
    # Test 1 — Scientist
    # --------------------------------------------------------------

    print("\nTesting scientist_female_1...")

    scientist = resolver.resolve(
        "scientist_female_1",
        asset_type=ASSET_AVATAR,
    )

    print(f"  Asset ID:       {scientist.asset_id}")
    print(f"  Type:           {scientist.asset_type}")
    print(f"  Category:       {scientist.category}")
    print(f"  File:           {scientist.path}")
    print(
        f"  Canvas:         "
        f"{scientist.canvas_width}x{scientist.canvas_height}"
    )
    print(f"  Content bounds: {scientist.content_bounds}")
    print(
        f"  Content ratio:  "
        f"{scientist.content_ratio:.3f}"
    )
    print(f"  Has alpha:      {scientist.has_alpha}")
    print(f"  Metadata:       {scientist.metadata}")

    # --------------------------------------------------------------
    # Test 2 — Scientist pose
    # --------------------------------------------------------------

    print("\nTesting scientist_male_2:head_sideways...")

    pose = resolver.resolve(
        "scientist_male_2:head_sideways",
        asset_type=ASSET_AVATAR_POSE,
    )

    print(f"  Asset ID:       {pose.asset_id}")
    print(f"  Type:           {pose.asset_type}")
    print(f"  Category:       {pose.category}")
    print(f"  Pose:           {pose.pose}")
    print(f"  File:           {pose.path}")
    print(
        f"  Canvas:         "
        f"{pose.canvas_width}x{pose.canvas_height}"
    )
    print(f"  Content bounds: {pose.content_bounds}")
    print(
        f"  Content ratio:  "
        f"{pose.content_ratio:.3f}"
    )
    print(f"  Has alpha:      {pose.has_alpha}")
    print(f"  Metadata:       {pose.metadata}")

    # --------------------------------------------------------------
    # Test 3 — Scaling
    # --------------------------------------------------------------

    print("\nTesting scaling...")

    desired_height = 900.0
    scientist_scale = scientist.scale_for_content_height(
        desired_height
    )

    scaled_rect = scientist.scaled_content_rect(
        scientist_scale
    )

    canvas_size = scientist.rendered_canvas_size(
        scientist_scale
    )

    print(f"  Desired content height: {desired_height:.1f}")
    print(f"  Scale:                  {scientist_scale:.4f}")
    print(f"  Scaled content rect:    {scaled_rect}")
    print(
        f"  Rendered canvas size:   "
        f"{canvas_size[0]:.1f} x {canvas_size[1]:.1f}"
    )

    # --------------------------------------------------------------
    # Test 4 — Summary
    # --------------------------------------------------------------

    print("\nGeometry summary:")

    print(
        f"  Scientist visible area: "
        f"{scientist.content_width:.1f} x "
        f"{scientist.content_height:.1f}"
    )

    print(
        f"  Scientist canvas: "
        f"{scientist.canvas_width} x "
        f"{scientist.canvas_height}"
    )

    print(
        f"  Pose visible area: "
        f"{pose.content_width:.1f} x "
        f"{pose.content_height:.1f}"
    )

    print(
        f"  Pose canvas: "
        f"{pose.canvas_width} x "
        f"{pose.canvas_height}"
    )

    print("\n" + "=" * 70)
    print("ASSET GEOMETRY READY")
    print("=" * 70)