"""
Kestrel Shorts — Layout Engine
==============================

Purpose
-------
The Layout Engine is the spatial authority for Kestrel Shorts.

It does NOT render images or text.

It answers questions such as:

    - Where should this object go?
    - How large is its visible area?
    - Does it collide with another object?
    - Is it inside the canvas?
    - Which semantic region does it occupy?
    - How much free space remains?

The engine treats different visual elements as spatial objects:

    Avatar
    Text
    Image
    Diagram
    Shape
    Overlay
    Caption
    ...

All of them eventually become LayoutObjects.

Important architectural principle
----------------------------------
Upstream intelligence describes intent:

    "Put the scientist in the lower-right."

    "Put MONEY in the upper-left."

The Layout Engine converts that intent into actual pixel geometry.

The LLM / Architect / Director should NOT need to calculate pixels.

Typography integration
----------------------
The Typography Engine is responsible for:

    - text content
    - font selection
    - styling
    - text measurement
    - text rendering

The Layout Engine is responsible for:

    - placement
    - spatial geometry
    - collision detection
    - canvas validation
    - composition relationships

The two systems communicate through the TextObject's measured rectangle.

Rendering remains outside this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920

GRID_COLUMNS = 24
GRID_ROWS = 32


# ---------------------------------------------------------------------------
# Rectangle
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    """
    Pixel-space rectangle.

    x, y:
        Top-left position.

    width, height:
        Size of the rectangle.
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
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, other: "Rect") -> bool:
        """
        Return True if this rectangle completely contains another rectangle.
        """

        return (
            other.x >= self.x
            and other.y >= self.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def intersects(self, other: "Rect") -> bool:
        """
        Return True when two rectangles overlap.
        """

        return not (
            other.right <= self.x
            or other.x >= self.right
            or other.bottom <= self.y
            or other.y >= self.bottom
        )

    def intersection(
        self,
        other: "Rect",
    ) -> Optional["Rect"]:
        """
        Return the overlapping rectangle.

        Returns None when there is no overlap.
        """

        if not self.intersects(other):
            return None

        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.right, other.right)
        y2 = min(self.bottom, other.bottom)

        return Rect(
            x=x1,
            y=y1,
            width=x2 - x1,
            height=y2 - y1,
        )

    def inset(
        self,
        amount: float,
    ) -> "Rect":
        """
        Return a rectangle inset on all sides.
        """

        return Rect(
            x=self.x + amount,
            y=self.y + amount,
            width=max(0, self.width - amount * 2),
            height=max(0, self.height - amount * 2),
        )

    def expand(
        self,
        amount: float,
    ) -> "Rect":
        """
        Return a rectangle expanded on all sides.
        """

        return Rect(
            x=self.x - amount,
            y=self.y - amount,
            width=self.width + amount * 2,
            height=self.height + amount * 2,
        )

    def translated(
        self,
        dx: float,
        dy: float,
    ) -> "Rect":
        """
        Return a copy moved by dx/dy.
        """

        return Rect(
            x=self.x + dx,
            y=self.y + dy,
            width=self.width,
            height=self.height,
        )

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
    Describes the relationship between an asset's full PNG canvas and its
    actual visible content.

    This is important for Kestrel avatars because the canonical PNGs contain
    transparent padding.
    """

    canvas_width: float
    canvas_height: float

    content_x: float
    content_y: float

    content_width: float
    content_height: float

    @property
    def content_right(self) -> float:
        return self.content_x + self.content_width

    @property
    def content_bottom(self) -> float:
        return self.content_y + self.content_height

    @property
    def content_rect(self) -> Rect:
        return Rect(
            self.content_x,
            self.content_y,
            self.content_width,
            self.content_height,
        )

    def scale_for_content_width(
        self,
        desired_width: float,
    ) -> float:
        """
        Calculate scale required to make visible content the requested width.
        """

        if self.content_width <= 0:
            raise ValueError(
                "content_width must be greater than zero"
            )

        return desired_width / self.content_width

    def scale_for_content_height(
        self,
        desired_height: float,
    ) -> float:
        """
        Calculate scale required to make visible content the requested height.
        """

        if self.content_height <= 0:
            raise ValueError(
                "content_height must be greater than zero"
            )

        return desired_height / self.content_height

    def rendered_canvas_size(
        self,
        scale: float,
    ) -> Tuple[float, float]:
        """
        Return the full PNG size after scaling.
        """

        return (
            self.canvas_width * scale,
            self.canvas_height * scale,
        )

    def scaled_content_rect(
        self,
        scale: float,
    ) -> Rect:
        """
        Return the visible content rectangle after scaling.
        """

        return Rect(
            x=self.content_x * scale,
            y=self.content_y * scale,
            width=self.content_width * scale,
            height=self.content_height * scale,
        )


# ---------------------------------------------------------------------------
# Layout Object
# ---------------------------------------------------------------------------

@dataclass
class LayoutObject:
    """
    Generic spatial representation of a visual object.
    """

    name: str
    object_type: str

    rect: Rect

    anchor: Optional[str] = None

    canvas_rect: Optional[Rect] = None

    asset_geometry: Optional[AssetGeometry] = None

    scale: Optional[float] = None

    z_index: int = 0

    role: Optional[str] = None

    start_time: Optional[float] = None
    end_time: Optional[float] = None

    metadata: Dict[str, object] = field(
        default_factory=dict
    )

    @property
    def right(self) -> float:
        return self.rect.right

    @property
    def bottom(self) -> float:
        return self.rect.bottom

    @property
    def center_x(self) -> float:
        return self.rect.center_x

    @property
    def center_y(self) -> float:
        return self.rect.center_y

    def intersects(
        self,
        other: "LayoutObject",
    ) -> bool:
        return self.rect.intersects(
            other.rect
        )

    def is_inside(
        self,
        canvas_width: int = CANVAS_WIDTH,
        canvas_height: int = CANVAS_HEIGHT,
    ) -> bool:
        canvas = Rect(
            0,
            0,
            canvas_width,
            canvas_height,
        )

        return canvas.contains(
            self.rect
        )

    def __str__(self) -> str:
        return (
            f"{self.name} "
            f"[{self.object_type}] "
            f"{self.rect}"
        )


# ---------------------------------------------------------------------------
# Layout Engine
# ---------------------------------------------------------------------------

class LayoutEngine:
    """
    Central spatial authority for Kestrel Shorts.
    """

    def __init__(
        self,
        canvas_width: int = CANVAS_WIDTH,
        canvas_height: int = CANVAS_HEIGHT,
        grid_columns: int = GRID_COLUMNS,
        grid_rows: int = GRID_ROWS,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.grid_columns = grid_columns
        self.grid_rows = grid_rows

        self.cell_width = (
            canvas_width / grid_columns
        )

        self.cell_height = (
            canvas_height / grid_rows
        )

        self.objects: Dict[
            str,
            LayoutObject,
        ] = {}

    # ------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------

    @property
    def canvas_rect(self) -> Rect:
        return Rect(
            0,
            0,
            self.canvas_width,
            self.canvas_height,
        )

    # ------------------------------------------------------------------
    # Grid conversion
    # ------------------------------------------------------------------

    def grid_to_pixels(
        self,
        column: int,
        row: int,
        column_span: int = 1,
        row_span: int = 1,
    ) -> Rect:
        """
        Convert logical grid coordinates into pixel geometry.
        """

        return Rect(
            x=column * self.cell_width,
            y=row * self.cell_height,
            width=column_span * self.cell_width,
            height=row_span * self.cell_height,
        )

    def pixels_to_grid(
        self,
        rect: Rect,
    ) -> Dict[str, int]:
        """
        Approximate a pixel rectangle using grid coordinates.

        This is descriptive, not restrictive.
        """

        column = round(
            rect.x / self.cell_width
        )

        row = round(
            rect.y / self.cell_height
        )

        column_span = max(
            1,
            round(
                rect.width
                / self.cell_width
            ),
        )

        row_span = max(
            1,
            round(
                rect.height
                / self.cell_height
            ),
        )

        return {
            "column": column,
            "row": row,
            "column_span": column_span,
            "row_span": row_span,
        }

    # ------------------------------------------------------------------
    # Semantic anchors
    # ------------------------------------------------------------------

    def anchor_position(
        self,
        anchor: str,
        object_width: float,
        object_height: float,
        margin: float = 60,
    ) -> Tuple[float, float]:
        """
        Calculate a pixel position for a semantic anchor.
        """

        if anchor == "top_left":
            return (
                margin,
                margin,
            )

        if anchor == "top_center":
            return (
                (self.canvas_width - object_width) / 2,
                margin,
            )

        if anchor == "top_right":
            return (
                self.canvas_width
                - object_width
                - margin,
                margin,
            )

        if anchor == "upper_left":
            return (
                margin,
                self.canvas_height * 0.25
                - object_height / 2,
            )

        if anchor == "upper_center":
            return (
                (self.canvas_width - object_width) / 2,
                self.canvas_height * 0.25
                - object_height / 2,
            )

        if anchor == "upper_right":
            return (
                self.canvas_width
                - object_width
                - margin,
                self.canvas_height * 0.25
                - object_height / 2,
            )

        if anchor == "center_left":
            return (
                margin,
                (self.canvas_height - object_height) / 2,
            )

        if anchor == "center":
            return (
                (self.canvas_width - object_width) / 2,
                (self.canvas_height - object_height) / 2,
            )

        if anchor == "center_right":
            return (
                self.canvas_width
                - object_width
                - margin,
                (self.canvas_height - object_height) / 2,
            )

        if anchor == "lower_left":
            return (
                margin,
                self.canvas_height * 0.75
                - object_height / 2,
            )

        if anchor == "lower_center":
            return (
                (self.canvas_width - object_width) / 2,
                self.canvas_height * 0.75
                - object_height / 2,
            )

        if anchor == "lower_right":
            return (
                self.canvas_width
                - object_width
                - margin,
                self.canvas_height * 0.75
                - object_height / 2,
            )

        if anchor == "bottom_left":
            return (
                margin,
                self.canvas_height
                - object_height
                - margin,
            )

        if anchor == "bottom_center":
            return (
                (self.canvas_width - object_width) / 2,
                self.canvas_height
                - object_height
                - margin,
            )

        if anchor == "bottom_right":
            return (
                self.canvas_width
                - object_width
                - margin,
                self.canvas_height
                - object_height
                - margin,
            )

        raise ValueError(
            f"Unknown anchor: {anchor}"
        )

    # ------------------------------------------------------------------
    # Generic object placement
    # ------------------------------------------------------------------

    def create_object(
        self,
        name: str,
        object_type: str,
        width: float,
        height: float,
        anchor: Optional[str] = None,
        margin: float = 60,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z_index: int = 0,
        role: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> LayoutObject:
        """
        Create a generic visual object.
        """

        if width <= 0:
            raise ValueError(
                "width must be greater than zero"
            )

        if height <= 0:
            raise ValueError(
                "height must be greater than zero"
            )

        if anchor is not None:
            anchor_x, anchor_y = (
                self.anchor_position(
                    anchor=anchor,
                    object_width=width,
                    object_height=height,
                    margin=margin,
                )
            )

            if x is None:
                x = anchor_x

            if y is None:
                y = anchor_y

        if x is None:
            x = 0

        if y is None:
            y = 0

        obj = LayoutObject(
            name=name,
            object_type=object_type,
            rect=Rect(
                x=x,
                y=y,
                width=width,
                height=height,
            ),
            anchor=anchor,
            z_index=z_index,
            role=role,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {},
        )

        return obj

    # ------------------------------------------------------------------
    # Add object
    # ------------------------------------------------------------------

    def add_object(
        self,
        obj: LayoutObject,
        check_collision: bool = False,
        allow_outside_canvas: bool = False,
    ) -> LayoutObject:
        """
        Register an object in the layout.

        Collision checking is optional because sometimes overlap is
        intentional.
        """

        if obj.name in self.objects:
            raise ValueError(
                f"Object already exists: {obj.name}"
            )

        if (
            not allow_outside_canvas
            and not obj.is_inside(
                self.canvas_width,
                self.canvas_height,
            )
        ):
            raise ValueError(
                f"Object is outside canvas: {obj}"
            )

        if check_collision:
            collisions = self.check_collision(
                obj,
                ignore_name=None,
            )

            if collisions:
                names = ", ".join(
                    item.name
                    for item in collisions
                )

                raise ValueError(
                    f"Object '{obj.name}' "
                    f"collides with: {names}"
                )

        self.objects[obj.name] = obj

        return obj

    # ------------------------------------------------------------------
    # Generic anchored object
    # ------------------------------------------------------------------

    def add_at_anchor(
        self,
        name: str,
        object_type: str,
        width: float,
        height: float,
        anchor: str,
        margin: float = 60,
        z_index: int = 0,
        role: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
        check_collision: bool = False,
    ) -> LayoutObject:
        """
        Create and register a generic object at an anchor.
        """

        obj = self.create_object(
            name=name,
            object_type=object_type,
            width=width,
            height=height,
            anchor=anchor,
            margin=margin,
            z_index=z_index,
            role=role,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
        )

        return self.add_object(
            obj,
            check_collision=check_collision,
        )

    # ------------------------------------------------------------------
    # Avatar placement
    # ------------------------------------------------------------------

    def add_avatar(
        self,
        name: str,
        geometry: AssetGeometry,
        anchor: str,
        desired_width: Optional[float] = None,
        desired_height: Optional[float] = None,
        margin: float = 60,
        z_index: int = 10,
        role: str = "avatar",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
        check_collision: bool = False,
    ) -> LayoutObject:
        """
        Add an avatar using visible-content geometry.

        IMPORTANT:
            desired_width / desired_height refers to the visible character,
            NOT the full transparent PNG.
        """

        if (
            desired_width is None
            and desired_height is None
        ):
            raise ValueError(
                "Provide desired_width or desired_height"
            )

        if (
            desired_width is not None
            and desired_height is not None
        ):
            raise ValueError(
                "Provide only desired_width "
                "or desired_height"
            )

        if desired_width is not None:
            scale = geometry.scale_for_content_width(
                desired_width
            )
        else:
            scale = geometry.scale_for_content_height(
                desired_height
            )

        visible_rect_local = (
            geometry.scaled_content_rect(
                scale
            )
        )

        canvas_width, canvas_height = (
            geometry.rendered_canvas_size(
                scale
            )
        )

        # Find where the visible character should go.
        visible_x, visible_y = (
            self.anchor_position(
                anchor=anchor,
                object_width=visible_rect_local.width,
                object_height=visible_rect_local.height,
                margin=margin,
            )
        )

        # Convert visible-content position into the position of the
        # complete PNG canvas.
        canvas_x = (
            visible_x
            - visible_rect_local.x
        )

        canvas_y = (
            visible_y
            - visible_rect_local.y
        )

        visible_rect = Rect(
            x=visible_x,
            y=visible_y,
            width=visible_rect_local.width,
            height=visible_rect_local.height,
        )

        canvas_rect = Rect(
            x=canvas_x,
            y=canvas_y,
            width=canvas_width,
            height=canvas_height,
        )

        obj = LayoutObject(
            name=name,
            object_type="avatar",
            rect=visible_rect,
            anchor=anchor,
            canvas_rect=canvas_rect,
            asset_geometry=geometry,
            scale=scale,
            z_index=z_index,
            role=role,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {},
        )

        return self.add_object(
            obj,
            check_collision=check_collision,
        )

    # ------------------------------------------------------------------
    # Text placement
    # ------------------------------------------------------------------

    def add_text_object(
        self,
        name: str,
        width: float,
        height: float,
        anchor: str,
        margin: float = 60,
        role: str = "attention",
        z_index: int = 20,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
        check_collision: bool = False,
    ) -> LayoutObject:
        """
        Register text geometry with the layout engine.

        The Typography Engine remains responsible for measuring the text.

        This method is useful when a caller only has measured dimensions.

        For a real TypographyEngine.TextObject, use:

            add_typography_object(...)
        """

        obj = self.create_object(
            name=name,
            object_type="text",
            width=width,
            height=height,
            anchor=anchor,
            margin=margin,
            z_index=z_index,
            role=role,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
        )

        return self.add_object(
            obj,
            check_collision=check_collision,
        )

    # ------------------------------------------------------------------
    # Typography Engine integration
    # ------------------------------------------------------------------

    def add_typography_object(
        self,
        text_object,
        anchor: Optional[str] = None,
        margin: float = 60,
        z_index: Optional[int] = None,
        role: Optional[str] = None,
        check_collision: bool = False,
    ) -> LayoutObject:
        """
        Register an actual TypographyEngine.TextObject.

        This method intentionally does NOT import TypographyEngine.

        The Layout Engine only requires the TextObject to provide:

            - name
            - rect.width
            - rect.height

        Optional properties are read when available:

            - anchor
            - role
            - z_index
            - start_time
            - end_time
            - metadata

        The Typography Engine remains responsible for measuring and
        rendering the text.

        The Layout Engine becomes responsible for its final position.
        """

        if text_object is None:
            raise ValueError(
                "text_object cannot be None"
            )

        if not hasattr(text_object, "rect"):
            raise TypeError(
                "text_object must provide a 'rect' attribute"
            )

        text_rect = text_object.rect

        if text_rect is None:
            raise ValueError(
                "text_object.rect cannot be None"
            )

        width = float(text_rect.width)
        height = float(text_rect.height)

        if width <= 0:
            raise ValueError(
                "TextObject width must be greater than zero"
            )

        if height <= 0:
            raise ValueError(
                "TextObject height must be greater than zero"
            )

        object_name = getattr(
            text_object,
            "name",
            "text_object",
        )

        selected_anchor = (
            anchor
            if anchor is not None
            else getattr(
                text_object,
                "anchor",
                None,
            )
        )

        if selected_anchor is None:
            raise ValueError(
                "Provide an anchor either to add_typography_object() "
                "or on the TextObject itself"
            )

        selected_role = (
            role
            if role is not None
            else getattr(
                text_object,
                "role",
                "attention",
            )
        )

        selected_z_index = (
            z_index
            if z_index is not None
            else getattr(
                text_object,
                "z_index",
                20,
            )
        )

        start_time = getattr(
            text_object,
            "start_time",
            None,
        )

        end_time = getattr(
            text_object,
            "end_time",
            None,
        )

        metadata = getattr(
            text_object,
            "metadata",
            {},
        )

        if metadata is None:
            metadata = {}

        # Calculate the authoritative spatial position.
        x, y = self.anchor_position(
            anchor=selected_anchor,
            object_width=width,
            object_height=height,
            margin=margin,
        )

        # Update the TextObject's geometry so the renderer can use the
        # same position calculated by the Layout Engine.
        text_object.rect = Rect(
            x=x,
            y=y,
            width=width,
            height=height,
        )

        layout_object = LayoutObject(
            name=object_name,
            object_type="text",
            rect=Rect(
                x=x,
                y=y,
                width=width,
                height=height,
            ),
            anchor=selected_anchor,
            z_index=selected_z_index,
            role=selected_role,
            start_time=start_time,
            end_time=end_time,
            metadata={
                **metadata,
                "source_object": text_object,
            },
        )

        return self.add_object(
            layout_object,
            check_collision=check_collision,
        )

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def check_collision(
        self,
        obj: LayoutObject,
        ignore_name: Optional[str] = None,
    ) -> List[LayoutObject]:
        """
        Return all currently registered objects that intersect obj.
        """

        collisions: List[LayoutObject] = []

        for name, existing in self.objects.items():

            if ignore_name is not None:
                if name == ignore_name:
                    continue

            if obj.rect.intersects(
                existing.rect
            ):
                collisions.append(
                    existing
                )

        return collisions

    def check_pair_collision(
        self,
        first: LayoutObject,
        second: LayoutObject,
    ) -> bool:
        """
        Explicit pairwise collision test.
        """

        return first.rect.intersects(
            second.rect
        )

    # ------------------------------------------------------------------
    # Canvas validation
    # ------------------------------------------------------------------

    def is_inside_canvas(
        self,
        obj: LayoutObject,
    ) -> bool:
        """
        Check whether an object's visible bounds are completely inside
        the canvas.
        """

        return obj.is_inside(
            self.canvas_width,
            self.canvas_height,
        )

    # ------------------------------------------------------------------
    # Object lookup
    # ------------------------------------------------------------------

    def get_object(
        self,
        name: str,
    ) -> LayoutObject:
        if name not in self.objects:
            raise KeyError(
                f"Unknown layout object: {name}"
            )

        return self.objects[name]

    def remove_object(
        self,
        name: str,
    ) -> LayoutObject:
        if name not in self.objects:
            raise KeyError(
                f"Unknown layout object: {name}"
            )

        return self.objects.pop(name)

    def clear(self):
        """
        Remove all registered layout objects.
        """

        self.objects.clear()

    # ------------------------------------------------------------------
    # Z ordering
    # ------------------------------------------------------------------

    def get_objects_by_z_index(
        self,
    ) -> List[LayoutObject]:
        """
        Return objects from back to front.
        """

        return sorted(
            self.objects.values(),
            key=lambda obj: obj.z_index,
        )

    # ------------------------------------------------------------------
    # Free regions
    # ------------------------------------------------------------------

    def get_free_regions(
        self,
        minimum_width: float = 100,
        minimum_height: float = 100,
    ) -> List[Rect]:
        """
        Estimate simple rectangular free regions.

        This is intentionally conservative.

        It is NOT a full packing algorithm.

        The purpose is to give the Director/Layout system useful information
        about available space.
        """

        occupied = [
            obj.rect
            for obj in self.objects.values()
        ]

        candidate_regions = [
            self.canvas_rect
        ]

        for occupied_rect in occupied:

            next_regions: List[Rect] = []

            for region in candidate_regions:

                if not region.intersects(
                    occupied_rect
                ):
                    next_regions.append(
                        region
                    )
                    continue

                # Above occupied region.
                if occupied_rect.y > region.y:
                    next_regions.append(
                        Rect(
                            region.x,
                            region.y,
                            region.width,
                            occupied_rect.y
                            - region.y,
                        )
                    )

                # Below occupied region.
                if occupied_rect.bottom < region.bottom:
                    next_regions.append(
                        Rect(
                            region.x,
                            occupied_rect.bottom,
                            region.width,
                            region.bottom
                            - occupied_rect.bottom,
                        )
                    )

                # Left.
                if occupied_rect.x > region.x:
                    next_regions.append(
                        Rect(
                            region.x,
                            region.y,
                            occupied_rect.x
                            - region.x,
                            region.height,
                        )
                    )

                # Right.
                if occupied_rect.right < region.right:
                    next_regions.append(
                        Rect(
                            occupied_rect.right,
                            region.y,
                            region.right
                            - occupied_rect.right,
                            region.height,
                        )
                    )

            candidate_regions = next_regions

        regions: List[Rect] = []

        for region in candidate_regions:

            if (
                region.width >= minimum_width
                and region.height >= minimum_height
            ):
                regions.append(region)

        return regions

    # ------------------------------------------------------------------
    # Layout inspection
    # ------------------------------------------------------------------

    def print_layout(self):
        """
        Print the current layout in a human-readable form.
        """

        print(
            f"Canvas "
            f"{self.canvas_width}x"
            f"{self.canvas_height}"
        )

        print(
            f"Grid "
            f"{self.grid_columns}x"
            f"{self.grid_rows}"
        )

        if not self.objects:
            print("No objects.")
            return

        print()

        for obj in self.get_objects_by_z_index():

            grid = self.pixels_to_grid(
                obj.rect
            )

            print(
                f"{obj.name}:"
            )

            print(
                f"  Type:        "
                f"{obj.object_type}"
            )

            print(
                f"  Visual Rect: "
                f"{obj.rect}"
            )

            print(
                f"  Anchor:      "
                f"{obj.anchor}"
            )

            print(
                f"  Z-index:     "
                f"{obj.z_index}"
            )

            if obj.canvas_rect is not None:
                print(
                    f"  Canvas Rect:  "
                    f"{obj.canvas_rect}"
                )

            if obj.scale is not None:
                print(
                    f"  Scale:       "
                    f"{obj.scale:.3f}"
                )

            print(
                f"  Grid: "
                f"column={grid['column']}, "
                f"row={grid['row']}, "
                f"span="
                f"{grid['column_span']}x"
                f"{grid['row_span']}"
            )

            print()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def run_test():
    """
    Test the layout engine using:

        1. Avatar geometry
        2. Generic text geometry
        3. Collision detection
        4. Canvas validation
        5. Z-order
        6. Free-region calculation
    """

    print("=" * 60)
    print("KESTREL SHORTS — GENERIC LAYOUT ENGINE TEST")
    print("=" * 60)

    engine = LayoutEngine()

    # ---------------------------------------------------------------
    # Avatar geometry
    # ---------------------------------------------------------------

    scientist_geometry = AssetGeometry(
        canvas_width=1200,
        canvas_height=1200,
        content_x=505,
        content_y=255,
        content_width=215,
        content_height=708,
    )

    print("\nAdding scientist avatar...")

    scientist = engine.add_avatar(
        name="scientist_female_1",
        geometry=scientist_geometry,
        anchor="bottom_right",
        desired_height=700,
        margin=60,
        z_index=10,
        role="avatar",
    )

    print(
        f"  Visible character: "
        f"{scientist.rect}"
    )

    print(
        f"  Full PNG canvas:   "
        f"{scientist.canvas_rect}"
    )

    print(
        f"  Scale:             "
        f"{scientist.scale:.3f}"
    )

    print(
        f"  Inside canvas:     "
        f"{engine.is_inside_canvas(scientist)}"
    )

    # ---------------------------------------------------------------
    # Generic text object
    # ---------------------------------------------------------------

    print("\nAdding attention text...")

    text = engine.add_text_object(
        name="money_keyword",
        width=672,
        height=102,
        anchor="upper_left",
        margin=80,
        role="attention",
        z_index=20,
    )

    print(
        f"  Text bounds:       "
        f"{text.rect}"
    )

    print(
        f"  Inside canvas:     "
        f"{engine.is_inside_canvas(text)}"
    )

    # ---------------------------------------------------------------
    # Collision
    # ---------------------------------------------------------------

    collisions = engine.check_collision(
        text,
        ignore_name=text.name,
    )

    print("\nText collision check:")

    if collisions:
        for collision in collisions:
            print(
                f"  COLLISION: "
                f"{collision.name}"
            )
    else:
        print(
            "  No collision detected."
        )

    # ---------------------------------------------------------------
    # Intentional test object
    # ---------------------------------------------------------------

    print(
        "\nAdding test image near avatar..."
    )

    test_image = engine.add_at_anchor(
        name="test_image",
        object_type="image",
        width=300,
        height=350,
        anchor="lower_right",
        margin=80,
        z_index=5,
        role="visual",
    )

    image_collisions = engine.check_collision(
        test_image,
        ignore_name="test_image",
    )

    if image_collisions:
        print(
            "  Image collides with:"
        )

        for collision in image_collisions:
            print(
                f"    - {collision.name}"
            )
    else:
        print(
            "  Image has no collision."
        )

    # ---------------------------------------------------------------
    # Layout printout
    # ---------------------------------------------------------------

    print("\nCurrent layout:")
    engine.print_layout()

    # ---------------------------------------------------------------
    # Free regions
    # ---------------------------------------------------------------

    print(
        "Estimated free regions:"
    )

    free_regions = engine.get_free_regions(
        minimum_width=200,
        minimum_height=200,
    )

    for index, region in enumerate(
        free_regions[:10],
        start=1,
    ):
        print(
            f"  {index}: {region}"
        )

    print()
    print(
        "Layout engine test complete."
    )

    print("=" * 60)


if __name__ == "__main__":
    run_test()