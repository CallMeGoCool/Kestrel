"""
Kestrel Shorts - Geometry
=========================

The one Rect class, shared by asset measuring, the layout spec and (later) the
Frame Supervisor, so no file defines its own. Pixel space: x to the right,
y downwards, origin at the top-left of the 1080x1920 canvas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Rect:
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
        return (other.x >= self.x and other.y >= self.y
                and other.right <= self.right and other.bottom <= self.bottom)

    def intersects(self, other: "Rect") -> bool:
        return not (other.right <= self.x or other.x >= self.right
                    or other.bottom <= self.y or other.y >= self.bottom)

    def intersection(self, other: "Rect") -> Optional["Rect"]:
        if not self.intersects(other):
            return None
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        return Rect(x1, y1, min(self.right, other.right) - x1, min(self.bottom, other.bottom) - y1)

    def inset(self, amount: float) -> "Rect":
        return Rect(self.x + amount, self.y + amount,
                    max(0.0, self.width - 2 * amount), max(0.0, self.height - 2 * amount))

    def expand(self, amount: float) -> "Rect":
        return Rect(self.x - amount, self.y - amount, self.width + 2 * amount, self.height + 2 * amount)

    def translated(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def clamp_inside(self, outer: "Rect") -> "Rect":
        """Move (never resize) this rectangle so it lies inside `outer`.
        If it is larger than `outer`, it is aligned to outer's top-left."""
        x = outer.x if self.width >= outer.width else min(max(self.x, outer.x), outer.right - self.width)
        y = outer.y if self.height >= outer.height else min(max(self.y, outer.y), outer.bottom - self.height)
        return Rect(x, y, self.width, self.height)

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 1), "y": round(self.y, 1),
                "width": round(self.width, 1), "height": round(self.height, 1)}

    def __str__(self) -> str:
        return f"Rect(x={self.x:.1f}, y={self.y:.1f}, w={self.width:.1f}, h={self.height:.1f})"