"""
Kestrel Shorts — Composition Rules
===================================

Purpose
-------
Define high-level composition constraints that operate ABOVE the
low-level LayoutEngine.

The LayoutEngine answers:

    "Do these objects geometrically overlap?"

CompositionRules answers:

    "Are these objects ALLOWED to overlap?"

This module does NOT move objects.

It only describes and evaluates composition constraints.

Current responsibilities
------------------------
- Minimum spacing between objects
- Collision avoidance
- Intentional overlap
- Safe-area constraints
- Constraint evaluation

Future responsibilities may include:
- Preferred regions
- Priority-based placement
- Automatic repositioning
- Object avoidance
- Layout optimization

Important architectural boundary
--------------------------------
LayoutEngine:
    Geometry + placement + raw collision detection

CompositionRules:
    Higher-level composition intent and constraints

Renderer:
    Actual visual rendering
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constraint Types
# ---------------------------------------------------------------------------

CONSTRAINT_AVOID = "avoid"
CONSTRAINT_ALLOW_OVERLAP = "allow_overlap"
CONSTRAINT_MIN_SPACING = "min_spacing"
CONSTRAINT_SAFE_AREA = "safe_area"


# ---------------------------------------------------------------------------
# Composition Constraint
# ---------------------------------------------------------------------------

@dataclass
class CompositionConstraint:
    """
    Describes a relationship or requirement between layout objects.
    """

    constraint_type: str

    object_a: str

    object_b: Optional[str] = None

    minimum_spacing: float = 0.0

    name: Optional[str] = None

    priority: int = 100

    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

        if self.name is None:
            if self.object_b is not None:
                self.name = (
                    f"{self.constraint_type}:"
                    f"{self.object_a}:"
                    f"{self.object_b}"
                )
            else:
                self.name = (
                    f"{self.constraint_type}:"
                    f"{self.object_a}"
                )


# ---------------------------------------------------------------------------
# Constraint Result
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    """
    Result of evaluating one composition constraint.
    """

    constraint: CompositionConstraint

    passed: bool

    message: str

    actual_spacing: Optional[float] = None

    overlap_detected: Optional[bool] = None


# ---------------------------------------------------------------------------
# Composition Rules
# ---------------------------------------------------------------------------

class CompositionRules:
    """
    High-level composition constraint manager.

    This class does not modify LayoutEngine objects.

    It evaluates whether the current layout satisfies the requested
    composition rules.
    """

    def __init__(self):
        self.constraints: List[CompositionConstraint] = []

    # ------------------------------------------------------------------
    # Add constraints
    # ------------------------------------------------------------------

    def add_avoid(
        self,
        object_a: str,
        object_b: str,
        minimum_spacing: float = 0.0,
        priority: int = 100,
        name: Optional[str] = None,
    ) -> CompositionConstraint:
        """
        Require two objects not to overlap and optionally maintain
        a minimum spacing between them.
        """

        constraint = CompositionConstraint(
            constraint_type=CONSTRAINT_AVOID,
            object_a=object_a,
            object_b=object_b,
            minimum_spacing=minimum_spacing,
            priority=priority,
            name=name,
        )

        self.constraints.append(constraint)

        return constraint

    # ------------------------------------------------------------------

    def add_min_spacing(
        self,
        object_a: str,
        object_b: str,
        minimum_spacing: float,
        priority: int = 100,
        name: Optional[str] = None,
    ) -> CompositionConstraint:
        """
        Require two objects to maintain at least the specified
        spacing between their visible rectangles.
        """

        constraint = CompositionConstraint(
            constraint_type=CONSTRAINT_MIN_SPACING,
            object_a=object_a,
            object_b=object_b,
            minimum_spacing=minimum_spacing,
            priority=priority,
            name=name,
        )

        self.constraints.append(constraint)

        return constraint

    # ------------------------------------------------------------------

    def add_allow_overlap(
        self,
        object_a: str,
        object_b: str,
        priority: int = 100,
        name: Optional[str] = None,
    ) -> CompositionConstraint:
        """
        Explicitly declare that two objects are allowed to overlap.

        This does not force overlap.

        It simply tells the composition system that overlap between
        these two objects is intentional/acceptable.
        """

        constraint = CompositionConstraint(
            constraint_type=CONSTRAINT_ALLOW_OVERLAP,
            object_a=object_a,
            object_b=object_b,
            priority=priority,
            name=name,
        )

        self.constraints.append(constraint)

        return constraint

    # ------------------------------------------------------------------

    def add_safe_area(
        self,
        object_a: str,
        left: float = 0.0,
        top: float = 0.0,
        right: float = 0.0,
        bottom: float = 0.0,
        priority: int = 100,
        name: Optional[str] = None,
    ) -> CompositionConstraint:
        """
        Require an object to remain inside a safe area defined by
        edge margins.
        """

        constraint = CompositionConstraint(
            constraint_type=CONSTRAINT_SAFE_AREA,
            object_a=object_a,
            priority=priority,
            name=name,
            metadata={
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
        )

        self.constraints.append(constraint)

        return constraint

    # ------------------------------------------------------------------
    # Object lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _get_object(
        layout,
        name: str,
    ):
        """
        Retrieve a LayoutObject by name.

        Returns None if the object does not exist.
        """

        return layout.get_object(name)

    # ------------------------------------------------------------------
    # Rectangle spacing
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_spacing(rect_a, rect_b) -> float:
        """
        Calculate the minimum horizontal/vertical gap between two
        rectangles.

        If rectangles overlap on one axis, that axis contributes zero.

        For example:

            A       B

        returns the gap between them.

        If they overlap:

            AAAAA
              BBBBB

        the result is zero.
        """

        horizontal_gap = max(
            0.0,
            max(
                rect_a.x - rect_b.right,
                rect_b.x - rect_a.right,
            ),
        )

        vertical_gap = max(
            0.0,
            max(
                rect_a.y - rect_b.bottom,
                rect_b.y - rect_a.bottom,
            ),
        )

        # If objects are separated diagonally, return the smaller
        # axis gap. If they overlap, this naturally becomes zero.
        if horizontal_gap > 0.0 and vertical_gap > 0.0:
            return min(
                horizontal_gap,
                vertical_gap,
            )

        if horizontal_gap > 0.0:
            return horizontal_gap

        if vertical_gap > 0.0:
            return vertical_gap

        return 0.0

    # ------------------------------------------------------------------
    # Evaluate one constraint
    # ------------------------------------------------------------------

    def evaluate_constraint(
        self,
        layout,
        constraint: CompositionConstraint,
    ) -> ConstraintResult:
        """
        Evaluate a single composition constraint against the current
        LayoutEngine state.
        """

        object_a = self._get_object(
            layout,
            constraint.object_a,
        )

        if object_a is None:
            return ConstraintResult(
                constraint=constraint,
                passed=False,
                message=(
                    f"Object '{constraint.object_a}' "
                    f"does not exist in layout."
                ),
            )

        # --------------------------------------------------------------
        # SAFE AREA
        # --------------------------------------------------------------

        if constraint.constraint_type == CONSTRAINT_SAFE_AREA:

            metadata = constraint.metadata or {}

            left = float(
                metadata.get("left", 0.0)
            )

            top = float(
                metadata.get("top", 0.0)
            )

            right = float(
                metadata.get("right", 0.0)
            )

            bottom = float(
                metadata.get("bottom", 0.0)
            )

            safe_left = left
            safe_top = top

            safe_right = (
                layout.canvas_width
                - right
            )

            safe_bottom = (
                layout.canvas_height
                - bottom
            )

            rect = object_a.rect

            passed = (
                rect.x >= safe_left
                and rect.y >= safe_top
                and rect.right <= safe_right
                and rect.bottom <= safe_bottom
            )

            if passed:
                message = (
                    f"{object_a.name} is inside "
                    f"the safe area."
                )
            else:
                message = (
                    f"{object_a.name} violates "
                    f"the safe area."
                )

            return ConstraintResult(
                constraint=constraint,
                passed=passed,
                message=message,
            )

        # --------------------------------------------------------------
        # Object-to-object constraints
        # --------------------------------------------------------------

        object_b = self._get_object(
            layout,
            constraint.object_b,
        )

        if object_b is None:
            return ConstraintResult(
                constraint=constraint,
                passed=False,
                message=(
                    f"Object '{constraint.object_b}' "
                    f"does not exist in layout."
                ),
            )

        rect_a = object_a.rect
        rect_b = object_b.rect

        overlap = rect_a.intersects(rect_b)

        spacing = self._calculate_spacing(
            rect_a,
            rect_b,
        )

        # --------------------------------------------------------------
        # ALLOW OVERLAP
        # --------------------------------------------------------------

        if (
            constraint.constraint_type
            == CONSTRAINT_ALLOW_OVERLAP
        ):

            return ConstraintResult(
                constraint=constraint,
                passed=True,
                message=(
                    f"Overlap between "
                    f"{object_a.name} and "
                    f"{object_b.name} is allowed."
                ),
                actual_spacing=spacing,
                overlap_detected=overlap,
            )

        # --------------------------------------------------------------
        # AVOID
        # --------------------------------------------------------------

        if constraint.constraint_type == CONSTRAINT_AVOID:

            required_spacing = (
                constraint.minimum_spacing
            )

            passed = (
                not overlap
                and spacing >= required_spacing
            )

            if passed:
                message = (
                    f"{object_a.name} and "
                    f"{object_b.name} satisfy "
                    f"avoidance rule "
                    f"(spacing={spacing:.1f}px)."
                )
            else:
                message = (
                    f"{object_a.name} and "
                    f"{object_b.name} violate "
                    f"avoidance rule "
                    f"(spacing={spacing:.1f}px, "
                    f"required={required_spacing:.1f}px)."
                )

            return ConstraintResult(
                constraint=constraint,
                passed=passed,
                message=message,
                actual_spacing=spacing,
                overlap_detected=overlap,
            )

        # --------------------------------------------------------------
        # MINIMUM SPACING
        # --------------------------------------------------------------

        if (
            constraint.constraint_type
            == CONSTRAINT_MIN_SPACING
        ):

            required_spacing = (
                constraint.minimum_spacing
            )

            passed = (
                not overlap
                and spacing >= required_spacing
            )

            if passed:
                message = (
                    f"{object_a.name} and "
                    f"{object_b.name} maintain "
                    f"{spacing:.1f}px spacing "
                    f"(required={required_spacing:.1f}px)."
                )
            else:
                message = (
                    f"{object_a.name} and "
                    f"{object_b.name} have insufficient "
                    f"spacing "
                    f"(actual={spacing:.1f}px, "
                    f"required={required_spacing:.1f}px)."
                )

            return ConstraintResult(
                constraint=constraint,
                passed=passed,
                message=message,
                actual_spacing=spacing,
                overlap_detected=overlap,
            )

        # --------------------------------------------------------------
        # UNKNOWN
        # --------------------------------------------------------------

        return ConstraintResult(
            constraint=constraint,
            passed=False,
            message=(
                f"Unknown constraint type: "
                f"{constraint.constraint_type}"
            ),
        )

    # ------------------------------------------------------------------
    # Evaluate all constraints
    # ------------------------------------------------------------------

    def evaluate(
        self,
        layout,
    ) -> List[ConstraintResult]:
        """
        Evaluate all registered constraints.

        Results are returned in priority order.
        """

        sorted_constraints = sorted(
            self.constraints,
            key=lambda constraint: constraint.priority,
            reverse=True,
        )

        return [
            self.evaluate_constraint(
                layout,
                constraint,
            )
            for constraint in sorted_constraints
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_results(
        self,
        results: List[ConstraintResult],
    ):
        """
        Print a readable constraint evaluation summary.
        """

        print(
            "\nComposition Constraint Results:"
        )

        if not results:
            print(
                "  No constraints registered."
            )
            return

        passed_count = 0

        for result in results:

            status = (
                "PASS"
                if result.passed
                else "FAIL"
            )

            if result.passed:
                passed_count += 1

            print(
                f"  [{status}] "
                f"{result.constraint.name}"
            )

            print(
                f"         {result.message}"
            )

        print(
            f"\n  Passed: "
            f"{passed_count}/{len(results)}"
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self):
        """
        Remove all composition constraints.
        """

        self.constraints.clear()

    def __len__(self):
        return len(self.constraints)


# ---------------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "CompositionRules module loaded successfully."
    )

    rules = CompositionRules()

    rules.add_avoid(
        "avatar",
        "image",
        minimum_spacing=40,
    )

    rules.add_allow_overlap(
        "avatar",
        "background",
    )

    rules.add_safe_area(
        "text",
        left=60,
        top=60,
        right=60,
        bottom=60,
    )

    print(
        f"Registered constraints: {len(rules)}"
    )

    for constraint in rules.constraints:
        print(
            f"  - {constraint.name}"
        )