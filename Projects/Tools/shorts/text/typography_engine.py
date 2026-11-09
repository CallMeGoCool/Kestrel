"""
Kestrel Shorts — Typography Engine
===================================

Purpose
-------
The Typography Engine converts structured text intent into measurable,
renderable text objects.

Important architectural rule:
    The Typography Engine does NOT decide what should be emphasized.

Upstream systems (Architect / Visual Planner / Director) provide intent such as:

    MONEY -> hero_keyword
    advertising -> keyword
    supporting sentence -> supporting

This engine is responsible for:
    - font selection
    - text measurement
    - line wrapping
    - span styling
    - visible text geometry
    - semantic anchoring
    - rendering
    - basic timing / animation metadata

It deliberately does NOT:
    - generate scripts
    - detect important keywords
    - decide what information matters
    - choose the content of captions
    - perform AI inference

The eventual flow is:

    Script / Visual Intent
            |
            v
    Typography Engine
            |
            +--> TextObject
                    |
                    +--> visible bounds
                    +--> style
                    +--> anchor
                    +--> timing
                    +--> rendering data
                    |
                    v
               Layout Engine
                    |
                    v
               Composition

Canvas:
    1080 x 1920 (9:16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920


# ---------------------------------------------------------------------------
# Font candidates
# ---------------------------------------------------------------------------
#
# Windows-first because Kestrel is currently being developed on Windows.
# The engine tries these in order and falls back to PIL's default font.
#

FONT_CANDIDATES = {
    "bold": [
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ],
    "regular": [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ],
}


# ---------------------------------------------------------------------------
# Basic geometry
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    """
    A simple rectangle representing visible geometry.

    Coordinates are in pixels relative to the 1080x1920 canvas.
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

    def contains(self, other: "Rect") -> bool:
        return (
            other.x >= self.x
            and other.y >= self.y
            and other.right <= self.right
            and other.bottom <= self.bottom
        )

    def intersects(self, other: "Rect") -> bool:
        return not (
            other.right <= self.x
            or other.x >= self.right
            or other.bottom <= self.y
            or other.y >= self.bottom
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
# Text styles
# ---------------------------------------------------------------------------

@dataclass
class TextStyle:
    """
    Defines how a span of text should visually appear.

    The style is deliberately declarative. It can later be loaded from a
    channel profile rather than being hardcoded here.
    """

    name: str

    font_size: int
    font_path: Optional[str] = None

    fill: Tuple[int, int, int, int] = (255, 255, 255, 255)

    stroke_fill: Tuple[int, int, int, int] = (0, 0, 0, 255)
    stroke_width: int = 0

    line_spacing: int = 10
    align: str = "left"

    # Future animation/rendering hints.
    scale: float = 1.0
    opacity: int = 255


@dataclass
class TextSpan:
    """
    A piece of text with a specific visual style.

    Example:
        TextSpan("MONEY", "hero_keyword")
        TextSpan(" is life", "supporting")
    """

    text: str
    style: str


# ---------------------------------------------------------------------------
# Measured span / line
# ---------------------------------------------------------------------------

@dataclass
class MeasuredSpan:
    """
    A TextSpan after its actual font geometry has been measured.
    """

    text: str
    style: TextStyle

    width: float
    height: float

    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float

    @property
    def visual_width(self) -> float:
        return self.bbox_right - self.bbox_left

    @property
    def visual_height(self) -> float:
        return self.bbox_bottom - self.bbox_top


@dataclass
class TextLine:
    """
    A single rendered line containing one or more differently styled spans.
    """

    spans: List[MeasuredSpan] = field(default_factory=list)

    @property
    def width(self) -> float:
        return sum(span.width for span in self.spans)

    @property
    def height(self) -> float:
        if not self.spans:
            return 0

        return max(span.height for span in self.spans)


# ---------------------------------------------------------------------------
# Text block
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """
    A measurable group of text lines.

    The block's x/y position describes its final placement on the canvas.
    """

    lines: List[TextLine]

    width: float
    height: float

    x: float = 0
    y: float = 0

    align: str = "left"

    @property
    def rect(self) -> Rect:
        return Rect(
            self.x,
            self.y,
            self.width,
            self.height,
        )


# ---------------------------------------------------------------------------
# Text Object
# ---------------------------------------------------------------------------

@dataclass
class TextObject:
    """
    The main object passed between Typography, Layout, and Composition.

    This is intentionally analogous to the avatar geometry model.

    A TextObject represents a piece of visual information, not merely a
    string of characters.

    Example:

        TextObject(
            name="money_keyword",
            content="MONEY",
            block=...,
            rect=...,
            role="attention",
            anchor="upper_left",
            z_index=20,
            start_time=2.4,
            end_time=5.1,
        )
    """

    name: str
    content: str
    block: TextBlock
    rect: Rect

    role: str = "attention"

    anchor: Optional[str] = None

    z_index: int = 20

    start_time: Optional[float] = None
    end_time: Optional[float] = None

    animation: Optional[str] = None

    visible: bool = True

    # Optional metadata for future composition logic.
    metadata: Dict[str, object] = field(default_factory=dict)

    def intersects(self, other: "TextObject") -> bool:
        return self.rect.intersects(other.rect)

    def is_inside_canvas(
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

        return canvas.contains(self.rect)


# ---------------------------------------------------------------------------
# Typography Engine
# ---------------------------------------------------------------------------

class TypographyEngine:
    """
    Kestrel Shorts typography system.

    Responsibilities:
        1. Load fonts.
        2. Measure text.
        3. Wrap text.
        4. Create TextBlocks.
        5. Create TextObjects.
        6. Render TextObjects.

    It does NOT decide what should be emphasized.
    """

    def __init__(
        self,
        canvas_width: int = CANVAS_WIDTH,
        canvas_height: int = CANVAS_HEIGHT,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

        self.styles = self._build_default_styles()

    # ------------------------------------------------------------------
    # Font handling
    # ------------------------------------------------------------------

    def _find_font(
        self,
        weight: str,
    ) -> Optional[str]:
        """
        Find the first available font for the requested weight.
        """

        candidates = FONT_CANDIDATES.get(
            weight,
            FONT_CANDIDATES["regular"],
        )

        for candidate in candidates:
            if Path(candidate).exists():
                return candidate

        return None

    def _load_font(
        self,
        font_path: Optional[str],
        font_size: int,
    ):
        """
        Load and cache a PIL font.
        """

        cache_key = (
            font_path or "default",
            font_size,
        )

        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(
                font_path,
                font_size,
            )
        else:
            try:
                font = ImageFont.truetype(
                    "arial.ttf",
                    font_size,
                )
            except OSError:
                font = ImageFont.load_default()

        self.font_cache[cache_key] = font

        return font

    # ------------------------------------------------------------------
    # Default styles
    # ------------------------------------------------------------------

    def _build_default_styles(self) -> Dict[str, TextStyle]:
        """
        Default visual vocabulary.

        These are starting points, not permanent channel rules.
        """

        bold_font = self._find_font("bold")
        regular_font = self._find_font("regular")

        return {
            "hero": TextStyle(
                name="hero",
                font_size=132,
                font_path=bold_font,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=2,
                line_spacing=12,
                align="left",
            ),

            "hero_keyword": TextStyle(
                name="hero_keyword",
                font_size=132,
                font_path=bold_font,
                fill=(255, 220, 80, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=3,
                line_spacing=12,
                align="left",
            ),

            "keyword": TextStyle(
                name="keyword",
                font_size=118,
                font_path=bold_font,
                fill=(255, 220, 80, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=2,
                line_spacing=10,
                align="left",
            ),

            "supporting": TextStyle(
                name="supporting",
                font_size=72,
                font_path=regular_font,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=1,
                line_spacing=8,
                align="left",
            ),

            "caption": TextStyle(
                name="caption",
                font_size=58,
                font_path=bold_font,
                fill=(255, 255, 255, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=2,
                line_spacing=6,
                align="center",
            ),

            "caption_keyword": TextStyle(
                name="caption_keyword",
                font_size=62,
                font_path=bold_font,
                fill=(255, 220, 80, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=2,
                line_spacing=6,
                align="center",
            ),

            "small": TextStyle(
                name="small",
                font_size=42,
                font_path=regular_font,
                fill=(230, 230, 230, 255),
                stroke_fill=(0, 0, 0, 255),
                stroke_width=1,
                line_spacing=5,
                align="left",
            ),
        }

    # ------------------------------------------------------------------
    # Style access
    # ------------------------------------------------------------------

    def get_style(
        self,
        style_name: str,
    ) -> TextStyle:
        """
        Return a named text style.
        """

        if style_name not in self.styles:
            raise ValueError(
                f"Unknown text style: {style_name}"
            )

        return self.styles[style_name]

    # ------------------------------------------------------------------
    # Text measurement
    # ------------------------------------------------------------------

    def measure_text(
        self,
        text: str,
        style_name: str,
    ) -> MeasuredSpan:
        """
        Measure a piece of text using the actual PIL font metrics.

        Important:
            PIL's textbbox coordinates do not necessarily start at (0, 0).

        We preserve the real bbox offsets so the renderer can compensate
        correctly and avoid clipping the top/bottom of characters.
        """

        style = self.get_style(style_name)

        font_size = max(
            1,
            int(style.font_size * style.scale),
        )

        font = self._load_font(
            style.font_path,
            font_size,
        )

        dummy = Image.new(
            "RGBA",
            (10, 10),
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(dummy)

        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font,
            stroke_width=style.stroke_width,
        )

        left, top, right, bottom = bbox

        width = right - left
        height = bottom - top

        return MeasuredSpan(
            text=text,
            style=style,
            width=width,
            height=height,
            bbox_left=left,
            bbox_top=top,
            bbox_right=right,
            bbox_bottom=bottom,
        )

    # ------------------------------------------------------------------
    # Wrapping
    # ------------------------------------------------------------------

    def wrap_spans(
        self,
        spans: List[TextSpan],
        max_width: float,
    ) -> List[TextLine]:
        """
        Wrap spans into lines while preserving individual styles.

        This is intentionally conservative.

        We don't attempt sophisticated linguistic line-breaking yet.
        That can be improved later without changing the TextObject model.
        """

        if max_width <= 0:
            raise ValueError("max_width must be greater than zero")

        lines: List[TextLine] = []

        current_spans: List[MeasuredSpan] = []
        current_width = 0.0

        for span in spans:
            if not span.text:
                continue

            measured = self.measure_text(
                span.text,
                span.style,
            )

            # If the complete span fits, append it.
            if (
                current_width + measured.width <= max_width
                or not current_spans
            ):
                current_spans.append(measured)
                current_width += measured.width
                continue

            # If the span doesn't fit, attempt word-level wrapping.
            words = span.text.split(" ")

            for index, word in enumerate(words):
                if not word:
                    continue

                token = word

                if index < len(words) - 1:
                    token += " "

                measured_token = self.measure_text(
                    token,
                    span.style,
                )

                if (
                    current_width + measured_token.width <= max_width
                    or not current_spans
                ):
                    current_spans.append(
                        measured_token
                    )
                    current_width += measured_token.width

                else:
                    lines.append(
                        TextLine(
                            spans=current_spans
                        )
                    )

                    current_spans = [
                        measured_token
                    ]

                    current_width = (
                        measured_token.width
                    )

        if current_spans:
            lines.append(
                TextLine(
                    spans=current_spans
                )
            )

        return lines

    # ------------------------------------------------------------------
    # Text block creation
    # ------------------------------------------------------------------

    def create_block(
        self,
        spans: List[TextSpan],
        max_width: float,
        x: float = 0,
        y: float = 0,
        align: str = "left",
    ) -> TextBlock:
        """
        Create a measurable TextBlock.
        """

        lines = self.wrap_spans(
            spans,
            max_width,
        )

        if not lines:
            return TextBlock(
                lines=[],
                width=0,
                height=0,
                x=x,
                y=y,
                align=align,
            )

        width = max(
            line.width
            for line in lines
        )

        height = 0

        for index, line in enumerate(lines):
            height += line.height

            if index < len(lines) - 1:
                spacing = max(
                    span.style.line_spacing
                    for span in line.spans
                )

                height += spacing

        return TextBlock(
            lines=lines,
            width=width,
            height=height,
            x=x,
            y=y,
            align=align,
        )

    # ------------------------------------------------------------------
    # Text Object creation
    # ------------------------------------------------------------------

    def create_text_object(
        self,
        name: str,
        spans: List[TextSpan],
        max_width: float,
        role: str = "attention",
        anchor: Optional[str] = None,
        x: float = 0,
        y: float = 0,
        align: str = "left",
        z_index: int = 20,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        animation: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> TextObject:
        """
        Build the main TextObject.

        The caller supplies the content and visual intent.
        """

        block = self.create_block(
            spans=spans,
            max_width=max_width,
            x=x,
            y=y,
            align=align,
        )

        rect = block.rect

        content = "".join(
            span.text
            for span in spans
        )

        return TextObject(
            name=name,
            content=content,
            block=block,
            rect=rect,
            role=role,
            anchor=anchor,
            z_index=z_index,
            start_time=start_time,
            end_time=end_time,
            animation=animation,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Anchor placement
    # ------------------------------------------------------------------

    def place_at_anchor(
        self,
        text_object: TextObject,
        anchor: str,
        margin: float = 60,
    ) -> TextObject:
        """
        Place a TextObject using a semantic anchor.

        The actual pixel position is calculated here.

        LayoutEngine will eventually become the central authority for this.
        This method exists now so Typography can be tested independently.
        """

        rect = text_object.rect

        positions = {
            "top_left": (
                margin,
                margin,
            ),

            "top_center": (
                (self.canvas_width - rect.width) / 2,
                margin,
            ),

            "top_right": (
                self.canvas_width
                - rect.width
                - margin,
                margin,
            ),

            "center_left": (
                margin,
                (self.canvas_height - rect.height) / 2,
            ),

            "center": (
                (self.canvas_width - rect.width) / 2,
                (self.canvas_height - rect.height) / 2,
            ),

            "center_right": (
                self.canvas_width
                - rect.width
                - margin,
                (self.canvas_height - rect.height) / 2,
            ),

            "bottom_left": (
                margin,
                self.canvas_height
                - rect.height
                - margin,
            ),

            "bottom_center": (
                (self.canvas_width - rect.width) / 2,
                self.canvas_height
                - rect.height
                - margin,
            ),

            "bottom_right": (
                self.canvas_width
                - rect.width
                - margin,
                self.canvas_height
                - rect.height
                - margin,
            ),

		"upper_left": (
    margin,
    self.canvas_height * 0.25 - rect.height / 2,
),

"upper_center": (
    (self.canvas_width - rect.width) / 2,
    self.canvas_height * 0.25 - rect.height / 2,
),

"upper_right": (
    self.canvas_width - rect.width - margin,
    self.canvas_height * 0.25 - rect.height / 2,
),

"lower_left": (
    margin,
    self.canvas_height * 0.75 - rect.height / 2,
),

"lower_center": (
    (self.canvas_width - rect.width) / 2,
    self.canvas_height * 0.75 - rect.height / 2,
),

"lower_right": (
    self.canvas_width - rect.width - margin,
    self.canvas_height * 0.75 - rect.height / 2,
),
        }

        if anchor not in positions:
            raise ValueError(
                f"Unknown text anchor: {anchor}"
            )

        x, y = positions[anchor]

        text_object.block.x = x
        text_object.block.y = y

        text_object.rect = Rect(
            x,
            y,
            rect.width,
            rect.height,
        )

        text_object.anchor = anchor

        return text_object

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_block(
        self,
        image: Image.Image,
        block: TextBlock,
    ) -> Image.Image:
        """
        Render a TextBlock onto an RGBA image.

        The renderer compensates for PIL's font bbox offsets.

        This fixes the earlier issue where the lower part of MONEY could
        become clipped because the drawing origin did not account for the
        font's negative/positive bbox offsets.
        """

        draw = ImageDraw.Draw(
            image,
            "RGBA",
        )

        current_y = block.y

        for line in block.lines:
            current_x = block.x

            if block.align == "center":
                current_x = (
                    block.x
                    + (block.width - line.width) / 2
                )

            elif block.align == "right":
                current_x = (
                    block.x
                    + block.width
                    - line.width
                )

            line_height = line.height

            for measured in line.spans:
                style = measured.style

                font_size = max(
                    1,
                    int(
                        style.font_size
                        * style.scale
                    ),
                )

                font = self._load_font(
                    style.font_path,
                    font_size,
                )

                # PIL's bbox may have a non-zero top/left offset.
                #
                # We compensate so the actual glyphs occupy the measured
                # visual area instead of being shifted/clipped.
                draw_x = (
                    current_x
                    - measured.bbox_left
                )

                draw_y = (
                    current_y
                    - measured.bbox_top
                )

                draw.text(
                    (draw_x, draw_y),
                    measured.text,
                    font=font,
                    fill=style.fill,
                    stroke_width=style.stroke_width,
                    stroke_fill=style.stroke_fill,
                )

                current_x += measured.width

            current_y += line_height

            if line.spans:
                current_y += max(
                    span.style.line_spacing
                    for span in line.spans
                )

        return image

    def render_text_object(
        self,
        image: Image.Image,
        text_object: TextObject,
    ) -> Image.Image:
        """
        Render a TextObject.

        Timing/animation is not evaluated yet. That belongs to the future
        composition/timeline system.
        """

        if not text_object.visible:
            return image

        return self.render_block(
            image=image,
            block=text_object.block,
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def run_test():
    """
    Basic typography test.

    Demonstrates:
        - hero keyword
        - supporting text
        - real TextObject geometry
        - semantic anchoring
        - different styles inside one text object
        - rendering
    """

    print("=" * 60)
    print("KESTREL SHORTS — TYPOGRAPHY ENGINE TEST")
    print("=" * 60)

    engine = TypographyEngine()

    spans = [
        TextSpan(
            text="MONEY",
            style="hero_keyword",
        ),
        TextSpan(
            text=" is life",
            style="supporting",
        ),
    ]

    text_object = engine.create_text_object(
        name="money_statement",
        spans=spans,
        max_width=850,
        role="attention",
        anchor="upper_left",
        z_index=20,
        start_time=0.0,
        end_time=4.0,
        animation="scale_in",
    )

    engine.place_at_anchor(
        text_object,
        anchor="upper_left",
        margin=80,
    )

    print("\nText Object:")
    print(f"  Name:       {text_object.name}")
    print(f"  Content:    {text_object.content}")
    print(f"  Role:       {text_object.role}")
    print(f"  Anchor:     {text_object.anchor}")
    print(f"  Z-index:    {text_object.z_index}")
    print(f"  Start:      {text_object.start_time}s")
    print(f"  End:        {text_object.end_time}s")
    print(f"  Animation:  {text_object.animation}")
    print(f"  Visual Rect:{text_object.rect}")
    print(
        f"  Inside:     "
        f"{text_object.is_inside_canvas()}"
    )

    print("\nLines:")

    for index, line in enumerate(
        text_object.block.lines,
        start=1,
    ):
        print(
            f"  Line {index}: "
            f"{line.width:.1f} x "
            f"{line.height:.1f}"
        )

        for span in line.spans:
            print(
                f"    '{span.text}' "
                f"-> {span.style.name} "
                f"({span.width:.1f} x "
                f"{span.height:.1f})"
            )

    image = Image.new(
        "RGBA",
        (
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
        ),
        (20, 20, 20, 255),
    )

    engine.render_text_object(
        image,
        text_object,
    )

    output_path = (
        Path(
            r"C:\AI_Studio\Outputs\Images"
        )
        / "shorts_typography_test.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)

    print("\nSaved:")
    print(f"  {output_path}")

    print("\nTypography test complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
