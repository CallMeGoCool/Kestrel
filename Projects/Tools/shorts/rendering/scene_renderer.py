"""
Kestrel Shorts — Scene Renderer
================================

Purpose
-------
Render a CompositionEngine scene into a 1080x1920 image.

Architecture
------------

    TypographyEngine
    Asset Geometry
          ↓
    LayoutEngine
          ↓
    CompositionRules
          ↓
    CompositionEngine
          ↓
    SceneRenderer
          ↓
    PNG Frame

Responsibilities
----------------
Current version:
- Create the scene canvas
- Render objects in z-index order
- Render typography using TypographyEngine
- Render real transparent avatar assets
- Render placeholder geometry for visuals
- Save a PNG frame

This renderer intentionally does NOT:
- Use an LLM
- Decide scene composition
- Move objects
- Solve layout constraints
- Generate images
- Call ComfyUI
- Generate avatars
- Generate video
- Handle animation

Important architectural boundary
---------------------------------
CompositionEngine:
    Decides what objects belong to the scene and their geometry.

SceneRenderer:
    Draws those objects.

Future versions will replace placeholder visuals
with real generated media and add animation/video support.
"""

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

from typography_engine import TypographyEngine


DEFAULT_BACKGROUND = (30, 30, 30, 255)

AVATAR_OUTLINE = (100, 220, 255, 255)
VISUAL_OUTLINE = (255, 200, 100, 255)
LABEL_COLOR = (255, 255, 255, 255)


class SceneRenderer:
    def __init__(
        self,
        composition_engine,
        typography_engine: Optional[TypographyEngine] = None,
        background=DEFAULT_BACKGROUND,
        avatar_root=None,
    ):
        self.composition = composition_engine

        if typography_engine is None:
            typography_engine = TypographyEngine(
                canvas_width=composition_engine.scene.width,
                canvas_height=composition_engine.scene.height,
            )

        self.typography = typography_engine
        self.background = background

        # Default avatar root:
        #
        # C:\AI_Studio\Projects\Tools\shorts\
        #     avatars\
        #
        # AvatarRenderer searches recursively inside this folder
        # for the canonical avatar PNG.
        if avatar_root is None:
            avatar_root = Path(__file__).resolve().parent / "avatars"

        self.avatar_root = Path(avatar_root)

        self.image = None
        self.draw = None

    # ------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------

    def create_canvas(self):
        width = self.composition.scene.width
        height = self.composition.scene.height

        self.image = Image.new(
            "RGBA",
            (width, height),
            self.background,
        )

        self.draw = ImageDraw.Draw(self.image)

        return self.image

    # ------------------------------------------------------------------
    # Object dispatch
    # ------------------------------------------------------------------

    def render_object(self, layout_object):
        object_type = layout_object.object_type

        if object_type == "text":
            return self.render_text(layout_object)

        if object_type == "avatar":
            return self.render_avatar(layout_object)

        if object_type == "image":
            return self.render_visual(layout_object)

        return self.render_generic(layout_object)

    # ------------------------------------------------------------------
    # Typography
    # ------------------------------------------------------------------

    def render_text(self, layout_object):
        metadata = layout_object.metadata or {}

        text_object = metadata.get("source_object")

        if text_object is None:
            raise ValueError(
                f"Text object '{layout_object.name}' does not contain "
                "a source TypographyEngine TextObject."
            )

        self.typography.render_text_object(
            image=self.image,
            text_object=text_object,
        )

    # ------------------------------------------------------------------
    # Avatar asset discovery
    # ------------------------------------------------------------------

    def find_avatar_asset(self, avatar_name):
        """
        Find the canonical avatar PNG.

        Expected canonical structure:

            avatars/
                science/
                    canonical/
                        scientist_female_1.png

        The renderer searches recursively so the renderer does not
        need to know about individual avatar categories.
        """

        if not self.avatar_root.exists():
            raise FileNotFoundError(
                f"Avatar root does not exist:\n"
                f"  {self.avatar_root}"
            )

        filename = f"{avatar_name}.png"

        matches = list(self.avatar_root.rglob(filename))

        # Prefer canonical assets when multiple matching files exist.
        canonical_matches = [
            path
            for path in matches
            if "canonical" in path.parts
        ]

        if canonical_matches:
            return canonical_matches[0]

        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Could not find avatar asset '{filename}' under:\n"
            f"  {self.avatar_root}"
        )

    # ------------------------------------------------------------------
    # Avatar rendering
    # ------------------------------------------------------------------

    def render_avatar(self, layout_object):
        """
        Render a real transparent avatar PNG.

        LayoutEngine has already calculated the correct full PNG canvas
        rectangle in layout_object.canvas_rect.

        The important distinction is:

            canvas_rect
                = where the complete transparent PNG belongs

            rect
                = where the visible character belongs

        Therefore we resize and place the complete PNG using
        canvas_rect rather than drawing only the visible bounds.
        """

        avatar_name = layout_object.name

        asset_path = self.find_avatar_asset(avatar_name)

        avatar = Image.open(asset_path).convert("RGBA")

        canvas_rect = layout_object.canvas_rect

        if canvas_rect is None:
            raise ValueError(
                f"Avatar '{avatar_name}' does not contain a canvas_rect."
            )

        target_width = max(1, round(canvas_rect.width))
        target_height = max(1, round(canvas_rect.height))

        avatar = avatar.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )

        paste_x = round(canvas_rect.x)
        paste_y = round(canvas_rect.y)

        self.image.alpha_composite(
            avatar,
            dest=(paste_x, paste_y),
        )

    # ------------------------------------------------------------------
    # Visual placeholder
    # ------------------------------------------------------------------

    def render_visual(self, layout_object):
        """
        Render a placeholder for generated visual assets.

        This will later be replaced with real ComfyUI-generated
        images without changing the scene composition architecture.
        """

        rect = layout_object.rect

        self.draw.rectangle(
            [rect.x, rect.y, rect.right, rect.bottom],
            outline=VISUAL_OUTLINE,
            width=6,
        )

        self.draw.text(
            (rect.x + 10, rect.y + 10),
            "VISUAL",
            fill=VISUAL_OUTLINE,
        )

    # ------------------------------------------------------------------
    # Generic objects
    # ------------------------------------------------------------------

    def render_generic(self, layout_object):
        rect = layout_object.rect

        self.draw.rectangle(
            [rect.x, rect.y, rect.right, rect.bottom],
            outline=LABEL_COLOR,
            width=4,
        )

        self.draw.text(
            (rect.x + 10, rect.y + 10),
            layout_object.name,
            fill=LABEL_COLOR,
        )

    # ------------------------------------------------------------------
    # Full scene render
    # ------------------------------------------------------------------

    def render(self):
        # Constraint evaluation remains here for now so the renderer
        # always receives the current composition state.
        self.composition.evaluate_constraints()

        self.create_canvas()

        render_order = self.composition.get_render_order()

        for layout_object in render_order:
            self.render_object(layout_object)

        return self.image

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_path, image=None):
        if image is None:
            image = self.image

        if image is None:
            raise RuntimeError(
                "No rendered image exists. "
                "Call render() before save()."
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            output_path,
            format="PNG",
        )

        return output_path


# ======================================================================
# Standalone module test
# ======================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("KESTREL SHORTS — SCENE RENDERER")
    print("=" * 60)

    print("\nSceneRenderer module loaded successfully.")

    print("\nArchitecture:")
    print("  CompositionEngine")
    print("       ↓")
    print("  SceneRenderer")
    print("       ↓")
    print("  PNG Frame")

    print("\nCurrent renderer responsibilities:")
    print("  - Canvas creation")
    print("  - Z-order rendering")
    print("  - Typography rendering")
    print("  - Real transparent avatar rendering")
    print("  - Visual placeholder rendering")

    print("\nAvatar system:")
    print("  - Searches avatars/ recursively")
    print("  - Prefers canonical PNG assets")
    print("  - Preserves transparency")
    print("  - Uses full PNG canvas geometry")
    print("  - Keeps visible-content geometry separate")

    print("\nFuture responsibilities:")
    print("  - Generated visuals")
    print("  - Image-to-image assets")
    print("  - Animation")
    print("  - Video frames")

    print("\nSceneRenderer ready.")