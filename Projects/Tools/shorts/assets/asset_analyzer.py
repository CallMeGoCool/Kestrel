from pathlib import Path
from PIL import Image


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SHORTS_ROOT = Path(r"C:\AI_Studio\Projects\Tools\shorts")
AVATARS_ROOT = SHORTS_ROOT / "avatars"

# Alpha values below this are treated as transparent.
# 0 = only completely transparent pixels are ignored.
# 10-16 = slightly more tolerant of antialiased edges.
ALPHA_THRESHOLD = 10


# ---------------------------------------------------------
# Asset analysis
# ---------------------------------------------------------

def get_alpha_bbox(image: Image.Image, threshold: int = 10):
    """
    Find the bounding box of visible pixels.

    Returns:
        (left, top, right, bottom)
    or None if the image contains no visible pixels.
    """

    image = image.convert("RGBA")
    alpha = image.getchannel("A")

    if threshold > 0:
        alpha = alpha.point(
            lambda value: 255 if value >= threshold else 0
        )

    return alpha.getbbox()


def analyze_asset(path: Path):
    """
    Analyze a transparent PNG and return geometry information.
    """

    with Image.open(path) as image:
        image = image.convert("RGBA")

        canvas_width, canvas_height = image.size

        bbox = get_alpha_bbox(
            image,
            threshold=ALPHA_THRESHOLD
        )

        if bbox is None:
            return {
                "file": path.name,
                "path": str(path),
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
                "has_content": False,
            }

        left, top, right, bottom = bbox

        content_width = right - left
        content_height = bottom - top

        padding_left = left
        padding_top = top
        padding_right = canvas_width - right
        padding_bottom = canvas_height - bottom

        canvas_area = canvas_width * canvas_height
        content_area = content_width * content_height

        coverage = (
            content_area / canvas_area * 100
            if canvas_area > 0
            else 0
        )

        return {
            "file": path.name,
            "path": str(path),

            "canvas_width": canvas_width,
            "canvas_height": canvas_height,

            "content_x": left,
            "content_y": top,
            "content_width": content_width,
            "content_height": content_height,

            "padding_left": padding_left,
            "padding_top": padding_top,
            "padding_right": padding_right,
            "padding_bottom": padding_bottom,

            "content_width_percent": (
                content_width / canvas_width * 100
            ),

            "content_height_percent": (
                content_height / canvas_height * 100
            ),

            "content_area_percent": coverage,

            "has_content": True,
        }


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def print_analysis(data):
    print()
    print("=" * 70)
    print(f"ASSET: {data['file']}")
    print("=" * 70)

    print(
        f"Canvas:        "
        f"{data['canvas_width']} x {data['canvas_height']}"
    )

    if not data["has_content"]:
        print("Content:       NONE")
        return

    print(
        f"Content box:   "
        f"x={data['content_x']}, "
        f"y={data['content_y']}, "
        f"w={data['content_width']}, "
        f"h={data['content_height']}"
    )

    print()
    print("Transparent padding:")
    print(f"  Left:        {data['padding_left']} px")
    print(f"  Top:         {data['padding_top']} px")
    print(f"  Right:       {data['padding_right']} px")
    print(f"  Bottom:      {data['padding_bottom']} px")

    print()
    print("Visual coverage:")
    print(
        f"  Width:       "
        f"{data['content_width_percent']:.1f}%"
    )

    print(
        f"  Height:      "
        f"{data['content_height_percent']:.1f}%"
    )

    print(
        f"  Area box:    "
        f"{data['content_area_percent']:.1f}%"
    )


# ---------------------------------------------------------
# Test assets
# ---------------------------------------------------------

def main():

    test_assets = [
        AVATARS_ROOT
        / "science"
        / "canonical"
        / "scientist_female_1.png",

        AVATARS_ROOT
        / "history"
        / "canonical"
        / "old_man.png",

        AVATARS_ROOT
        / "genius"
        / "canonical"
        / "genius_male.png",
    ]

    print()
    print("KESTREL SHORTS — ASSET ANALYZER")
    print("=" * 70)
    print(f"Alpha threshold: {ALPHA_THRESHOLD}")
    print()

    for path in test_assets:

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        try:
            data = analyze_asset(path)
            print_analysis(data)

        except Exception as exc:
            print()
            print(f"[ERROR] {path.name}")
            print(f"        {exc}")

    print()
    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()