"""
The visual vocabulary the Visual Director chooses from, and the Frame
Supervisor and the renderer implement. Names and hints only: sizes,
positions and speeds are numbers owned by code, never by a model.
"""
from typing import Any, Dict, List

# Media arrangements inside one shot. slots = how many images it holds.
LAYOUTS: Dict[str, Dict[str, Any]] = {
    "single": {"slots": (1, 1), "description": "One image fills the frame."},
    "inset": {"slots": (2, 2), "description": "A large image with a small inset image in a corner."},
    "stack_2": {"slots": (2, 2), "description": "Two images, one above the other."},
    "stack_3": {"slots": (3, 3), "description": "Three images stacked."},
    "grid_4": {"slots": (4, 4), "description": "A 2x2 grid; comparisons, lists."},
    "grid_6": {"slots": (6, 6), "description": "A 2x3 grid; six items side by side."},
    "scatter": {"slots": (3, 6), "description": "Photos scattered at slight angles, like a pinned collage."},
    "text_only": {"slots": (0, 0), "description": "No image; text and graphics carry the beat."},
}

IMAGE_MOTIONS: Dict[str, str] = {
    "static": "No movement.",
    "slow_zoom_in": "A slow push in.",
    "slow_zoom_out": "A slow pull back.",
    "pan_left": "A slow pan to the left.",
    "pan_right": "A slow pan to the right.",
    "pan_up": "A slow pan upward.",
    "pan_down": "A slow pan downward.",
    "drift": "A gentle floating drift.",
    "shake_light": "A slight handheld shake.",
    "parallax": "Layers move at different speeds (needs a cut-out subject).",
}

AVATAR_MOTIONS: Dict[str, str] = {
    "static": "No movement.",
    "bob": "A gentle up-and-down bob.",
    "sway": "A slow side-to-side sway.",
    "wobble": "A playful wobble.",
    "slide_in": "Slides in from the edge.",
    "pop_in": "Pops in with a small bounce.",
}

TEXT_MOTIONS: Dict[str, str] = {
    "fade": "Fades in.",
    "pop": "Pops in.",
    "slide_up": "Slides up into place.",
    "typewriter": "Appears letter by letter.",
}

TRANSITIONS: Dict[str, str] = {
    "cut": "A straight cut.",
    "fade": "A cross-fade.",
    "slide": "The next shot slides in.",
    "zoom_through": "Zoom into the current image to reveal the next (body to face to eye).",
    "page_tear": "A paper-tear reveal.",
    "wipe": "A wipe across the frame.",
}

OVERLAYS: Dict[str, str] = {
    "grain": "Film grain.",
    "vignette": "Darkened edges.",
    "paper_texture": "A paper texture.",
    "light_leak": "A soft light leak.",
    "dust": "Floating dust.",
}

# The four kinds of on-screen text.
#   subtitle / kinetic_subtitle follow the narration (the director only picks the style);
#   attention / explanatory carry text the director writes.
TEXT_KINDS: Dict[str, str] = {
    "subtitle": "The narration as regular captions.",
    "kinetic_subtitle": "The narration as animated captions with flair.",
    "attention": "A short punchy phrase or number that grabs attention.",
    "explanatory": "A short label or definition that explains something on screen.",
}

ANCHORS: List[str] = [
    "top_left", "top_center", "top_right", "upper_left", "upper_center", "upper_right",
    "center_left", "center", "center_right", "lower_left", "lower_center", "lower_right",
    "bottom_left", "bottom_center", "bottom_right",
]

AVATAR_SIZES: Dict[str, str] = {"small": "A small presence.", "medium": "A normal presence.", "large": "A dominant presence."}
MEDIA_SOURCES: Dict[str, str] = {
    "generate": "A new image.",
    "reuse": "An earlier image, shown again.",
    "variation": "An image-to-image variation of an earlier image (new angle or small change).",
}
CHANGE_LEVELS = ["small", "medium", "large"]
FOCUS = ["image", "text", "avatar", "balanced"]
# Framings that a still image can really deliver.
FRAMINGS = ["extreme_wide", "wide", "medium", "full_body", "close_up", "extreme_close_up", "detail_insert", "top_down"]


def names(catalog: Dict[str, Any]) -> List[str]:
    return list(catalog)


def layout_for_count(count: int) -> str:
    """The smallest layout that holds `count` images."""
    return {0: "text_only", 1: "single", 2: "stack_2", 3: "stack_3", 4: "grid_4"}.get(count, "grid_6")


def slot_range(layout: str) -> tuple:
    return LAYOUTS.get(layout, LAYOUTS["single"])["slots"]

# ----------------------------------------------------------------------
# Screen elements (Remotion draws them; the Visual Director chooses and fills them)
# ----------------------------------------------------------------------
# data=True: the numbers inside must come from research findings (the director cites finding numbers).

ELEMENT_KINDS: Dict[str, Dict[str, Any]] = {
    "table": {"data": True, "description": "Rows and columns of facts or values.",
              "content": {"columns": ["header", "..."], "rows": [["cell", "..."]]}},
    "list": {"data": False, "description": "A short list of points, bulleted or numbered.",
             "content": {"items": ["point", "..."], "numbered": False}},
    "chart": {"data": True, "description": "A bar or line chart of values.",
              "content": {"type": "bar | line", "labels": ["A", "B"], "values": [1, 2], "unit": "", "title": ""}},
    "stat": {"data": True, "description": "One big number with a caption.",
             "content": {"value": "120 million", "caption": "what it counts"}},
    "timeline": {"data": True, "description": "Events in order along a line.",
                 "content": {"events": [{"when": "1066", "what": "what happened"}]}},
    "comparison": {"data": False, "description": "Two things side by side, point by point.",
                   "content": {"left": {"title": "", "points": [""]}, "right": {"title": "", "points": [""]}}},
    "flow": {"data": False,
             "description": "Steps in a process joined by arrows (for example light, cornea, lens, retina, brain).",
             "content": {"steps": ["step", "..."]}},
    "labels": {"data": False,
               "description": "Short names pointing at parts of an image; the user places them in the preview.",
               "content": {"targets": [{"text": "name"}]}},
}

# What can sit behind a cut-out subject (colours come from the theme's palette roles).
BACKDROPS: Dict[str, str] = {
    "flat": "A flat colour.",
    "gradient": "A soft gradient.",
    "circle": "A large circle behind the subject.",
    "stripes": "Bold diagonal stripes.",
    "grid": "A faint graph-paper grid.",
    "texture": "A texture from the asset library.",
}
BACKDROP_COLORS: List[str] = ["background", "surface", "accent", "accent_alt", "highlight"]

# How a cut-out subject moves.
SUBJECT_MOTIONS: Dict[str, str] = {
    "static": "No movement.",
    "shake_light": "A slight handheld shake.",
    "wobble": "A playful wobble.",
    "bob": "A gentle up-and-down bob.",
    "float": "Slowly floating.",
    "pop_in": "Pops in with a small bounce.",
    "slide_in": "Slides in from the edge.",
}

# Easing curves: the 'feel' of every cut and move.
EASINGS: Dict[str, str] = {
    "linear": "Constant speed.",
    "ease_in_out_cubic": "Smooth start and stop.",
    "ease_out_expo": "Fast start, long soft landing (snappy).",
    "spring": "A small overshoot and settle (playful).",
    "snap": "Almost instant, for hard cuts on key words.",
}

# Particles: small repeated images that float, fall or drift (bubbles, dust, leaves...). The image files come
# from the asset library (kind "particle"); these are the ways they can move.
PARTICLE_MOTIONS: Dict[str, str] = {
    "float_up": "Rise slowly, like bubbles.",
    "fall": "Fall, like rain, snow or leaves.",
    "drift_across": "Drift sideways.",
    "twinkle": "Fade in and out in place.",
}

# ----------------------------------------------------------------------
# More options. Every name needs a matching Remotion component or asset before a video can use it.
# ----------------------------------------------------------------------

ELEMENT_KINDS["list"]["content"] = {"items": ["point", "..."], "marker": "bullet | number | check | cross",
                                    "reveal": "together | one_by_one"}
ELEMENT_KINDS["chart"]["content"]["type"] = "bar | line | pie | donut"
ELEMENT_KINDS.update({
    "counter": {"data": True, "description": "A number that counts up (or down) to its final value.",
                "content": {"from": 0, "to": 100, "unit": "%", "caption": "what it counts"}},
    "ranking": {"data": True, "description": "Items ranked by value as horizontal bars (a top 5).",
                "content": {"items": [{"label": "name", "value": 10}], "unit": ""}},
    "gauge": {"data": True, "description": "A ring or bar showing a value out of a maximum (a share, a score).",
              "content": {"value": 72, "max": 100, "label": "what it measures"}},
    "quote": {"data": True, "description": "A quotation card with its speaker; only real, sourced quotes.",
              "content": {"text": "the quote", "speaker": "who said it", "year": ""}},
    "headline": {"data": True, "description": "A newspaper-style headline card (a real headline, with outlet and date).",
                 "content": {"text": "headline", "outlet": "", "date": ""}},
    "definition": {"data": False, "description": "A term with its plain-words meaning, like a dictionary card.",
                   "content": {"term": "word", "meaning": "plain words", "pronunciation": ""}},
    "callout": {"data": False, "description": "A highlighted phrase, stamp or sticker that stresses one idea.",
                "content": {"text": "short phrase", "style": "highlight | stamp | sticker"}},
    "lower_third": {"data": False, "description": "A name tag for a person, place or object (title and subtitle).",
                    "content": {"title": "name", "subtitle": "role or detail"}},
    "equation": {"data": False, "description": "A formula or equation, typeset.",
                 "content": {"expression": "E = mc^2", "caption": ""}},
})
BACKDROPS.update({
    "dots": "A halftone dot pattern.", "paper": "Plain paper.", "blueprint": "Blueprint blue with fine lines.",
    "spotlight": "A soft spotlight on a dark ground.", "waves": "Slow moving waves.", "noise": "Subtle moving noise.",
})
SUBJECT_MOTIONS.update({
    "tilt": "A slight sway of the angle.", "pulse": "A gentle size pulse.", "spin_in": "Spins in.",
    "drop_in": "Drops in with a bounce.", "slide_out": "Slides out of the frame.", "swing": "Swings like a pendulum.",
})
EASINGS.update({
    "ease_in": "Starts slowly, ends fast.", "ease_out": "Starts fast, ends slowly.",
    "bounce": "Bounces at the end.", "back_out": "Overshoots a little, then settles.",
})
PARTICLE_MOTIONS.update({
    "burst": "Fly outwards from one point.", "orbit": "Circle around the subject.",
    "swirl": "Swirl in a spiral.", "rain_down": "Fall in fast, straight lines.",
})
TEXT_MOTIONS.update({
    "word_by_word": "Words appear one by one.", "highlight_sweep": "A marker stroke sweeps behind the words.",
    "count_up": "Numbers count up.", "shake": "A short shake for emphasis.", "scale_up": "Grows into place.",
})
TRANSITIONS.update({
    "flip": "The frame flips over.", "clock_wipe": "A clock-hand wipe.", "whip_pan": "A fast sideways whip with blur.",
    "flash": "A quick white flash.", "iris": "A circle opens or closes.", "glitch": "A short digital glitch.",
})
LAYOUTS.update({
    "stack": {"slots": (2, 3), "description": "Two or three images stacked."},
    "grid": {"slots": (4, 30), "description": "A grid; columns and rows follow the image count."},
    "sequence": {"slots": (2, 8), "description": "Images shown one after another inside one shot (a quick zoom chain)."},
    "scatter": {"slots": (2, 12), "description": "Photos scattered at slight angles, like a pinned collage."},
})