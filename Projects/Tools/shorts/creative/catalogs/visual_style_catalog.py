"""
Catalog of visual styles available to the Visual Director.

This catalog defines creative options and their intended uses.
It does not generate images or override channel, world, or avatar
identity definitions stored in the registries.
"""

from typing import Any, Dict, Optional

VISUAL_STYLE_CATALOG: Dict[str, Dict[str, Any]] = {
    "default": {
        "name": "Channel Default",
        "description": (
            "Follow the channel's established visual identity and "
            "the selected world's visual conventions. Avoid adding "
            "unnecessary stylistic treatment."
        ),
    },
    "cinematic_realism": {
        "name": "Cinematic Realism",
        "description": (
            "Use a realistic visual treatment with cinematic "
            "lighting, depth, atmosphere, and composition."
        ),
    },
    "documentary": {
        "name": "Documentary",
        "description": (
            "Use a grounded, observational visual treatment suited "
            "to factual storytelling and documentary-style scenes."
        ),
    },
    "historical_reconstruction": {
        "name": "Historical Reconstruction",
        "description": (
            "Depict historically situated environments, objects, "
            "and events using the available historical evidence. "
            "Avoid presenting uncertain details as established facts."
        ),
    },
    "stylized_illustration": {
        "name": "Stylized Illustration",
        "description": (
            "Use a deliberately illustrated visual treatment rather "
            "than aiming for photographic realism."
        ),
    },
    "mythological_fantasy": {
        "name": "Mythological Fantasy",
        "description": (
            "Use an imaginative, myth-inspired visual treatment "
            "suited to legendary beings, symbolic settings, and "
            "mythological storytelling."
        ),
    },
    "scientific_visualization": {
        "name": "Scientific Visualization",
        "description": (
            "Present scientific subjects through clear visual "
            "representations. Distinguish evidence-based depictions "
            "from illustrative or hypothetical reconstructions."
        ),
    },
    "graphic_conceptual": {
        "name": "Graphic and Conceptual",
        "description": (
            "Use deliberate graphic forms, visual metaphors, and "
            "simplified representations to communicate abstract ideas."
        ),
    },
    "none": {
        "name": "No Additional Style",
        "description": (
            "Request no additional style treatment. Preserve necessary "
            "content and follow applicable channel and world constraints."
        ),
    },
}


def get_visual_style_catalog() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the available visual style definitions."""
    return {
        style_id: dict(style)
        for style_id, style in VISUAL_STYLE_CATALOG.items()
    }


def get_visual_style_options() -> Dict[str, Dict[str, Any]]:
    """Return all available visual style options."""
    return get_visual_style_catalog()


def get_visual_style_option_names() -> list[str]:
    """Return the available visual style IDs."""
    return list(VISUAL_STYLE_CATALOG.keys())


def is_valid_visual_style(
    style_id: Optional[str],
) -> bool:
    """Check whether a visual style ID exists in the catalog."""
    return (
        isinstance(style_id, str)
        and style_id in VISUAL_STYLE_CATALOG
    )