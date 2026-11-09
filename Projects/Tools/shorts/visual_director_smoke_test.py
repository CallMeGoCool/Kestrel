import json
import traceback
from pathlib import Path
from datetime import datetime

import core.config
from core.models.script import Script, ScriptSection
from core.models.character import Character
from core.models.channel import Channel
from core.models.world import World
from core.models.story_plan import StoryPlan
from core.run_context import RunContext
from agents.visual_director import run_visual_director

# ------------------------------------------------------------
# Test configuration: local to this test only
# ------------------------------------------------------------

core.config.OLLAMA_TIMEOUT = 900

root = Path.cwd()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = (
    root / "rendering" / "visual_director_smoke_test" / timestamp
)
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("KESTREL SHORTS — SMALL VISUAL DIRECTOR TEST")
print("=" * 70)
print(f"Model:   {core.config.OLLAMA_MODEL}")
print(f"Timeout: {core.config.OLLAMA_TIMEOUT} seconds")
print("Scope:   Three beats; no Reviewer, ComfyUI, or rendering")
print(f"Output:  {output_dir}")
print()

# Synthetic test character. No registered avatar/pose is claimed.
character = Character(
    character_id="smoke_test_host",
    display_name="Test Host",
    description="A presenter explaining a solar eclipse.",
    avatar_id="",
)

channel = Channel(
    channel_id="smoke_test_science",
    name="Science Shorts Test",
    family_id="science",
    description="Short, clear science explainers.",
    topic_focus=("science", "space"),
    available_character_ids=("smoke_test_host",),
    storytelling_style="Curiosity-driven and concise.",
    visual_style="cinematic_realism",
)

world = World(
    world_id="smoke_test_space",
    name="Space Science",
    description="A scientifically grounded space setting.",
    visual_aesthetic="Cinematic scientific documentary.",
    atmosphere="Mysterious and awe-inspiring.",
)

script = Script(
    title="A Solar Eclipse",
    sections=[
        ScriptSection(
            section_id="beat_1",
            text="What happens when the Sun suddenly disappears?",
            speaker_character_id="smoke_test_host",
            performance_intent="Create curiosity.",
        ),
        ScriptSection(
            section_id="beat_2",
            text=(
                "A total solar eclipse occurs when the Moon passes "
                "between Earth and the Sun."
            ),
            speaker_character_id="smoke_test_host",
            performance_intent="Explain the science clearly.",
        ),
        ScriptSection(
            section_id="beat_3",
            text=(
                "During totality, the Sun's faint outer atmosphere "
                "can become visible."
            ),
            speaker_character_id="smoke_test_host",
            performance_intent="Deliver a memorable visual payoff.",
        ),
    ],
)

story_plan = StoryPlan(
    title="A Solar Eclipse",
    premise="Explain a total solar eclipse in three concise beats.",
    hook="What happens when the Sun seems to disappear?",
    narrative_goal="Move from curiosity to explanation to visual payoff.",
    story_beats=[
        "Establish the surprising disappearance of sunlight.",
        "Explain the alignment of the Sun, Moon, and Earth.",
        "Reveal the solar corona during totality.",
    ],
    ending="An eclipse briefly reveals the Sun's outer atmosphere.",
)

context = RunContext(
    original_prompt=(
        "Create a concise three-beat science short about a total solar "
        "eclipse. Produce a coherent visual plan and a generation manifest "
        "for any assets that genuinely need generation."
    ),
    explicit_preferences={
        "visual_style": "cinematic_realism",
        "scientific_accuracy": "high",
    },
    constraints=[
        "Use exactly the supplied three script sections.",
        "Do not invent registered avatar or existing asset IDs.",
        "Do not call ComfyUI or render images.",
    ],
)

# ------------------------------------------------------------
# Call Visual Director
# ------------------------------------------------------------

try:
    print("Calling Visual Director. Please wait...\n")

    result = run_visual_director(
        context=context,
        script=script,
        channel=channel,
        world=world,
        characters=[character],
        story_plan=story_plan,
        revision_feedback=None,
        previous_plan=None,
        phase="draft",
        num_predict=4000,
        registered_assets=[],
    )

    output_path = output_dir / "visual_director_draft.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    visual_plan = result.get("visual_plan", {})
    manifest = result.get("generation_manifest", {})
    shots = visual_plan.get("shots", [])
    targets = manifest.get("generation_targets", [])

    print("=" * 70)
    print("DRAFT GENERATED")
    print("=" * 70)
    print(f"Saved to: {output_path}")
    print(f"Returned phase: {result.get('phase', 'not specified')}")
    print(f"Shots: {len(shots)}")
    print(f"Generation targets: {len(targets)}")

    print("\nSHOT SUMMARY")
    for shot in shots:
        print(
            f"- {shot.get('shot_id')} | "
            f"section={shot.get('section_id')} | "
            f"type={shot.get('shot_type')} | "
            f"composition={shot.get('composition_mode')}"
        )
        print(f"  Content: {shot.get('visual_content')}")
        print(
            f"  Characters: {len(shot.get('character_staging', []))}; "
            f"assets: {len(shot.get('asset_requirements', []))}; "
            f"text: {len(shot.get('text_elements', []))}; "
            f"effects: {len(shot.get('visual_effects', []))}"
        )

    print("\nGENERATION TARGETS")
    for target in targets:
        print(f"- Asset: {target.get('asset_id')}")
        print(f"  Method: {target.get('generation_method')}")
        print(f"  Purpose: {target.get('purpose')}")
        print(f"  Positive prompt: {target.get('positive_prompt')}")
        print(f"  Negative prompt: {target.get('negative_prompt')}")
        print(f"  Model: {target.get('model')}")
        print(f"  LoRAs: {target.get('loras')}")
        print(f"  Dimensions: {target.get('dimensions')}")
        print(f"  Output: {target.get('output')}")

    print("\nThe complete draft is saved in the JSON file.")
    print("No Reviewer, ComfyUI, or renderer was called.")
    print(
        "Character staging is not fully validated because this test "
        "character has no registered avatar/pose inventory."
    )

except Exception as exc:
    error_path = output_dir / "visual_director_error.txt"
    error_path.write_text(traceback.format_exc(), encoding="utf-8")

    print(f"\nTEST FAILED: {type(exc).__name__}: {exc}")
    print(f"Full traceback: {error_path}")
    raise

