from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ContentFamily:
    """
    A broad creative/content archetype used as a foundation for channels.

    A ContentFamily does not define a specific channel, topic list, character,
    or world. It provides a broad creative lens that a Channel can specialize.
    """

    family_id: str
    name: str
    description: str
    creative_lens: str
    audience_profile_id: str
    topic_examples: Tuple[str, ...] = ()


DRAGONS = ContentFamily(
    family_id="dragons",
    name="Fantasy & Fairy Tales",
    description="Fantasy, fairy tales, magical creatures, and imaginative storytelling.",
    creative_lens="Imagination",
    audience_profile_id="children",
    topic_examples=(
        "fairy tales",
        "dragons",
        "magical creatures",
        "fantasy adventures",
        "fantasy folklore",
    ),
)

GUESTS = ContentFamily(
    family_id="guests",
    name="Kids Adventure & Learning",
    description="Simple adventures, curiosity, basic learning, and age-appropriate life lessons.",
    creative_lens="Curiosity",
    audience_profile_id="children",
    topic_examples=(
        "simple adventures",
        "basic science",
        "nature",
        "life lessons",
        "children's stories",
        "curiosity questions",
    ),
)

MYTHOLOGY = ContentFamily(
    family_id="mythology",
    name="Mythology & Ancient Beliefs",
    description="Myths, legends, gods, monsters, ancient cultures, and traditional beliefs.",
    creative_lens="Belief & Wonder",
    audience_profile_id="teen_plus",
    topic_examples=(
        "Greek mythology",
        "Norse mythology",
        "Egyptian mythology",
        "Hindu mythology",
        "ancient beliefs",
        "creation myths",
        "legends",
    ),
)

HISTORY = ContentFamily(
    family_id="history",
    name="History",
    description="Historical events, civilizations, rulers, conflicts, and forgotten stories.",
    creative_lens="The Past",
    audience_profile_id="teen_plus",
    topic_examples=(
        "wars",
        "empires",
        "civilizations",
        "rulers",
        "historical disasters",
        "forgotten events",
        "historical mysteries",
    ),
)

MYSTERY = ContentFamily(
    family_id="mystery",
    name="Mystery & The Unknown",
    description="Unsolved mysteries, unexplained events, strange cases, and investigative stories.",
    creative_lens="The Unknown",
    audience_profile_id="teen_plus",
    topic_examples=(
        "unsolved mysteries",
        "disappearances",
        "unexplained events",
        "strange cases",
        "internet mysteries",
        "historical mysteries",
    ),
)

SCIENCE = ContentFamily(
    family_id="science",
    name="Science & Discovery",
    description="Science, space, nature, medicine, experiments, and scientific discoveries.",
    creative_lens="Discovery",
    audience_profile_id="teen_plus",
    topic_examples=(
        "physics",
        "biology",
        "astronomy",
        "space",
        "scientific discoveries",
        "strange animals",
        "experiments",
    ),
)

FORGE = ContentFamily(
    family_id="forge",
    name="Engineering & Creation",
    description="Engineering, inventions, machines, manufacturing, construction, and making things.",
    creative_lens="Creation",
    audience_profile_id="teen_plus",
    topic_examples=(
        "engineering",
        "inventions",
        "machines",
        "manufacturing",
        "construction",
        "industrial history",
        "how things are made",
    ),
)

GENIUS = ContentFamily(
    family_id="genius",
    name="Ideas, Technology & Business",
    description="Technology, business, economics, psychology, strategy, and influential ideas.",
    creative_lens="Ideas",
    audience_profile_id="teen_plus",
    topic_examples=(
        "technology",
        "business",
        "economics",
        "psychology",
        "entrepreneurs",
        "companies",
        "strategy",
        "innovation",
    ),
)

SURVIVAL = ContentFamily(
    family_id="survival",
    name="Survival & Extremes",
    description="Survival stories, disasters, expeditions, extreme environments, and human resilience.",
    creative_lens="Extremes & Resilience",
    audience_profile_id="teen_plus",
    topic_examples=(
        "wilderness survival",
        "disasters",
        "shipwrecks",
        "plane crashes",
        "expeditions",
        "extreme environments",
        "survival psychology",
    ),
)


CONTENT_FAMILIES = {
    family.family_id: family
    for family in (
        DRAGONS,
        GUESTS,
        MYTHOLOGY,
        HISTORY,
        MYSTERY,
        SCIENCE,
        FORGE,
        GENIUS,
        SURVIVAL,
    )
}