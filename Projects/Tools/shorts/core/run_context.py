"""
Kestrel Shorts - Run Context
============================

The shared, per-run state that agents read and the workflow controller
writes. It holds DECISIONS and SHORT SUMMARIES. Large outputs (full stage
results, images, audio, clips) live as files in the run folder:

    runtime/runs/<date>_<topic-slug>_<id>/
        shared_context.json
        stages/<stage>.json
        images/  audio/  clips/
        report_catchy.md   report_summary.md     <- kept after cleanup

When a video is finished, `cleanup()` deletes everything except the two
reports.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"
REPORT_FILES = ("report_catchy.md", "report_summary.md")
DURATION_SOURCES = ("user", "architect", "story_director", "default")
_STAGE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    return uuid.uuid4().hex


def _slugify(text: str, max_len: int = 32) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len].strip("-")
    return slug or "video"


def _atomic_write(path: Path, text: str) -> None:
    """Write text atomically; retry briefly if Windows holds the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(4):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.2 * (attempt + 1))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


@dataclass
class RunContext:
    """Shared state for a single Kestrel Shorts run."""

    original_prompt: str

    run_id: str = field(default_factory=generate_run_id)
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    # When and where this run started (local time, timezone, location).
    time_context: Dict[str, Any] = field(default_factory=config.now_stamp)

    explicit_preferences: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)

    # Written by the Orchestrator
    interpretation: Dict[str, Any] = field(default_factory=dict)
    channel_selection: Dict[str, Any] = field(default_factory=dict)  # channel_id, world_id, rationale
    contextual_search: Dict[str, Any] = field(default_factory=dict)
    research_brief: Dict[str, Any] = field(default_factory=dict)
    clarification: Dict[str, Any] = field(default_factory=dict)      # required, question

    # Written by the Architect (or the user's request)
    duration: Dict[str, Any] = field(default_factory=dict)

    workflow: Dict[str, Any] = field(
        default_factory=lambda: {
            "current_stage": "orchestrator",
            "status": "created",
            "completed_stages": [],
            "failed_stages": [],
            "pending_review": False,
        }
    )

    artifacts: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)

    finalization: Dict[str, Any] = field(
        default_factory=lambda: {
            "status": "pending",
            "video_artifact": None,
            "completed_at": None,
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.original_prompt, str):
            raise TypeError("original_prompt must be a string.")
        if not self.original_prompt.strip():
            raise ValueError("original_prompt cannot be empty.")
        if not self.run_id or not self.run_id.strip():
            raise ValueError("run_id cannot be empty.")

    # ------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------

    @property
    def run_directory(self) -> Path:
        date = self.created_at[:10].replace("-", "")
        name = f"{date}_{_slugify(self.original_prompt)}_{self.run_id[:6]}"
        return Path(config.RUNS_DIR) / name

    @property
    def context_path(self) -> Path:
        return self.run_directory / "shared_context.json"

    @property
    def stages_directory(self) -> Path:
        return self.run_directory / "stages"

    @property
    def images_directory(self) -> Path:
        return self.run_directory / "images"

    @property
    def audio_directory(self) -> Path:
        return self.run_directory / "audio"

    @property
    def clips_directory(self) -> Path:
        return self.run_directory / "clips"

    @property
    def artifacts_directory(self) -> Path:   # kept for older callers
        return self.run_directory / "artifacts"

    # ------------------------------------------------------------
    # Decisions: one accessor per decision, so agents never guess keys
    # ------------------------------------------------------------

    @property
    def time_line(self) -> str:
        """One-line 'now' for prompts, e.g. 'Saturday 2026-10-17 19:05 (UTC+0530), Chennai, India'."""
        return config.describe_stamp(self.time_context)

    @property
    def channel_id(self) -> Optional[str]:
        return self._first_text(self.channel_selection, "channel_id", "selected_channel_id")

    @property
    def world_id(self) -> Optional[str]:
        return self._first_text(self.channel_selection, "world_id", "selected_world_id")

    @staticmethod
    def _first_text(source: Dict[str, Any], *keys: str) -> Optional[str]:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def select_channel(self, channel_id: str, rationale: str = "") -> None:
        if not isinstance(channel_id, str) or not channel_id.strip():
            raise ValueError("channel_id must be a non-empty string.")
        self.channel_selection["channel_id"] = channel_id.strip()
        if rationale:
            self.channel_selection["rationale"] = rationale.strip()
        self.updated_at = utc_now()

    def select_world(self, world_id: str, rationale: str = "") -> None:
        if not isinstance(world_id, str) or not world_id.strip():
            raise ValueError("world_id must be a non-empty string.")
        self.channel_selection["world_id"] = world_id.strip()
        if rationale:
            self.channel_selection["world_rationale"] = rationale.strip()
        self.updated_at = utc_now()

    def set_duration(
        self,
        target_seconds: float,
        source: str = "architect",
        reason: str = "",
    ) -> Dict[str, Any]:
        """Record the chosen length, clamped to limits, with its word budget."""
        if source not in DURATION_SOURCES:
            raise ValueError(f"source must be one of {DURATION_SOURCES}.")
        if (
            isinstance(target_seconds, bool)
            or not isinstance(target_seconds, (int, float))
            or target_seconds <= 0
        ):
            raise ValueError("target_seconds must be a positive number.")

        target = config.clamp_duration(target_seconds)
        self.duration = {
            "target_seconds": target,
            "requested_seconds": float(target_seconds),
            "clamped": target != float(target_seconds),
            "word_budget": config.word_budget(target),
            "min_seconds": config.MIN_VIDEO_SECONDS,
            "max_seconds": config.MAX_VIDEO_SECONDS,
            "source": source,
            "reason": reason.strip(),
        }
        self.updated_at = utc_now()
        return dict(self.duration)

    @property
    def word_budget(self) -> Optional[int]:
        value = self.duration.get("word_budget")
        return value if isinstance(value, int) else None

    # ------------------------------------------------------------
    # Large outputs live in files
    # ------------------------------------------------------------

    def stage_output_path(self, stage: str) -> Path:
        if not isinstance(stage, str) or not _STAGE_NAME.match(stage):
            raise ValueError(f"Invalid stage name: {stage!r}")
        return self.stages_directory / f"{stage}.json"

    def write_stage_output(self, stage: str, data: Any) -> Path:
        """Save a stage's full output as a file and return its path."""
        path = self.stage_output_path(stage)
        _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")
        return path

    def read_stage_output(self, stage: str) -> Any:
        """Load a stage's saved output, or None if it does not exist."""
        path = self.stage_output_path(stage)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[Path | str] = None) -> Path:
        destination = Path(path).resolve() if path is not None else self.context_path
        self.updated_at = utc_now()
        _atomic_write(
            destination,
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> "RunContext":
        """
        Load a saved context. Unknown fields are ignored with a warning, so
        adding a field never makes earlier runs unloadable. A different
        schema_version is rejected with a clear message.
        """
        source = Path(path).resolve()
        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Shared context must contain a JSON object.")

        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported context schema version {data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION!r}. Start a new run."
            )

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            logger.warning("Ignoring unknown context fields: %s", ", ".join(unknown))

        return cls(**{k: v for k, v in data.items() if k in known})

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """Merge values into a supported dictionary section."""
        supported = {
            "explicit_preferences", "interpretation", "channel_selection",
            "contextual_search", "research_brief", "clarification", "duration",
            "workflow", "artifacts", "stage_results", "finalization",
        }
        if section not in supported:
            raise ValueError(f"Unsupported context section: {section!r}")
        if not isinstance(values, dict):
            raise TypeError("Section updates must be dictionaries.")
        getattr(self, section).update(values)
        self.updated_at = utc_now()

    # ------------------------------------------------------------
    # End of run: reports and cleanup
    # ------------------------------------------------------------

    def plain_summary(self) -> str:
        """A plain-text summary built from the context (no LLM)."""
        topic = self.interpretation.get("topic") or self.original_prompt
        research = (
            self.read_stage_output("researcher")
            or (self.stage_results.get("researcher") or {}).get("result")
            or {}
        )
        script = (
            self.read_stage_output("story_director")
            or (self.stage_results.get("story_director") or {}).get("result")
            or {}
        )

        characters: List[str] = []
        for section in script.get("sections", []) if isinstance(script, dict) else []:
            speaker = section.get("speaker_character_id") if isinstance(section, dict) else None
            if speaker and speaker not in characters:
                characters.append(speaker)

        findings = [str(f) for f in (research.get("findings") or [])[:5]] if isinstance(research, dict) else []
        sources = research.get("sources") or [] if isinstance(research, dict) else []

        duration = self.duration
        length = (
            f"{duration['target_seconds']:.0f}s target ({duration.get('source', '?')})"
            if duration.get("target_seconds") else "not recorded"
        )

        lines = [
            f"# {topic}",
            "",
            f"- Run: {self.run_id}",
            f"- Channel: {self.channel_id or 'n/a'}",
            f"- World: {self.world_id or 'n/a'}",
            f"- Characters: {', '.join(characters) if characters else 'n/a'}",
            f"- Length: {length}",
            f"- Sources used: {len(sources)}",
            f"- Video: {self.finalization.get('video_artifact') or 'n/a'}",
            f"- Finished: {self.finalization.get('completed_at') or 'n/a'}",
        ]
        if findings:
            lines += ["", "## Key facts", *[f"- {f}" for f in findings]]
        return "\n".join(lines) + "\n"

    def write_reports(self, catchy: str, summary: Optional[str] = None) -> List[Path]:
        """Write the two end-of-run reports into the run folder."""
        summary = summary if summary is not None else self.plain_summary()
        paths = []
        for name, text in zip(REPORT_FILES, (catchy, summary)):
            path = self.run_directory / name
            _atomic_write(path, text.rstrip() + "\n")
            paths.append(path)
        return paths

    def cleanup(self, keep: tuple = REPORT_FILES) -> List[str]:
        """
        Delete everything in the run folder except the reports. Only allowed
        once the workflow is complete. Do not call save() afterwards.
        """
        if self.workflow.get("status") not in ("completed", "completed_with_warnings"):
            raise RuntimeError("cleanup() is only allowed after the workflow has completed.")
        directory = self.run_directory
        removed: List[str] = []
        if not directory.is_dir():
            return removed
        for item in directory.iterdir():
            if item.name in keep:
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
            removed.append(item.name)
        return removed