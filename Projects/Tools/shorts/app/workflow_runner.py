from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from app.workflow_controller import WorkflowController
from core.models.research_result import ResearchResult
from core.run_context import RunContext, utc_now


logger = logging.getLogger(__name__)


class WorkflowRunnerError(RuntimeError):
    """Base exception for workflow runner errors."""


class MissingStageHandlerError(WorkflowRunnerError):
    """Raised when a required workflow stage has no implementation."""


class StageExecutionError(WorkflowRunnerError):
    """Raised when a workflow stage fails."""


class WorkflowRunner:
    """
    Execute the configured Kestrel Shorts workflow.

    The application Orchestrator runs before this downstream pipeline.
    Its prepared inputs must already be present in RunContext.

    WorkflowController manages persistent state and stage transitions.
    The runner executes agents and passes their outputs downstream.

    Successful stages may finish normally or with warnings. A stage
    completed with warnings still provides a usable result to downstream
    stages and does not count as a failed stage.
    """

    DEFAULT_STAGE_ORDER = (
        "orchestrator",
        "researcher",
        "architect",
        "script_refiner",
        "story_director",
    )

    SUCCESSFUL_STAGE_STATUSES = (
        "completed",
        "completed_with_warnings",
    )

    def __init__(
        self,
        controller: WorkflowController,
        stage_handlers: Optional[
            Mapping[str, Callable[..., Any]]
        ] = None,
        *,
        stage_order: Optional[tuple[str, ...] | list[str]] = None,
        handler_kwargs: Optional[
            Mapping[str, Mapping[str, Any]]
        ] = None,
    ) -> None:
        if not isinstance(controller, WorkflowController):
            raise TypeError(
                "controller must be a WorkflowController instance."
            )

        self.controller = controller
        self.context = controller.context

        self.stage_handlers = self._build_default_handlers()

        if stage_handlers:
            self.stage_handlers.update(stage_handlers)

        self.handler_kwargs = {
            stage: dict(kwargs)
            for stage, kwargs in (handler_kwargs or {}).items()
        }

        self.stage_order = tuple(
            stage_order
            if stage_order is not None
            else self.DEFAULT_STAGE_ORDER
        )

        self._validate_configuration()

    # --------------------------------------------------------
    # CREATION AND RESUMPTION
    # --------------------------------------------------------

    @classmethod
    def create(
        cls,
        original_prompt: str,
        explicit_preferences: Optional[Dict[str, Any]] = None,
        constraints: Optional[list[str]] = None,
        *,
        stage_handlers: Optional[
            Mapping[str, Callable[..., Any]]
        ] = None,
        stage_order: Optional[tuple[str, ...] | list[str]] = None,
        handler_kwargs: Optional[
            Mapping[str, Mapping[str, Any]]
        ] = None,
    ) -> "WorkflowRunner":
        """Create a new persisted run and its workflow runner."""

        controller = WorkflowController.create_run(
            original_prompt=original_prompt,
            explicit_preferences=explicit_preferences,
            constraints=constraints,
        )

        return cls(
            controller,
            stage_handlers=stage_handlers,
            stage_order=stage_order,
            handler_kwargs=handler_kwargs,
        )

    @classmethod
    def resume(
        cls,
        context_path: Path | str,
        *,
        stage_handlers: Optional[
            Mapping[str, Callable[..., Any]]
        ] = None,
        stage_order: Optional[tuple[str, ...] | list[str]] = None,
        handler_kwargs: Optional[
            Mapping[str, Mapping[str, Any]]
        ] = None,
    ) -> "WorkflowRunner":
        """Load a persisted run without automatically executing it."""

        controller = WorkflowController.resume_run(context_path)

        return cls(
            controller,
            stage_handlers=stage_handlers,
            stage_order=stage_order,
            handler_kwargs=handler_kwargs,
        )

    # --------------------------------------------------------
    # AGENT REGISTRATION
    # --------------------------------------------------------

    @staticmethod
    def _build_default_handlers() -> Dict[str, Callable[..., Any]]:
        """
        Register adapters for the currently integrated agents.

        Imports are lazy so importing the runner does not initialize
        agents or their external dependencies.
        """

        return {
            "orchestrator": WorkflowRunner._run_orchestrator,
            "researcher": WorkflowRunner._run_researcher,
            "architect": WorkflowRunner._run_architect,
            "script_refiner": WorkflowRunner._run_script_refiner,
            "story_director": WorkflowRunner._run_story_director,
        }

    @staticmethod
    def _run_orchestrator(*, context: RunContext, **_: Any) -> Any:
        from app.orchestrator import run_orchestrator

        return run_orchestrator(context)

    @staticmethod
    def _run_researcher(
        *,
        topic: str,
        context: RunContext,
        **_: Any,
    ) -> Any:
        """Adapt the Researcher interface."""

        from agents.researcher import run_research

        return run_research(
            topic=topic,
            context=context,
            context_path=context.context_path,
        )

    @staticmethod
    def _run_architect(
        *,
        research: Any,
        context: RunContext,
        **_: Any,
    ) -> Any:
        """Adapt the Architect interface."""

        from agents.architect import run_architect

        typed_research = WorkflowRunner._as_research_result(
            research
        )

        return run_architect(
            research=typed_research,
            context=context,
        )

    @staticmethod
    def _run_script_refiner(*, story_plan: Any, research: Any, context: RunContext, **_: Any) -> Any:
        from agents.script_refiner import run_script_refiner
        from core.models.story_plan import StoryPlan

        return run_script_refiner(
            story_plan=WorkflowRunner._as_dataclass(story_plan, StoryPlan, label="StoryPlan"),
            research=WorkflowRunner._as_research_result(research),
            context=context,
        )

    @staticmethod
    def _run_story_director(
        *,
        story_plan: Any,
        research: Any,
        script: Any,
        channel: Any,
        world: Any,
        characters: Any,
        context: RunContext,
        voice_catalog: Optional[Mapping[str, Any]] = None,
        revision_feedback: Any = None,
        previous_script: Any = None,
        phase: str = "draft",
        num_predict: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Adapt the Story Director interface.

        The runner supplies the voice catalogue as data. The Story Director
        selects validated creative directions; the catalogue itself does
        not generate audio or determine backend-specific TTS settings.
        """

        from agents.story_director import run_story_director
        from core.models.story_plan import StoryPlan
        from core.models.script import Script
        from core.models.channel import Channel
        from core.models.world import World
        from core.models.character import Character
        from creative.catalogs.voice_catalog import get_voice_catalog

        typed_story_plan = WorkflowRunner._as_dataclass(
            story_plan,
            StoryPlan,
            label="StoryPlan",
        )

        typed_research = WorkflowRunner._as_research_result(
            research
        )

        typed_script = WorkflowRunner._as_dataclass(
            script,
            Script,
            label="Script",
        )

        typed_channel = WorkflowRunner._as_dataclass(
            channel,
            Channel,
            label="Channel",
        )

        typed_world = WorkflowRunner._as_dataclass(
            world,
            World,
            label="World",
        )

        if characters is None:
            typed_characters = []
        elif not isinstance(characters, (list, tuple)):
            raise TypeError(
                "Story Director characters must be a list or tuple."
            )
        else:
            typed_characters = [
                WorkflowRunner._as_dataclass(
                    character,
                    Character,
                    label="Character",
                )
                for character in characters
            ]

        if previous_script is not None:
            previous_script = WorkflowRunner._as_dataclass(
                previous_script,
                Script,
                label="previous Script",
            )

        reserved_keys = {
            "story_plan",
            "research",
            "script",
            "channel",
            "world",
            "characters",
            "context",
            "voice_catalog",
            "revision_feedback",
            "previous_script",
            "phase",
            "num_predict",
        }

        conflicting_keys = reserved_keys.intersection(kwargs)

        if conflicting_keys:
            raise WorkflowRunnerError(
                "handler_kwargs cannot override required Story Director "
                f"inputs: {sorted(conflicting_keys)}"
            )

        # The catalogue can be injected for tests or specialized runs.
        # Otherwise load the project's shared source of truth.
        resolved_voice_catalog = (
            dict(voice_catalog)
            if voice_catalog is not None
            else get_voice_catalog()
        )

        if not isinstance(resolved_voice_catalog, Mapping):
            raise TypeError(
                "The voice catalogue must be a mapping."
            )

        return run_story_director(
            context=context,
            research=typed_research,
            story_plan=typed_story_plan,
            script=typed_script,
            channel=typed_channel,
            world=typed_world,
            characters=typed_characters,
            voice_catalog=resolved_voice_catalog,
            revision_feedback=revision_feedback,
            previous_script=previous_script,
            phase=phase,
            num_predict=num_predict,
        )

    def register_handler(
        self,
        stage_name: str,
        handler: Callable[..., Any],
        *,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Register or replace a handler for an executable stage."""

        if stage_name not in self._executable_stages():
            raise ValueError(
                f"Cannot register handler for stage {stage_name!r}."
            )

        if not callable(handler):
            raise TypeError("handler must be callable.")

        self.stage_handlers[stage_name] = handler

        if kwargs is not None:
            self.handler_kwargs[stage_name] = dict(kwargs)

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the configured stage sequence.

        Stages completed normally or with warnings are skipped on
        subsequent execution. Warning-completed results remain available
        to downstream stages. Clarification requests pause execution.
        """

        status = self.context.workflow.get("status", "created")

        if status in (
            "completed",
            "completed_with_warnings",
            "awaiting_clarification",
        ):
            return self.get_status()

        if status == "failed":
            raise WorkflowRunnerError(
                "The workflow has failed. Call resume_after_failure() "
                "before retrying."
            )

        if status == "running":
            current_stage = self.context.workflow.get("current_stage")
            current_result = self.context.stage_results.get(
                current_stage, {}
            )

            if current_result.get("status") == "running":
                raise WorkflowRunnerError(
                    f"Stage {current_stage!r} is already marked as "
                    "running. Recover the interrupted stage first."
                )

        self.controller.start()

        for stage_name in self.stage_order:
            existing = self.context.stage_results.get(
                stage_name, {}
            )

            if existing.get("status") in self.SUCCESSFUL_STAGE_STATUSES:
                continue

            handler = self.stage_handlers.get(stage_name)

            if handler is None:
                raise MissingStageHandlerError(
                    f"No handler is registered for stage "
                    f"{stage_name!r}. Earlier results are preserved."
                )

            inputs = self._build_stage_inputs(stage_name)

            outcome = self._execute_stage(
                stage_name,
                handler,
                inputs=inputs,
            )

            if outcome.get("requires_clarification"):
                reason = (
                    outcome.get("clarification_question")
                    or outcome.get("reason")
                    or "Additional user clarification is required."
                )

                self.controller.request_clarification(str(reason))

                return self.get_status()

        self.controller.complete_workflow()

        return self.get_status()

    def resume_after_failure(self) -> Dict[str, Any]:
        """Prepare a failed stage for retry and continue execution."""

        self.controller.resume_after_failure()

        logger.info(
            "Resuming run %s after failure recovery",
            self.context.run_id,
        )

        return self.run()

    def resolve_clarification(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """Record the user's answer and continue the workflow."""

        self.controller.resolve_clarification(response)

        return self.run()

    def get_status(self) -> Dict[str, Any]:
        """Return a compact workflow status summary."""

        workflow = self.context.workflow

        return {
            "run_id": self.context.run_id,
            "status": workflow.get("status"),
            "current_stage": workflow.get("current_stage"),
            "completed_stages": list(
                workflow.get("completed_stages", [])
            ),
            "failed_stages": list(
                workflow.get("failed_stages", [])
            ),
            "clarification_reason": workflow.get(
                "clarification_reason"
            ),
            "failure_reason": workflow.get("failure_reason"),
            "context_path": str(self.controller.context_path),
        }

    # --------------------------------------------------------
    # STAGE EXECUTION
    # --------------------------------------------------------

    def _execute_stage(
        self,
        stage_name: str,
        handler: Callable[..., Any],
        *,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Execute a stage and persist its serializable result."""

        self.controller.begin_stage(stage_name)

        try:
            kwargs = dict(inputs)
            kwargs.update(
                self.handler_kwargs.get(stage_name, {})
            )

            result = handler(**kwargs)
            normalized = self._normalize_result(result)

            self._store_stage_context(stage_name, normalized)

            stage_status = self._determine_stage_status(
                stage_name,
                normalized,
            )

            self.controller.complete_stage(
                stage_name=stage_name,
                result=normalized,
                status=stage_status,
            )

            logger.info(
                "Run %s completed stage %s with status %s",
                self.context.run_id,
                stage_name,
                stage_status,
            )

            if isinstance(normalized, dict):
                return normalized

            return {"result": normalized}

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            logger.exception(
                "Run %s failed at stage %s",
                self.context.run_id,
                stage_name,
            )

            try:
                self.controller.fail_stage(
                    stage_name,
                    error_message,
                )
            except Exception:
                logger.exception(
                    "Failed to persist error state for stage %s",
                    stage_name,
                )

            raise StageExecutionError(
                f"Stage {stage_name!r} failed: {error_message}"
            ) from exc

    @classmethod
    def _determine_stage_status(
        cls,
        stage_name: str,
        result: Any,
    ) -> str:
        """
        Determine whether a usable stage result completed with warnings.

        Warning metadata does not turn a successful execution into a
        failure. The existing result and its metadata are preserved.
        """

        if not isinstance(result, Mapping):
            return "completed"

        metadata = result.get("metadata", {})

        if not isinstance(metadata, Mapping):
            metadata = {}

        if result.get("status") in cls.SUCCESSFUL_STAGE_STATUSES:
            return result["status"]

        if metadata.get("status") == "completed_with_warnings":
            return "completed_with_warnings"

        if stage_name == "researcher":
            researcher_status = metadata.get("stage_status")

            if researcher_status == "completed":
                return "completed"

            if researcher_status == "completed_with_warnings":
                return "completed_with_warnings"

            if researcher_status == "failed":
                raise StageExecutionError(
                    "Researcher did not produce a usable research result."
                )

        if stage_name == "researcher":
            source_count = metadata.get("research_source_count")

            if (
                isinstance(source_count, (int, float))
                and not isinstance(source_count, bool)
                and source_count <= 0
            ):
                return "completed_with_warnings"

            search_statuses = metadata.get("search_statuses")

            if isinstance(search_statuses, (list, tuple)):
                for entry in search_statuses:
                    if not isinstance(entry, Mapping):
                        continue

                    search_status = str(
                        entry.get("status", "")
                    ).strip().lower()

                    if search_status in {
                        "failed",
                        "error",
                        "unavailable",
                        "disabled",
                        "no_results",
                        "no results",
                    }:
                        return "completed_with_warnings"

                    if entry.get("error"):
                        return "completed_with_warnings"

            if metadata.get("fallback_used") is True:
                return "completed_with_warnings"

        if stage_name == "architect":
            if metadata.get("fallback_used") is True:
                return "completed_with_warnings"

            architect_status = str(
                metadata.get("architect_status", "")
            ).strip().lower()

            if architect_status in {"fallback", "fallback_used"}:
                return "completed_with_warnings"

        if metadata.get("fallback_used") is True:
            return "completed_with_warnings"

        return "completed"

    # --------------------------------------------------------
    # STAGE INPUTS
    # --------------------------------------------------------

    def _build_stage_inputs(
        self,
        stage_name: str,
    ) -> Dict[str, Any]:
        """Construct inputs for each supported stage."""

        if stage_name == "orchestrator":
            return {"context": self.context}

        if stage_name == "researcher":
            return {
                "topic": self._research_topic(),
                "context": self.context,
            }

        if stage_name == "architect":
            return {
                "research": self._completed_result("researcher"),
                "context": self.context,
            }

        if stage_name == "script_refiner":
            return {
                "story_plan": self._completed_result("architect"),
                "research": self._completed_result("researcher"),
                "context": self.context,
            }

        if stage_name == "story_director":
            return self._build_story_director_inputs()

        raise ValueError(
            f"Cannot construct inputs for stage {stage_name!r}."
        )

    def _build_story_director_inputs(self) -> Dict[str, Any]:
        """
        Prepare typed Story Director inputs from completed upstream stages.

        Registry resolution happens here so Story Director receives
        explicit registered objects rather than selecting identities itself.
        """

        from core.registry_loader import resolve_channel
        from core.models.story_plan import StoryPlan
        from core.models.script import Script
        from core.models.channel import Channel
        from core.models.world import World
        from core.models.character import Character
        from creative.catalogs.voice_catalog import get_voice_catalog

        research = WorkflowRunner._as_research_result(
            self._completed_result("researcher")
        )

        story_plan = WorkflowRunner._as_dataclass(
            self._completed_result("architect"),
            StoryPlan,
            label="StoryPlan",
        )

        script = WorkflowRunner._as_dataclass(
            self._completed_result("script_refiner"),
            Script,
            label="Script",
        )

        channel_id = self._selected_channel_id(self.context)
        resolved = resolve_channel(channel_id)

        if not isinstance(resolved, Mapping):
            raise TypeError(
                "resolve_channel() must return a mapping."
            )

        channel = WorkflowRunner._as_dataclass(
            resolved.get("channel"),
            Channel,
            label="Channel",
        )

        if channel.channel_id != channel_id:
            raise WorkflowRunnerError(
                "The resolved channel ID does not match the "
                "selected channel ID."
            )

        allowed_world_ids = list(channel.world_ids or [])

        if not allowed_world_ids:
            raise WorkflowRunnerError(
                f"Channel {channel_id!r} has no permitted worlds."
            )

        worlds = resolved.get("worlds", [])

        if not isinstance(worlds, (list, tuple)):
            raise TypeError(
                "The resolved channel's worlds must be a list or tuple."
            )

        world_by_id = {
            (
                world.get("world_id")
                if isinstance(world, Mapping)
                else getattr(world, "world_id", None)
            ): world
            for world in worlds
        }

        selected_world_id = self._selected_world_id(
            self.context,
            allowed_world_ids,
        )

        if selected_world_id not in allowed_world_ids:
            raise WorkflowRunnerError(
                f"World {selected_world_id!r} is not permitted by "
                f"channel {channel_id!r}."
            )

        world_value = world_by_id.get(selected_world_id)

        if world_value is None:
            raise WorkflowRunnerError(
                f"World {selected_world_id!r} was not returned by "
                "resolve_channel()."
            )

        world = WorkflowRunner._as_dataclass(
            world_value,
            World,
            label="World",
        )

        # Story Director may use all characters registered and made
        # available to this channel, not only the optional default speaker.
        raw_characters = resolved.get("characters", [])

        if not isinstance(raw_characters, (list, tuple)):
            raise TypeError(
                "The resolved channel's characters must be a list or tuple."
            )

        available_character_ids = set(
            channel.available_character_ids or []
        )

        characters = []

        for item in raw_characters:
            character_id = (
                item.get("character_id")
                if isinstance(item, Mapping)
                else getattr(item, "character_id", None)
            )

            if (
                character_id
                and (
                    not available_character_ids
                    or character_id in available_character_ids
                )
            ):
                characters.append(
                    WorkflowRunner._as_dataclass(
                        item,
                        Character,
                        label="Character",
                    )
                )

        # Revisions can be configured through handler_kwargs or supplied
        # by a later collaboration coordinator.
        handler_options = self.handler_kwargs.get(
            "story_director",
            {},
        )

        phase = handler_options.get("phase", "draft")
        revision_feedback = handler_options.get(
            "revision_feedback"
        )
        previous_script = handler_options.get("previous_script")
        num_predict = handler_options.get("num_predict")

        if phase == "revision" and revision_feedback is None:
            raise WorkflowRunnerError(
                "Story Director phase='revision' requires "
                "revision_feedback in handler_kwargs."
            )

        if previous_script is not None:
            previous_script = WorkflowRunner._as_dataclass(
                previous_script,
                Script,
                label="previous Script",
            )

        return {
            "story_plan": story_plan,
            "research": research,
            "script": script,
            "channel": channel,
            "world": world,
            "characters": characters,
            "context": self.context,
            "voice_catalog": get_voice_catalog(),
            "revision_feedback": revision_feedback,
            "previous_script": previous_script,
            "phase": phase,
            "num_predict": num_predict,
        }

    def _research_topic(self) -> str:
        """Choose the best available research topic from shared context."""

        brief = self.context.research_brief

        for key in (
            "research_topic",
            "topic",
            "query",
            "search_query",
        ):
            value = brief.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        interpretation = self.context.interpretation

        for key in (
            "interpreted_topic",
            "topic",
            "subject",
            "summary",
        ):
            value = interpretation.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        return self.context.original_prompt

    def _completed_result(self, stage_name: str) -> Any:
        """Retrieve a successfully completed stage's persisted result."""

        stage_result = self.context.stage_results.get(
            stage_name, {}
        )

        if stage_result.get("status") not in self.SUCCESSFUL_STAGE_STATUSES:
            raise WorkflowRunnerError(
                f"Stage {stage_name!r} must complete before its "
                "result can be passed downstream."
            )

        return stage_result.get("result")

    # --------------------------------------------------------
    # CHANNEL, WORLD AND CHARACTER RESOLUTION
    # --------------------------------------------------------

    @staticmethod
    def _selected_channel_id(context: RunContext) -> str:
        """Extract the selected channel ID from shared context."""

        selection = context.channel_selection

        if not isinstance(selection, Mapping):
            raise WorkflowRunnerError(
                "RunContext.channel_selection must be a mapping."
            )

        candidates = (
            selection.get("channel_id"),
            selection.get("selected_channel_id"),
            selection.get("id"),
        )

        nested = selection.get("selected_channel")

        if isinstance(nested, Mapping):
            candidates += (
                nested.get("channel_id"),
                nested.get("id"),
            )
        elif isinstance(nested, str):
            candidates += (nested,)

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        raise WorkflowRunnerError(
            "No selected channel ID was found in "
            "RunContext.channel_selection. Ensure the Orchestrator "
            "has populated the shared context before running the "
            "downstream workflow."
        )

    @staticmethod
    def _selected_world_id(
        context: RunContext,
        allowed_world_ids: list[str],
    ) -> str:
        """
        Resolve a world explicitly selected in context, if present.

        If none is recorded, use the first world permitted by the
        channel as a deterministic default.
        """

        selection = context.channel_selection

        candidates = [
            selection.get("world_id"),
            selection.get("selected_world_id"),
        ]

        nested = selection.get("world_selection")

        if isinstance(nested, Mapping):
            candidates.extend(
                (
                    nested.get("world_id"),
                    nested.get("selected_world_id"),
                )
            )

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        return allowed_world_ids[0]

    @staticmethod
    def _world_selection_rationale(
        context: RunContext,
        world_id: str,
    ) -> str:
        """Explain whether the world came from context or a safe fallback."""

        selection = context.channel_selection

        explicit_ids = (
            selection.get("world_id"),
            selection.get("selected_world_id"),
        )

        if world_id in explicit_ids:
            return "World explicitly selected in workflow context."

        nested = selection.get("world_selection")

        if isinstance(nested, Mapping):
            if world_id in (
                nested.get("world_id"),
                nested.get("selected_world_id"),
            ):
                return "World explicitly selected in workflow context."

        return (
            "Defaulted to the first world permitted by the selected "
            "channel; no explicit world selection was present."
        )

    @staticmethod
    def _resolve_speaker_character(
        *,
        context: RunContext,
        resolved: Mapping[str, Any],
        channel: Any,
        character_model: type,
    ) -> Any:
        """Resolve an optional speaker character from registered characters."""

        selection = context.channel_selection

        character_id = (
            selection.get("speaker_character_id")
            or selection.get("selected_character_id")
        )

        if not character_id:
            return None

        if character_id not in (channel.available_character_ids or []):
            raise WorkflowRunnerError(
                f"Character {character_id!r} is not available for "
                f"channel {channel.channel_id!r}."
            )

        characters = resolved.get("characters", [])

        if not isinstance(characters, (list, tuple)):
            raise TypeError(
                "The resolved channel's characters must be a list or tuple."
            )

        character = next(
            (
                item
                for item in characters
                if (
                    item.get("character_id")
                    if isinstance(item, Mapping)
                    else getattr(item, "character_id", None)
                ) == character_id
            ),
            None,
        )

        if character is None:
            raise WorkflowRunnerError(
                f"Character {character_id!r} was not returned by "
                "resolve_channel()."
            )

        return WorkflowRunner._as_dataclass(
            character,
            character_model,
            label="Character",
        )

    # --------------------------------------------------------
    # CONTEXT AND DATA MODELS
    # --------------------------------------------------------

    @staticmethod
    def _as_dataclass(
        value: Any,
        model: type,
        *,
        label: str,
    ) -> Any:
        """Reconstruct a dataclass model from a typed object or dictionary."""

        if isinstance(value, model):
            return value

        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)

        if not isinstance(value, Mapping):
            raise TypeError(
                f"{label} requires a {model.__name__} instance "
                "or a serialized mapping."
            )

        allowed_fields = {item.name for item in fields(model)}

        payload = {
            key: item
            for key, item in value.items()
            if key in allowed_fields
        }

        try:
            return model(**payload)
        except Exception as exc:
            raise WorkflowRunnerError(
                f"Could not reconstruct {label}: {exc}"
            ) from exc

    @staticmethod
    def _as_research_result(value: Any) -> ResearchResult:
        """Reconstruct the typed ResearchResult expected downstream."""

        return WorkflowRunner._as_dataclass(
            value,
            ResearchResult,
            label="ResearchResult",
        )

    def _store_stage_context(self, stage_name: str, result: Any) -> None:
        """Copy a stage's decisions (channel, length, world) into the shared context."""
        if stage_name == "orchestrator":
            from app.orchestrator import apply_to_context
            apply_to_context(self.context, result)
        elif stage_name == "architect":
            from agents.architect import apply_to_context
            from core.models.story_plan import StoryPlan
            apply_to_context(self.context, self._as_dataclass(result, StoryPlan, label="StoryPlan"))
        elif stage_name == "script_refiner":
            from agents.script_refiner import apply_to_context
            from core.models.script import Script
            apply_to_context(self.context, Script(**result))
        elif stage_name == "story_director":
            from agents.story_director import apply_to_context
            from core.models.script import Script
            apply_to_context(self.context, Script(**result))

        self.context.updated_at = utc_now()
        self.context.save(self.controller.context_path)    


    # --------------------------------------------------------
    # SERIALIZATION
    # --------------------------------------------------------

    @classmethod
    def _normalize_result(cls, value: Any) -> Any:
        """Convert agent output to JSON-safe data before persistence."""

        if value is None or isinstance(
            value, (str, int, float, bool)
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, Enum):
            return cls._normalize_result(value.value)

        if is_dataclass(value) and not isinstance(value, type):
            return cls._normalize_result(asdict(value))

        if isinstance(value, Mapping):
            return {
                str(key): cls._normalize_result(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                cls._normalize_result(item)
                for item in value
            ]

        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Stage result contains a value that cannot be "
                f"serialized: {type(value).__name__}."
            ) from exc

        return value

    # --------------------------------------------------------
    # CONFIGURATION VALIDATION
    # --------------------------------------------------------

    def _executable_stages(self) -> tuple[str, ...]:
        """Return executable stages supported by the controller."""

        stages = getattr(
            self.controller,
            "EXECUTABLE_STAGES",
            None,
        )

        if stages is not None:
            return tuple(stages)

        valid_stages = getattr(
            self.controller,
            "VALID_STAGES",
            (),
        )

        return tuple(
            stage
            for stage in valid_stages
            if stage != "completed"
        )

    def _validate_configuration(self) -> None:
        executable_stages = set(self._executable_stages())

        if not self.stage_order:
            raise ValueError("stage_order cannot be empty.")

        if len(set(self.stage_order)) != len(self.stage_order):
            raise ValueError(
                "stage_order cannot contain duplicate stages."
            )

        for stage_name in self.stage_order:
            if stage_name not in executable_stages:
                raise ValueError(
                    f"Invalid stage in stage_order: {stage_name!r}."
                )

            if stage_name not in self.stage_handlers:
                raise MissingStageHandlerError(
                    f"No handler is registered for stage "
                    f"{stage_name!r}."
                )

        for stage_name, handler in self.stage_handlers.items():
            if stage_name not in executable_stages:
                raise ValueError(
                    f"Unknown handler stage: {stage_name!r}."
                )

            if not callable(handler):
                raise TypeError(
                    f"Handler for {stage_name!r} must be callable."
                )

        for stage_name, kwargs in self.handler_kwargs.items():
            if stage_name not in executable_stages:
                raise ValueError(
                    f"Arguments supplied for unknown stage {stage_name!r}."
                )

            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"Arguments for {stage_name!r} must be a dictionary."
                )