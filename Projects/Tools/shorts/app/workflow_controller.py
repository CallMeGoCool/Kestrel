from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from core.run_context import RunContext, utc_now


class WorkflowController:
    """
    Manage the persistent state of a single Kestrel Shorts workflow.

    Responsibilities:
    - Create and load workflow runs.
    - Validate stage state transitions.
    - Record stage results and failures.
    - Support controlled recovery and resumption.
    - Register artifact references.
    - Record workflow feedback.
    - Persist shared context after state changes.

    This class does not execute agents or determine creative decisions.
    Those responsibilities belong to the workflow runner and agents.
    """

    VALID_STAGES = (
        "orchestrator",
        "researcher",
        "architect",
        "script_refiner",
        "story_director",
        "visual_director",
        "reviewer",
        "completed",
    )

    EXECUTABLE_STAGES = tuple(
        stage for stage in VALID_STAGES if stage != "completed"
    )

    VALID_STATUSES = (
        "created",
        "running",
        "completed",
        "completed_with_warnings",
        "failed",
        "awaiting_clarification",
    )

    VALID_STAGE_STATUSES = (
        "pending",
        "running",
        "completed",
        "completed_with_warnings",
        "failed",
        "interrupted",
        "skipped",
    )

    VALID_FEEDBACK_SEVERITIES = (
        "info",
        "warning",
        "error",
        "critical",
    )

    SUCCESSFUL_STAGE_STATUSES = (
        "completed",
        "completed_with_warnings",
    )

    def __init__(
        self,
        context: RunContext,
        context_path: Optional[Path | str] = None,
    ) -> None:
        if not isinstance(context, RunContext):
            raise TypeError("context must be a RunContext instance.")

        self.context = context

        self.context_path = (
            Path(context_path).resolve()
            if context_path is not None
            else context.context_path
        )

        self._validate_context_state()

    # --------------------------------------------------------
    # RUN CREATION AND LOADING
    # --------------------------------------------------------

    @classmethod
    def create_run(
        cls,
        original_prompt: str,
        explicit_preferences: Optional[Dict[str, Any]] = None,
        constraints: Optional[list[str]] = None,
    ) -> "WorkflowController":
        """Create and persist a new workflow run."""

        if explicit_preferences is not None:
            if not isinstance(explicit_preferences, dict):
                raise TypeError(
                    "explicit_preferences must be a dictionary."
                )

        if constraints is not None:
            if not isinstance(constraints, list):
                raise TypeError("constraints must be a list.")

            if not all(
                isinstance(item, str) for item in constraints
            ):
                raise TypeError(
                    "Every constraint must be a string."
                )

        context = RunContext(
            original_prompt=original_prompt,
            explicit_preferences=(
                dict(explicit_preferences)
                if explicit_preferences is not None
                else {}
            ),
            constraints=(
                list(constraints)
                if constraints is not None
                else []
            ),
        )

        controller = cls(context)
        controller._persist()

        return controller

    @classmethod
    def resume_run(
        cls,
        context_path: Path | str,
    ) -> "WorkflowController":
        """
        Load an existing run without automatically executing agents.

        The workflow runner decides which stage should execute next.
        """

        context = RunContext.load(context_path)

        return cls(
            context,
            context_path=context_path,
        )

    # --------------------------------------------------------
    # WORKFLOW LIFECYCLE
    # --------------------------------------------------------

    def start(self) -> None:
        """
        Start a new run or resume an eligible run.

        Failed runs must use resume_after_failure().
        Runs awaiting clarification must use resolve_clarification().
        """

        status = self.context.workflow.get("status", "created")

        if status not in ("created", "running"):
            raise RuntimeError(
                f"Cannot start a workflow with status {status!r}."
            )

        if status == "running":
            active_stage = self.context.workflow.get("current_stage")

            if active_stage in self.EXECUTABLE_STAGES:
                stage_result = self.context.stage_results.get(
                    active_stage, {}
                )

                if stage_result.get("status") == "running":
                    raise RuntimeError(
                        f"Stage {active_stage!r} is already marked "
                        "as running. Recover the interrupted stage "
                        "before starting execution again."
                    )

        self.context.workflow["status"] = "running"
        self.context.workflow.setdefault("started_at", utc_now())

        self._persist()

    def begin_stage(self, stage_name: str) -> None:
        """Mark a valid, eligible stage as running."""

        self._validate_executable_stage(stage_name)

        workflow = self.context.workflow
        workflow_status = workflow.get("status", "created")

        if workflow_status not in ("created", "running"):
            raise RuntimeError(
                "Cannot begin a stage while workflow status is "
                f"{workflow_status!r}."
            )

        current_stage = workflow.get("current_stage")
        current_result = self.context.stage_results.get(
            current_stage, {}
        )

        if (
            current_stage in self.EXECUTABLE_STAGES
            and current_result.get("status") == "running"
        ):
            raise RuntimeError(
                f"Cannot begin {stage_name!r}; stage "
                f"{current_stage!r} is still running."
            )

        if current_stage == "completed":
            raise RuntimeError(
                "Cannot begin another stage after workflow completion."
            )

        stage_result = self.context.stage_results.get(
            stage_name, {}
        )

        if stage_result.get("status") in self.SUCCESSFUL_STAGE_STATUSES:
            raise RuntimeError(
                f"Stage {stage_name!r} has already completed "
                "successfully. Use its saved result instead."
            )

        now = utc_now()

        workflow["current_stage"] = stage_name
        workflow["status"] = "running"

        stage_result = dict(stage_result)

        stage_result.update(
            {
                "status": "running",
                "started_at": now,
                "finished_at": None,
                "error": None,
                "attempt_count": (
                    int(stage_result.get("attempt_count", 0)) + 1
                ),
            }
        )

        self.context.stage_results[stage_name] = stage_result

        # Clear a previous failure when the stage is retried.
        failed_stages = workflow.setdefault("failed_stages", [])

        workflow["failed_stages"] = [
            item for item in failed_stages if item != stage_name
        ]

        workflow.pop("failure_reason", None)

        self._persist()

    def complete_stage(
        self,
        stage_name: str,
        result: Any,
        artifact_path: Optional[Path | str] = None,
        *,
        status: str = "completed",
    ) -> None:
        """Record a successful result, optionally with warnings."""

        self._validate_executable_stage(stage_name)

        if status not in self.SUCCESSFUL_STAGE_STATUSES:
            raise ValueError(
                "status must be 'completed' or "
                "'completed_with_warnings'."
            )

        workflow = self.context.workflow

        if workflow.get("status") != "running":
            raise RuntimeError(
                "Cannot complete a stage unless the workflow is running."
            )

        if workflow.get("current_stage") != stage_name:
            raise RuntimeError(
                f"Cannot complete {stage_name!r}; current stage is "
                f"{workflow.get('current_stage')!r}."
            )

        stage_result = self.context.stage_results.get(
            stage_name, {}
        )

        if stage_result.get("status") != "running":
            raise RuntimeError(
                f"Cannot complete {stage_name!r}; it is not marked "
                "as running."
            )

        now = utc_now()

        stage_result.update(
            {
                "status": status,
                "finished_at": now,
                "error": None,
                "result": result,
            }
        )

        self.context.stage_results[stage_name] = stage_result

        completed_stages = workflow.setdefault(
            "completed_stages", []
        )

        if stage_name not in completed_stages:
            completed_stages.append(stage_name)

        failed_stages = workflow.setdefault("failed_stages", [])

        workflow["failed_stages"] = [
            item for item in failed_stages if item != stage_name
        ]

        workflow.pop("failure_reason", None)

        if artifact_path is not None:
            self._register_artifact(
                name=stage_name,
                path=artifact_path,
                stage=stage_name,
                metadata=None,
            )

        self._persist()

    def fail_stage(
        self,
        stage_name: str,
        error: str,
    ) -> None:
        """Record a stage failure and mark the run as failed."""

        self._validate_executable_stage(stage_name)

        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a non-empty string.")

        workflow = self.context.workflow

        if workflow.get("current_stage") != stage_name:
            raise RuntimeError(
                f"Cannot fail {stage_name!r}; current stage is "
                f"{workflow.get('current_stage')!r}."
            )

        stage_result = self.context.stage_results.get(
            stage_name, {}
        )

        if stage_result.get("status") != "running":
            raise RuntimeError(
                f"Cannot fail {stage_name!r}; it is not marked "
                "as running."
            )

        stage_result.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error": error.strip(),
            }
        )

        self.context.stage_results[stage_name] = stage_result

        failed_stages = workflow.setdefault("failed_stages", [])

        if stage_name not in failed_stages:
            failed_stages.append(stage_name)

        workflow["status"] = "failed"
        workflow["failure_reason"] = error.strip()

        self._persist()

    def resume_after_failure(self) -> str:
        """
        Prepare a failed workflow for another execution attempt.

        Returns the stage that should be retried.

        Completed stage results are preserved. The workflow runner
        remains responsible for validating prerequisites and invoking
        the appropriate agent.
        """

        workflow = self.context.workflow

        if workflow.get("status") != "failed":
            raise RuntimeError(
                "Only a failed workflow can be resumed this way."
            )

        stage_name = workflow.get("current_stage")

        self._validate_executable_stage(stage_name)

        stage_result = self.context.stage_results.get(
            stage_name, {}
        )

        if stage_result.get("status") != "failed":
            raise RuntimeError(
                "The current stage is not recorded as failed. "
                "Recover an interrupted stage explicitly if needed."
            )

        workflow["status"] = "running"
        workflow.pop("failure_reason", None)

        # begin_stage() increments the attempt count on retry.
        self._persist()

        return stage_name

    def recover_interrupted_stage(self) -> str:
        """
        Recover a stage left running after an unexpected interruption.

        This method does not rerun the agent. It marks the stage as
        interrupted so the runner can retry it deliberately.
        """

        workflow = self.context.workflow

        if workflow.get("status") != "running":
            raise RuntimeError(
                "Only a running workflow can recover an interrupted stage."
            )

        stage_name = workflow.get("current_stage")

        self._validate_executable_stage(stage_name)

        stage_result = self.context.stage_results.get(
            stage_name, {}
        )

        if stage_result.get("status") != "running":
            raise RuntimeError(
                f"Stage {stage_name!r} is not marked as running."
            )

        stage_result.update(
            {
                "status": "interrupted",
                "finished_at": utc_now(),
                "error": "Execution was interrupted before completion.",
            }
        )

        self.context.stage_results[stage_name] = stage_result

        # Preserve earlier successful results.
        workflow["status"] = "failed"
        workflow["failure_reason"] = (
            f"Stage {stage_name!r} was interrupted."
        )

        failed_stages = workflow.setdefault("failed_stages", [])

        if stage_name not in failed_stages:
            failed_stages.append(stage_name)

        self._persist()

        return stage_name

    # --------------------------------------------------------
    # CLARIFICATION
    # --------------------------------------------------------

    def request_clarification(self, reason: str) -> None:
        """Pause the workflow pending additional user input."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")

        status = self.context.workflow.get("status")

        if status not in ("created", "running"):
            raise RuntimeError(
                "Clarification can only be requested from a "
                "created or running workflow."
            )

        self.context.workflow["status"] = "awaiting_clarification"
        self.context.workflow["clarification_reason"] = reason.strip()
        self._persist()

    def resolve_clarification(
        self,
        response: str,
    ) -> None:
        """
        Record the user's clarification and allow the runner to continue.

        The response is stored in explicit_preferences so the existing
        RunContext schema remains unchanged.
        """

        if not isinstance(response, str) or not response.strip():
            raise ValueError("response must be a non-empty string.")

        workflow = self.context.workflow

        if workflow.get("status") != "awaiting_clarification":
            raise RuntimeError(
                "The workflow is not awaiting clarification."
            )

        preferences = self.context.explicit_preferences

        clarifications = preferences.setdefault(
            "clarification_responses", []
        )

        if not isinstance(clarifications, list):
            raise ValueError(
                "explicit_preferences['clarification_responses'] "
                "must be a list."
            )

        clarifications.append(
            {
                "response": response.strip(),
                "timestamp": utc_now(),
                "stage": workflow.get("current_stage"),
            }
        )

        workflow["status"] = "running"
        workflow.pop("clarification_reason", None)

        self._persist()

    # --------------------------------------------------------
    # ARTIFACTS
    # --------------------------------------------------------

    def register_artifact(
        self,
        name: str,
        path: Path | str,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an artifact and persist the updated context."""

        self._register_artifact(
            name=name,
            path=path,
            stage=stage,
            metadata=metadata,
        )

        self._persist()

    def _register_artifact(
        self,
        name: str,
        path: Path | str,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an artifact without triggering a nested save."""

        self._validate_stage(stage)

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Artifact name must be non-empty.")

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        artifact_path = Path(path).resolve()

        self.context.artifacts[name.strip()] = {
            "path": str(artifact_path),
            "stage": stage,
            "registered_at": utc_now(),
            "exists_at_registration": artifact_path.exists(),
            "metadata": dict(metadata) if metadata is not None else {},
        }

    # --------------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------------

    def add_feedback(
        self,
        source: str,
        message: str,
        stage: Optional[str] = None,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append feedback without overwriting previous entries."""

        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string.")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")

        if stage is not None:
            self._validate_stage(stage)

        if severity not in self.VALID_FEEDBACK_SEVERITIES:
            raise ValueError(
                f"Invalid feedback severity {severity!r}. "
                f"Expected one of: "
                f"{', '.join(self.VALID_FEEDBACK_SEVERITIES)}."
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        entry = {
            "feedback_id": datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%f"
            ),
            "timestamp": utc_now(),
            "source": source.strip(),
            "message": message.strip(),
            "stage": stage,
            "severity": severity,
            "metadata": dict(metadata) if metadata is not None else {},
        }

        self.context.feedback_history.append(entry)
        self._persist()

    # --------------------------------------------------------
    # WORKFLOW COMPLETION
    # --------------------------------------------------------

    def complete_workflow(self) -> None:
        """Complete the workflow, preserving any stage warnings."""

        workflow = self.context.workflow

        if workflow.get("status") != "running":
            raise RuntimeError(
                "Only a running workflow can be completed."
            )

        current_stage = workflow.get("current_stage")

        if current_stage in self.EXECUTABLE_STAGES:
            current_result = self.context.stage_results.get(
                current_stage, {}
            )

            if current_result.get("status") not in (
                "completed",
                "completed_with_warnings",
            ):
                raise RuntimeError(
                    "Cannot complete the workflow while the current "
                    f"stage {current_stage!r} is not completed."
                )

        now = utc_now()

        has_warnings = any(
            self.context.stage_results.get(stage_name, {}).get(
                "status"
            ) == "completed_with_warnings"
            for stage_name in self.EXECUTABLE_STAGES
        )

        final_status = (
            "completed_with_warnings"
            if has_warnings
            else "completed"
        )

        workflow["current_stage"] = "completed"
        workflow["status"] = final_status
        workflow["finished_at"] = now

        self.context.finalization.update(
            {
                "status": final_status,
                "completed_at": now,
            }
        )

        self._persist()

    # --------------------------------------------------------
    # INTERNAL VALIDATION AND PERSISTENCE
    # --------------------------------------------------------

    def _validate_stage(self, stage_name: str) -> None:
        if not isinstance(stage_name, str):
            raise TypeError("stage_name must be a string.")

        if stage_name not in self.VALID_STAGES:
            raise ValueError(
                f"Unknown workflow stage: {stage_name!r}. "
                f"Expected one of: {', '.join(self.VALID_STAGES)}."
            )

    def _validate_executable_stage(self, stage_name: str) -> None:
        self._validate_stage(stage_name)

        if stage_name == "completed":
            raise ValueError(
                "'completed' is a terminal workflow state, "
                "not an executable stage."
            )

    def _validate_context_state(self) -> None:
        """Validate persisted workflow state without modifying it."""

        workflow = self.context.workflow

        if not isinstance(workflow, dict):
            raise TypeError("context.workflow must be a dictionary.")

        status = workflow.get("status", "created")

        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Unsupported workflow status: {status!r}."
            )

        current_stage = workflow.get("current_stage", "orchestrator")

        self._validate_stage(current_stage)

        if not isinstance(self.context.stage_results, dict):
            raise TypeError("context.stage_results must be a dictionary.")

        if not isinstance(self.context.artifacts, dict):
            raise TypeError("context.artifacts must be a dictionary.")

        if not isinstance(self.context.feedback_history, list):
            raise TypeError(
                "context.feedback_history must be a list."
            )

    def _persist(self) -> None:
        """Persist the current context through RunContext's atomic save."""

        self.context.updated_at = utc_now()
        self.context.save(self.context_path)