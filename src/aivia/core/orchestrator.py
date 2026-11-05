"""Execute a recipe's stages in order and persist what each one produced.

The orchestrator is the deterministic control flow between the stage contract
and the durable records. It starts a run, walks the recipe in declaration
order, gives every stage a fresh :class:`~aivia.core.stages.StageContext`,
registers the artifacts the stage wrote and stores the resulting
:class:`~aivia.core.models.run.StageResult` before moving on. A stage that
raises, or returns a failed result, stops the recipe immediately: the failed
stage is persisted, the run is marked failed with the stage and its error code
and no later stage runs.

Stages remain independent of persistence. They receive plain collaborators and
write files under ``run_dir``; only this module touches the stage result
repository, the run state machine and the artifact registry.
"""

from __future__ import annotations

import logging
import re
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Engine

from aivia.core.config.formats import FormatRecipeConfig, StageConfig
from aivia.core.cost import CostTracker
from aivia.core.db.repos import RunRepository, StageRunRepository
from aivia.core.errors import AiviaError, ConfigError
from aivia.core.models.base import utcnow
from aivia.core.models.run import Run, RunState, StageResult, StageState
from aivia.core.resume import (
    compute_input_hash,
    compute_output_hash,
    should_skip_stage,
)
from aivia.core.router import Router
from aivia.core.run_dirs import RunDirectories, RunDirectory
from aivia.core.run_state import (
    IllegalRunTransition,
    RunNotFoundError,
    RunStateMachine,
    transition_run,
)
from aivia.core.snapshot import build_config_snapshot, write_config_snapshot
from aivia.core.stage_retry import (
    StageAttempt,
    StageRetryExhausted,
    StageRetryPolicy,
    execute_stage_with_retry,
)
from aivia.core.stages import Stage, StageContext
from aivia.providers.protocols.storage import StorageProvider

__all__ = [
    "ExecutionResult",
    "OrchestrationError",
    "OrchestrationResult",
    "Orchestrator",
    "OrchestratorError",
    "OrchestratorFailure",
    "OrchestratorResult",
    "PipelineOrchestrator",
    "RecipeExecutionResult",
    "RecipeOrchestrator",
    "RecipeRunner",
    "StageError",
    "StageExecutionError",
    "StageExecutionResult",
    "StageFailure",
    "StageOrchestrator",
    "StageRunner",
    "execute_recipe",
    "execute_stages",
    "orchestrate",
    "resume_recipe",
    "resume_run",
    "resume_stages",
    "run_recipe",
    "run_stages",
]

type StageLike = Stage | StageConfig | str | Mapping[str, Any]

_READY_STATES = frozenset({RunState.QUEUED, RunState.RUNNING, RunState.PAUSED_BUDGET})
_FAILURE_STATES = frozenset(
    {StageState.FAILED, StageState.CANCELLED, StageState.PAUSED_BUDGET}
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _error_code(error: BaseException) -> str:
    """Return a stable code for *error*.

    An :class:`~aivia.core.errors.AiviaError` supplies its own ``code``. Other
    exceptions fall back to the snake-cased class name so a stage failure is
    still machine-readable without leaking a vendor-specific message.
    """
    code = getattr(error, "code", "")
    if isinstance(code, str) and code.strip():
        return code.strip()
    explicit = getattr(error, "error_code", "")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return _CAMEL_BOUNDARY.sub("_", type(error).__name__).lower()


def _message(error: BaseException) -> str:
    text = str(error).strip()
    return text or type(error).__name__


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """The outcome of one orchestration pass.

    ``run`` is the latest persisted run record. ``stages`` holds every stage
    result in execution order, including the failed one when ``failed_stage``
    is set. Unknown attributes are forwarded to the run so callers that only
    need the run's state, id or error can use the result directly.
    """

    run: Run
    stages: tuple[StageResult, ...]
    failed_stage: str = ""
    error_code: str = ""
    error_message: str = ""

    @property
    def results(self) -> tuple[StageResult, ...]:
        """Alias for :attr:`stages`."""
        return self.stages

    @property
    def stage_results(self) -> tuple[StageResult, ...]:
        """Alias for :attr:`stages`."""
        return self.stages

    @property
    def skipped_stages(self) -> tuple[StageResult, ...]:
        """The stages a resume skipped because their inputs were unchanged."""
        return tuple(
            result for result in self.stages if result.status is StageState.SKIPPED
        )

    @property
    def executed_stages(self) -> tuple[StageResult, ...]:
        """The stages that actually ran during this pass."""
        return tuple(
            result for result in self.stages if result.status is not StageState.SKIPPED
        )

    @property
    def ok(self) -> bool:
        """Whether every stage completed without a failure."""
        return not self.failed_stage

    @property
    def succeeded(self) -> bool:
        """Alias for :attr:`ok`."""
        return self.ok

    @property
    def failed(self) -> bool:
        """Whether a stage stopped the recipe."""
        return not self.ok

    @property
    def failure(self) -> StageFailure | None:
        """The failure as an exception, or ``None`` when the run succeeded."""
        if self.ok:
            return None
        return StageFailure(
            self.failed_stage,
            self.error_code,
            self.error_message,
            result=self,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.run, name)


class StageFailure(AiviaError, RuntimeError):
    """A stage stopped the recipe.

    The exception carries the failing stage and its stable error code so a
    caller can report or retry precisely. ``result`` is the persisted
    orchestration result when one was produced, and ``cause`` the original
    exception when the stage raised.
    """

    code = "stage_failed"

    def __init__(
        self,
        stage: str,
        error_code: str,
        message: str,
        *,
        cause: BaseException | None = None,
        result: OrchestrationResult | None = None,
    ) -> None:
        super().__init__(
            message,
            details={"stage": stage, "error_code": error_code},
        )
        self.stage = stage
        self.stage_name = stage
        self.error_code = error_code
        self.error_message = message
        self.cause = cause
        self.result = result


OrchestrationError = StageFailure
"""Alias for :class:`StageFailure`."""

OrchestratorError = StageFailure
"""Alias for :class:`StageFailure`."""

OrchestratorFailure = StageFailure
"""Alias for :class:`StageFailure`."""

StageError = StageFailure
"""Alias for :class:`StageFailure`."""

StageExecutionError = StageFailure
"""Alias for :class:`StageFailure`."""

ExecutionResult = OrchestrationResult
"""Alias for :class:`OrchestrationResult`."""

OrchestratorResult = OrchestrationResult
"""Alias for :class:`OrchestrationResult`."""

StageExecutionResult = OrchestrationResult
"""Alias for :class:`OrchestrationResult`."""

RecipeExecutionResult = OrchestrationResult
"""Alias for :class:`OrchestrationResult`."""


def _coerce_stage(item: StageLike | type[Stage]) -> Stage:
    if isinstance(item, Stage):
        return item
    if isinstance(item, type) and issubclass(item, Stage):
        return item()
    if isinstance(item, StageConfig):
        return Stage(item.stage, **item.to_stage_kwargs())
    if isinstance(item, str):
        return Stage(item)
    if isinstance(item, Mapping):
        return _coerce_stage(StageConfig.model_validate(dict(item)))
    raise TypeError(
        "stages must be Stage instances, StageConfig entries or stage names"
    )


def _coerce_stages(
    stages: (
        Iterable[StageLike | type[Stage]]
        | Mapping[str, StageLike | type[Stage]]
        | FormatRecipeConfig
        | None
    ),
) -> list[Stage]:
    if stages is None:
        return []
    if isinstance(stages, FormatRecipeConfig):
        items: Iterable[StageLike | type[Stage]] = stages.stages
    elif isinstance(stages, Mapping):
        if {"stages", "steps", "pipeline"} & set(stages):
            items = FormatRecipeConfig.model_validate(dict(stages)).stages
        else:
            items = cast(Mapping[str, StageLike], stages).values()
    else:
        items = stages

    recipe: list[Stage] = []
    seen: set[str] = set()
    for item in items:
        stage = _coerce_stage(item)
        if stage.name in seen:
            raise ConfigError(f"duplicate stage in recipe: {stage.name}")
        seen.add(stage.name)
        recipe.append(stage)
    return recipe


class Orchestrator:
    """Run a recipe's stages in order and persist each result.

    *stages* may be stage instances, :class:`StageConfig` entries, bare stage
    names or a :class:`FormatRecipeConfig`. *run* is a :class:`Run` or a run id
    loaded from the repository. The stage result repository and the run state
    machine may be supplied directly; when only an engine is given the
    orchestrator builds both from it. Without a stage repository the recipe
    still runs but nothing is persisted, which keeps the class usable for a
    pure unit test.

    A failure is raised as :class:`StageFailure` by default. Pass
    ``raise_on_failure=False`` to receive the failed
    :class:`OrchestrationResult` instead; the run is marked failed either way.
    """

    def __init__(
        self,
        stages: (
            Iterable[StageLike | type[Stage]]
            | Mapping[str, StageLike | type[Stage]]
            | FormatRecipeConfig
            | None
        ) = None,
        run: Run | str | None = None,
        *,
        repository: StageRunRepository | Engine | None = None,
        engine: Engine | None = None,
        stage_repository: StageRunRepository | None = None,
        state_machine: RunStateMachine | None = None,
        run_repository: RunRepository | None = None,
        run_dir: str | Path | RunDirectory | None = None,
        data_dir: str | Path | None = None,
        config: Any = None,
        cost: CostTracker | None = None,
        router: Router | None = None,
        storage: StorageProvider | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        raise_on_failure: bool = True,
    ) -> None:
        if isinstance(stages, (Run, str)) and run is not None:
            stages, run = cast(Any, run), cast(Any, stages)

        self.stages = stages
        self._run = run
        self.run_dir = run_dir
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else None
        self.config = config
        self.cost = cost
        self.router = router
        self.storage = storage
        self.logger = logger or logging.getLogger("aivia.orchestrator")
        self.clock = clock
        self.sleep = sleep
        self.raise_on_failure = raise_on_failure
        self.result: OrchestrationResult | None = None
        self.engine = engine

        if isinstance(repository, Engine):
            self.engine = repository
            self.stage_repository: StageRunRepository | None = (
                stage_repository or StageRunRepository(repository)
            )
            if run_repository is None and state_machine is None:
                run_repository = RunRepository(repository)
        elif repository is not None:
            self.stage_repository = repository
            if self.engine is None:
                found_engine = getattr(repository, "engine", None)
                if isinstance(found_engine, Engine):
                    self.engine = found_engine
        else:
            self.stage_repository = stage_repository
            if self.stage_repository is None and self.engine is not None:
                self.stage_repository = StageRunRepository(self.engine)

        if state_machine is not None:
            self.state_machine: RunStateMachine | None = state_machine
            self.run_repository: RunRepository | None = state_machine.repository
        elif run_repository is not None:
            self.state_machine = RunStateMachine(run_repository)
            self.run_repository = run_repository
        else:
            stage_engine = getattr(self.stage_repository, "engine", None)
            if isinstance(stage_engine, Engine):
                self.run_repository = RunRepository(stage_engine)
                self.state_machine = RunStateMachine(self.run_repository)
            else:
                self.state_machine = None
                self.run_repository = None

        if self.engine is None:
            for candidate in (self.stage_repository, self.run_repository):
                found_engine = getattr(candidate, "engine", None)
                if isinstance(found_engine, Engine):
                    self.engine = found_engine
                    break

    async def execute(
        self,
        run: Run | str | None = None,
        stages: (
            Iterable[StageLike | type[Stage]]
            | Mapping[str, StageLike | type[Stage]]
            | FormatRecipeConfig
            | None
        ) = None,
    ) -> OrchestrationResult:
        """Execute every stage in order and return the persisted outcome."""
        if run is not None:
            self._run = run
        if stages is not None:
            self.stages = stages
        recipe = _coerce_stages(self.stages)
        run_record = self._resolve_run()
        directory = self._resolve_run_dir(run_record)
        run_record = self._start_run(run_record)
        run_record = self._save_snapshot(run_record, directory)
        self._run = run_record

        results: list[StageResult] = []
        total_cost = run_record.cost_usd
        previous_results = self._existing_results(run_record.id)
        upstream: list[StageResult] = []

        for index, stage in enumerate(recipe):
            started = self.clock()
            context = StageContext(
                run_dir=directory.path,
                config=self.config,
                cost=self.cost,
                router=self.router,
                storage=self.storage,
                logger=self.logger.getChild(stage.name),
                run=run_record,
                engine=self.engine,
            )
            expected_input = compute_input_hash(
                stage,
                config=self.config,
                upstream=upstream,
                base_dir=directory.path,
            )
            previous = self._previous_result(
                run_record.id,
                stage.name,
                previous_results,
            )
            superseded = self._superseded_artifacts(
                run_record.id,
                recipe,
                index,
                previous_results,
            )
            if should_skip_stage(
                previous,
                expected_input,
                directory=directory,
                superseded=superseded,
            ):
                skipped = self._skipped_result(cast(StageResult, previous), index)
                results.append(skipped)
                upstream.append(cast(StageResult, previous))
                self.logger.info(
                    "stage skipped",
                    extra={
                        "run_id": run_record.id,
                        "stage": stage.name,
                        "index": index,
                    },
                )
                continue

            self.logger.info(
                "stage started",
                extra={
                    "run_id": run_record.id,
                    "stage": stage.name,
                    "index": index,
                },
            )

            try:
                executed = await execute_stage_with_retry(
                    stage,
                    context,
                    policy=stage.retry_policy,
                    sleep=self.sleep,
                    clock=self.clock,
                )
                final = self._completed_result(
                    stage, executed.outcome, run_record, index, started
                )
                final = self._with_attempts(
                    final,
                    executed.attempts,
                    stage.retry_policy,
                    exhausted=executed.exhausted,
                )
                if final.status in _FAILURE_STATES:
                    final = final.model_copy(update={"input_hash": expected_input})
                    total_cost += executed.prior_cost_usd + final.cost_usd
                    run_record = self._apply_cost(run_record, total_cost)
                    return self._failed(
                        run_record,
                        stage,
                        final.error_code or "stage_failed",
                        final.error_message or f"stage {stage.name!r} failed",
                        results,
                        final,
                        directory,
                    )
                final = self._register(stage, final, directory)
                final = self._with_hashes(final, expected_input, directory)
                total_cost += executed.prior_cost_usd + final.cost_usd
            except StageRetryExhausted as exc:
                code = exc.error_code
                message = exc.error_message
                attempts = exc.attempts
                cause: BaseException | None = exc.last_error
                exhausted = True
                retry_cost = exc.prior_cost_usd
            except Exception as exc:
                code = _error_code(exc)
                message = _message(exc)
                attempts = tuple(getattr(exc, "attempts", ()))
                cause = exc
                exhausted = False
                retry_cost = float(getattr(exc, "prior_cost_usd", 0.0) or 0.0)
            else:
                stored = self._persist_result(final)
                results.append(stored)
                upstream.append(stored)
                self.logger.info(
                    "stage finished",
                    extra={
                        "run_id": run_record.id,
                        "stage": stage.name,
                        "index": index,
                    },
                )
                continue

            total_cost += retry_cost
            run_record = self._apply_cost(run_record, total_cost)
            failed = StageResult(
                run_id=run_record.id,
                stage=stage.name,
                status=StageState.FAILED,
                index=index,
                input_hash=expected_input,
                error_code=code,
                error_message=message,
                started_at=started,
                finished_at=self.clock(),
            )
            failed = self._with_attempts(
                failed,
                attempts,
                stage.retry_policy,
                exhausted=exhausted,
            )
            return self._failed(
                run_record,
                stage,
                code,
                message,
                results,
                failed,
                directory,
                cause=cause,
            )

        run_record = self._apply_cost(run_record, total_cost)
        run_record = self._finish_run(run_record, succeeded=True)
        result = OrchestrationResult(run=run_record, stages=tuple(results))
        self.result = result
        self._run = run_record
        return result

    async def run(
        self,
        run: Run | str | None = None,
        stages: (
            Iterable[StageLike | type[Stage]]
            | Mapping[str, StageLike | type[Stage]]
            | FormatRecipeConfig
            | None
        ) = None,
    ) -> OrchestrationResult:
        """Alias for :meth:`execute`."""
        return await self.execute(run, stages)

    async def run_recipe(
        self,
        run: Run | str | None = None,
        stages: (
            Iterable[StageLike | type[Stage]]
            | Mapping[str, StageLike | type[Stage]]
            | FormatRecipeConfig
            | None
        ) = None,
    ) -> OrchestrationResult:
        """Alias for :meth:`execute`."""
        return await self.execute(run, stages)

    async def resume(
        self,
        run: Run | str | None = None,
        stages: (
            Iterable[StageLike | type[Stage]]
            | Mapping[str, StageLike | type[Stage]]
            | FormatRecipeConfig
            | None
        ) = None,
    ) -> OrchestrationResult:
        """Resume a run, reusing succeeded stages whose inputs are unchanged."""
        return await self.execute(run, stages)

    def execute_sync(self) -> OrchestrationResult:
        """Run :meth:`execute` from synchronous code."""
        import asyncio

        return asyncio.run(self.execute())

    def run_sync(self) -> OrchestrationResult:
        """Alias for :meth:`execute_sync`."""
        return self.execute_sync()

    def _resolve_run(self) -> Run:
        if self._run is None:
            return Run()
        if isinstance(self._run, str):
            if self.run_repository is None:
                raise ConfigError(
                    "a run id requires a run repository",
                    details={"run_id": self._run},
                )
            found = self.run_repository.get(self._run)
            if found is None:
                raise RunNotFoundError(
                    f"run not found: {self._run}",
                    details={"run_id": self._run},
                )
            return found
        run = self._run
        if self.run_repository is not None:
            stored = self.run_repository.get(run.id)
            if stored is None:
                stored = self.run_repository.create(run)
            return stored
        return run

    def _resolve_run_dir(self, run: Run) -> RunDirectory:
        if isinstance(self.run_dir, RunDirectory):
            directory = self.run_dir
            directory.create()
            return directory
        if self.run_dir is not None:
            return RunDirectory(self.run_dir, run_id=run.id).create()
        if run.run_dir:
            return RunDirectory(run.run_dir, run_id=run.id).create()
        if self.data_dir is not None:
            return RunDirectories(self.data_dir).create(run)
        directory = RunDirectory(
            Path(tempfile.gettempdir()) / "aivia-runs" / run.id,
            run_id=run.id,
        )
        return directory.create()

    def _save_snapshot(self, run: Run, directory: RunDirectory) -> Run:
        if run.config_snapshot and directory.snapshot_path.is_file():
            return run
        snapshot = build_config_snapshot(self.config, run=run)
        write_config_snapshot(directory.path, snapshot)
        if dict(snapshot) == run.config_snapshot:
            return run
        run = run.model_copy(update={"config_snapshot": dict(snapshot)})
        if self.run_repository is not None:
            stored = self.run_repository.update(
                run.id,
                config_snapshot=dict(snapshot),
            )
            if stored is not None:
                run = stored
        return run

    def _existing_results(self, run_id: str) -> dict[str, StageResult]:
        repository = self.stage_repository
        lister = getattr(repository, "list", None)
        if not callable(lister):
            return {}
        try:
            records = lister(run_id=run_id)
        except (TypeError, ValueError):
            return {}
        return {record.stage: record for record in records}

    def _previous_result(
        self,
        run_id: str,
        stage: str,
        cache: dict[str, StageResult],
    ) -> StageResult | None:
        if stage in cache:
            return cache[stage]
        repository = self.stage_repository
        getter = getattr(repository, "get", None)
        if not callable(getter):
            return None
        try:
            found = cast(StageResult | None, getter(run_id, stage))
        except (TypeError, ValueError):
            return None
        if found is not None:
            cache[stage] = found
        return found

    def _superseded_artifacts(
        self,
        run_id: str,
        recipe: list[Stage],
        index: int,
        cache: dict[str, StageResult],
    ) -> set[str]:
        """Paths a later stage in this recipe successfully registered again."""
        paths: set[str] = set()
        for later in recipe[index + 1 :]:
            result = self._previous_result(run_id, later.name, cache)
            if result is not None and result.status is StageState.SUCCEEDED:
                paths.update(str(path) for path in result.artifacts)
        return paths

    def _skipped_result(self, previous: StageResult, index: int) -> StageResult:
        metadata = dict(previous.metadata)
        metadata.setdefault("resumed", True)
        metadata.setdefault("previous_status", previous.status.value)
        return previous.model_copy(
            update={
                "status": StageState.SKIPPED,
                "index": index,
                "metadata": metadata,
            }
        )

    def _with_hashes(
        self,
        result: StageResult,
        input_hash: str,
        directory: RunDirectory,
    ) -> StageResult:
        output_hash = result.output_hash
        if not output_hash:
            output_hash = compute_output_hash(directory.artifacts(result.stage))
        return result.model_copy(
            update={"input_hash": input_hash, "output_hash": output_hash}
        )

    def _start_run(self, run: Run) -> Run:
        if run.state in {RunState.FAILED, RunState.SUCCEEDED}:
            return self._restart_run(run)
        if run.state not in _READY_STATES:
            raise IllegalRunTransition(run.state, RunState.RUNNING)
        if run.state is RunState.RUNNING:
            return run
        if run.state is RunState.PAUSED_BUDGET and self.state_machine is not None:
            return self.state_machine.resume(
                run,
                resumed_by="orchestrator",
                reason="resume",
                now=self.clock(),
            )
        if self.state_machine is not None:
            return self.state_machine.start(run, now=self.clock())
        return transition_run(run, RunState.RUNNING, now=self.clock())

    def _restart_run(self, run: Run) -> Run:
        if self.state_machine is not None:
            return self.state_machine.restart(
                run,
                resumed_by="orchestrator",
                reason="resume after failure",
                now=self.clock(),
            )
        moment = self.clock()
        return run.model_copy(
            update={
                "state": RunState.RUNNING,
                "finished_at": None,
                "error_code": "",
                "error_message": "",
                "resumed_at": moment,
                "resumed_by": "orchestrator",
                "resume_reason": "resume after failure",
                "updated_at": moment,
            }
        )

    def _finish_run(self, run: Run, *, succeeded: bool) -> Run:
        if succeeded:
            if self.state_machine is not None:
                return self.state_machine.succeed(run, now=self.clock())
            return transition_run(run, RunState.SUCCEEDED, now=self.clock())
        raise AssertionError("failure goes through _failed")

    def _failed(
        self,
        run: Run,
        stage: Stage,
        code: str,
        message: str,
        results: list[StageResult],
        failed: StageResult,
        directory: RunDirectory,
        *,
        cause: BaseException | None = None,
    ) -> OrchestrationResult:
        failed = self._register(stage, failed, directory, require_exists=False)
        results.append(self._persist_result(failed))
        detail = f"stage {stage.name!r} failed: {message}"
        if self.state_machine is not None:
            run = self.state_machine.fail(
                run,
                error_code=code,
                error_message=detail,
                now=self.clock(),
            )
        else:
            run = transition_run(
                run,
                RunState.FAILED,
                error_code=code,
                error_message=detail,
                now=self.clock(),
            )
        result = OrchestrationResult(
            run=run,
            stages=tuple(results),
            failed_stage=stage.name,
            error_code=code,
            error_message=message,
        )
        self.result = result
        self._run = run
        self.logger.error(
            "stage failed",
            extra={"run_id": run.id, "stage": stage.name, "error_code": code},
        )
        if self.raise_on_failure:
            raise StageFailure(
                stage.name,
                code,
                detail,
                cause=cause,
                result=result,
            ) from cause
        return result

    def _with_attempts(
        self,
        result: StageResult,
        attempts: tuple[StageAttempt, ...],
        policy: StageRetryPolicy,
        *,
        exhausted: bool = False,
    ) -> StageResult:
        metadata = dict(result.metadata)
        metadata["attempts"] = len(attempts)
        metadata["attempt_count"] = len(attempts)
        metadata["max_attempts"] = policy.max_attempts
        metadata["exhausted"] = exhausted
        metadata["retry_policy"] = policy.to_dict()
        metadata["attempt_history"] = [attempt.to_dict() for attempt in attempts]
        if len(attempts) > 1:
            metadata["retries"] = len(attempts) - 1
        return result.model_copy(update={"metadata": metadata})

    def _completed_result(
        self,
        stage: Stage,
        outcome: StageResult | Mapping[str, Any] | None,
        run: Run,
        index: int,
        started: datetime,
    ) -> StageResult:
        if outcome is None:
            result = StageResult(stage=stage.name)
        elif isinstance(outcome, StageResult):
            result = outcome
        elif isinstance(outcome, Mapping):
            result = StageResult.model_validate(dict(outcome))
        else:
            raise TypeError("stage run must return a StageResult, a mapping or None")
        status = result.status
        if status in {StageState.PENDING, StageState.RUNNING}:
            status = StageState.SUCCEEDED
        return result.model_copy(
            update={
                "run_id": run.id,
                "stage": stage.name,
                "index": index if result.index is None else result.index,
                "status": status,
                "started_at": result.started_at or started,
                "finished_at": result.finished_at or self.clock(),
            }
        )

    def _register(
        self,
        stage: Stage,
        result: StageResult,
        directory: RunDirectory,
        *,
        require_exists: bool = True,
    ) -> StageResult:
        artifacts = result.artifacts or list(stage.artifacts)
        directory.register(
            stage.name,
            artifacts,
            result=result,
            require_exists=require_exists,
        )
        return result

    def _apply_cost(self, run: Run, total: float) -> Run:
        if total == run.cost_usd:
            return run
        if self.run_repository is not None:
            updated = self.run_repository.update(run.id, cost_usd=total)
            if updated is not None:
                return updated
        return run.model_copy(update={"cost_usd": total})

    def _persist_result(self, result: StageResult) -> StageResult:
        repository = self.stage_repository
        if repository is None:
            return result
        getter = getattr(repository, "get", None)
        existing = getter(result.run_id, result.stage) if callable(getter) else None
        stored: StageResult | None
        if existing is None:
            stored = repository.create(result)
        else:
            changes = result.model_dump(
                mode="python",
                exclude={"created_at", "updated_at"},
            )
            stored = repository.update(result.run_id, result.stage, **changes)
        return stored if stored is not None else result


RecipeOrchestrator = Orchestrator
"""Alias for :class:`Orchestrator`."""

StageOrchestrator = Orchestrator
"""Alias for :class:`Orchestrator`."""

PipelineOrchestrator = Orchestrator
"""Alias for :class:`Orchestrator`."""

RecipeRunner = Orchestrator
"""Alias for :class:`Orchestrator`."""

StageRunner = Orchestrator
"""Alias for :class:`Orchestrator`."""


async def execute_recipe(
    run: Run | str | None,
    stages: (
        Iterable[StageLike | type[Stage]]
        | Mapping[str, StageLike | type[Stage]]
        | FormatRecipeConfig
    ),
    **options: Any,
) -> OrchestrationResult:
    """Execute *stages* for *run* with the given orchestrator options."""
    return await Orchestrator(stages=stages, run=run, **options).execute()


async def run_recipe(
    run: Run | str | None,
    stages: (
        Iterable[StageLike | type[Stage]]
        | Mapping[str, StageLike | type[Stage]]
        | FormatRecipeConfig
    ),
    **options: Any,
) -> OrchestrationResult:
    """Alias for :func:`execute_recipe`."""
    return await execute_recipe(run, stages, **options)


async def resume_recipe(
    run: Run | str | None,
    stages: (
        Iterable[StageLike | type[Stage]]
        | Mapping[str, StageLike | type[Stage]]
        | FormatRecipeConfig
    ),
    **options: Any,
) -> OrchestrationResult:
    """Resume *run*, reusing succeeded stages whose inputs are unchanged."""
    return await execute_recipe(run, stages, **options)


resume_run = resume_recipe
"""Alias for :func:`resume_recipe`."""

resume_stages = resume_recipe
"""Alias for :func:`resume_recipe`."""

execute_stages = execute_recipe
"""Alias for :func:`execute_recipe`."""

run_stages = execute_recipe
"""Alias for :func:`execute_recipe`."""

orchestrate = execute_recipe
"""Alias for :func:`execute_recipe`."""
