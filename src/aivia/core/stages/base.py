"""The base contract every pipeline stage implements.

A stage is one idempotent, resumable step of a format recipe. It declares a
stable name, the artifacts it writes into the run directory, an estimate of
what it will cost and an asynchronous :meth:`Stage.run` that does the work.
Stages never choose providers or write cost events themselves: they receive a
:class:`StageContext` carrying the run directory, the resolved config, the cost
tracker, the router, the storage backend and a logger bound to the run.

The context is deliberately plain data plus already constructed collaborators.
That keeps a stage testable without a database, a network or a real provider:
a test supplies a temporary directory and the doubles it needs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from aivia.core.cost import CostTracker
from aivia.core.estimate import EstimateSource, StageEstimate
from aivia.core.models.run import Run, StageResult
from aivia.core.router import Router
from aivia.core.stage_retry import StageRetryPolicy
from aivia.providers.protocols.storage import StorageProvider

__all__ = [
    "Artifact",
    "ArtifactDeclaration",
    "ExternalInput",
    "ExternalInputDeclaration",
    "InputDeclaration",
    "Stage",
    "StageContext",
]

ArtifactDeclaration = str
"""The path a stage declares it writes, relative to the run directory."""

Artifact = ArtifactDeclaration
"""Alias for :data:`ArtifactDeclaration`."""

InputDeclaration = str
"""The path of an external file a stage reads as input."""

ExternalInputDeclaration = InputDeclaration
"""Alias for :data:`InputDeclaration`."""

ExternalInput = InputDeclaration
"""Alias for :data:`InputDeclaration`."""

_INPUT_ALIASES = (
    "input_files",
    "external_inputs",
    "external_input_files",
    "source_files",
)


def _normalise_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("stage name must be a string")
    name = value.strip()
    if not name:
        raise ValueError("stage name must be a non-empty string")
    return name


def _normalise_artifacts(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    artifacts: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError("artifact declarations must be strings")
        artifact = value.strip()
        if artifact and artifact not in artifacts:
            artifacts.append(artifact)
    return tuple(artifacts)


def _normalise_inputs(
    values: Iterable[str | Path] | str | Path,
) -> tuple[str, ...]:
    if isinstance(values, (str, Path)):
        values = (values,)
    inputs: list[str] = []
    for value in values:
        if isinstance(value, Path):
            declared = str(value)
        elif isinstance(value, str):
            declared = value.strip()
        else:
            raise TypeError("input declarations must be paths or strings")
        if declared and declared not in inputs:
            inputs.append(declared)
    return tuple(inputs)


def _class_inputs(stage: Stage) -> tuple[str, ...]:
    declared = getattr(type(stage), "inputs", ())
    if declared:
        return _normalise_inputs(declared)
    for alias in _INPUT_ALIASES:
        candidate = getattr(type(stage), alias, None)
        if isinstance(candidate, (str, Path, list, tuple)):
            normalised = _normalise_inputs(candidate)
            if normalised:
                return normalised
    return ()


@dataclass(slots=True)
class StageContext:
    """Everything a stage may use while it runs.

    ``run_dir`` is the directory the stage writes into; it is converted to a
    :class:`~pathlib.Path` so a caller may pass a string. ``config`` is the
    resolved configuration snapshot for the run and stays opaque here because
    stages only read the sections they need. ``cost``, ``router`` and
    ``storage`` are shared collaborators, never rebuilt per stage. ``logger``
    is already bound to the run and stage by the caller. ``run`` is optional so
    a unit test can build a context for a stage that does not inspect the run
    record.
    """

    run_dir: Path
    config: Any = field(default_factory=dict)
    cost: CostTracker | None = None
    router: Router | None = None
    storage: StorageProvider | None = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("aivia.stages")
    )
    run: Run | None = None
    _media: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).expanduser()

    @property
    def run_directory(self) -> Path:
        """Alias for :attr:`run_dir`."""
        return self.run_dir

    @property
    def cost_tracker(self) -> CostTracker | None:
        """Alias for :attr:`cost`."""
        return self.cost

    @property
    def media(self) -> Any:
        """The media facade shared by this stage, built on first use.

        The facade reuses the context router, cost tracker and storage and
        writes its metrics document into the run directory, so a stage never
        constructs providers or a separate ledger.
        """
        if self._media is None:
            from aivia.media.facade import MEDIA_METRICS_FILENAME, MediaServiceFacade

            if self.router is None:
                raise ValueError("a stage context needs a router to build media")
            config = self.config
            budget = (
                config.get("budget")
                if isinstance(config, Mapping)
                else getattr(config, "budget", None)
            )
            safety = (
                config.get("safety")
                if isinstance(config, Mapping)
                else getattr(config, "safety", None)
            )
            self._media = MediaServiceFacade(
                self.router,
                tracker=self.cost,
                storage=self.storage,
                budget=budget,
                safety=safety,
                metrics_path=self.run_dir / MEDIA_METRICS_FILENAME,
            )
        return self._media

    @property
    def media_facade(self) -> Any:
        """Alias for :attr:`media`."""
        return self.media


class Stage:
    """Base class for one recipe stage.

    A subclass normally sets :attr:`name`, :attr:`artifacts` and any
    :attr:`inputs` as class attributes and implements :meth:`run` (and, when it
    can be priced before a run, :meth:`estimate`). All may also be supplied to
    the constructor, which is useful for parameterised stages and for tests.
    ``inputs`` names external files the stage reads; their contents are hashed
    so editing a source forces a resume to run the stage again.

    The base :meth:`run` raises :class:`NotImplementedError`; the base
    :meth:`estimate` reports a free stage so a deterministic step does not have
    to override it. ``options`` holds the per-recipe options for this stage.
    """

    name: str = ""
    version: int | str = 1
    artifacts: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    max_attempts: int = 1
    retry_delay_seconds: float = 0.0
    backoff_multiplier: float = 1.0
    max_retry_delay_seconds: float | None = None
    retry_on_timeout: bool = True
    retry_codes: tuple[str, ...] = ()
    retry_policy: StageRetryPolicy

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        artifacts: Iterable[str] | None = None,
        inputs: Iterable[str | Path] | str | Path | None = None,
        input_files: Iterable[str | Path] | str | Path | None = None,
        input_paths: Iterable[str | Path] | str | Path | None = None,
        external_inputs: Iterable[str | Path] | str | Path | None = None,
        external_input_files: Iterable[str | Path] | str | Path | None = None,
        source_files: Iterable[str | Path] | str | Path | None = None,
        version: int | str | None = None,
        implementation_version: int | str | None = None,
        retry_policy: StageRetryPolicy | Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
        max_attempts: int | None = None,
        retry_delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        max_retry_delay_seconds: float | None = None,
        retry_on_timeout: bool | None = None,
        retry_codes: Iterable[str] | None = None,
    ) -> None:
        if name is not None:
            self.name = _normalise_name(name)
        elif self.name:
            self.name = _normalise_name(self.name)
        self.options: dict[str, Any] = dict(options or {})
        if implementation_version is not None:
            version = implementation_version
        if version is not None:
            self.version = version
        declared = self.artifacts if artifacts is None else artifacts
        self.artifacts = _normalise_artifacts(declared)
        declared_inputs: Iterable[str | Path] | str | Path | None = next(
            (
                value
                for value in (
                    inputs,
                    input_files,
                    input_paths,
                    external_inputs,
                    external_input_files,
                    source_files,
                )
                if value is not None
            ),
            None,
        )
        if declared_inputs is None:
            declared_inputs = _class_inputs(self)
        self.inputs = _normalise_inputs(declared_inputs)
        self.retry_policy = self._build_retry_policy(
            retry_policy,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            backoff_multiplier=backoff_multiplier,
            max_retry_delay_seconds=max_retry_delay_seconds,
            retry_on_timeout=retry_on_timeout,
            retry_codes=retry_codes,
        )

    def _build_retry_policy(
        self,
        declared: StageRetryPolicy | Mapping[str, Any] | None,
        *,
        timeout_seconds: float | None,
        max_attempts: int | None,
        retry_delay_seconds: float | None,
        backoff_multiplier: float | None,
        max_retry_delay_seconds: float | None,
        retry_on_timeout: bool | None,
        retry_codes: Iterable[str] | None,
    ) -> StageRetryPolicy:
        if declared is None:
            inherited = getattr(type(self), "retry_policy", None)
            if isinstance(inherited, (StageRetryPolicy, Mapping)):
                declared = inherited
        if declared is None:
            policy = StageRetryPolicy(
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.max_attempts,
                retry_delay_seconds=self.retry_delay_seconds,
                backoff_multiplier=self.backoff_multiplier,
                max_retry_delay_seconds=self.max_retry_delay_seconds,
                retry_on_timeout=self.retry_on_timeout,
                retry_codes=self.retry_codes,
            )
        elif isinstance(declared, StageRetryPolicy):
            policy = declared
        elif isinstance(declared, Mapping):
            policy = StageRetryPolicy(**dict(declared))
        else:
            raise TypeError("retry_policy must be a StageRetryPolicy or mapping")
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
            policy = replace(policy, timeout_seconds=timeout_seconds)
        if max_attempts is not None:
            self.max_attempts = max_attempts
            policy = replace(policy, max_attempts=max_attempts)
        if retry_delay_seconds is not None:
            self.retry_delay_seconds = retry_delay_seconds
            policy = replace(policy, retry_delay_seconds=retry_delay_seconds)
        if backoff_multiplier is not None:
            self.backoff_multiplier = backoff_multiplier
            policy = replace(policy, backoff_multiplier=backoff_multiplier)
        if max_retry_delay_seconds is not None:
            self.max_retry_delay_seconds = max_retry_delay_seconds
            policy = replace(
                policy,
                max_retry_delay_seconds=max_retry_delay_seconds,
            )
        if retry_on_timeout is not None:
            self.retry_on_timeout = retry_on_timeout
            policy = replace(policy, retry_on_timeout=retry_on_timeout)
        if retry_codes is not None:
            codes = tuple(retry_codes)
            self.retry_codes = codes
            policy = replace(policy, retry_codes=codes)
        return policy

    @property
    def stage_name(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def implementation_version(self) -> int | str:
        """The version of this stage implementation, used when resuming."""
        return self.version

    @implementation_version.setter
    def implementation_version(self, value: int | str) -> None:
        self.version = value

    @property
    def artifact_declarations(self) -> tuple[str, ...]:
        """Alias for :attr:`artifacts`."""
        return self.artifacts

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        """Alias for :attr:`artifacts`."""
        return self.artifacts

    @property
    def input_declarations(self) -> tuple[str, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @input_declarations.setter
    def input_declarations(self, value: Iterable[str | Path] | str | Path) -> None:
        self.inputs = _normalise_inputs(value)

    @property
    def input_paths(self) -> tuple[str, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @input_paths.setter
    def input_paths(self, value: Iterable[str | Path] | str | Path) -> None:
        self.inputs = _normalise_inputs(value)

    @property
    def input_files(self) -> tuple[str, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @input_files.setter
    def input_files(self, value: Iterable[str | Path] | str | Path) -> None:
        self.inputs = _normalise_inputs(value)

    @property
    def external_inputs(self) -> tuple[str, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @external_inputs.setter
    def external_inputs(self, value: Iterable[str | Path] | str | Path) -> None:
        self.inputs = _normalise_inputs(value)

    @property
    def external_input_files(self) -> tuple[str, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @external_input_files.setter
    def external_input_files(self, value: Iterable[str | Path] | str | Path) -> None:
        self.inputs = _normalise_inputs(value)

    @property
    def source_files(self) -> tuple[str, ...]:
        """Alias for :attr:`inputs`."""
        return self.inputs

    @source_files.setter
    def source_files(self, value: Iterable[str | Path] | str | Path) -> None:
        self.inputs = _normalise_inputs(value)

    def estimate(self, context: StageContext | None = None) -> StageEstimate:
        """Return the pre-run estimate for this stage.

        The default is a free, zero-amount estimate. A stage that needs a paid
        provider capability overrides this with the capability and quantity it
        will price through the router.
        """
        return StageEstimate(
            stage=self.name,
            capability="",
            provider="",
            units=1.0,
            unit_kind="",
            amount_usd=0.0,
            source=EstimateSource.FREE,
            free=True,
        )

    async def run(self, context: StageContext) -> StageResult:
        """Execute the stage and return its result.

        A concrete stage writes its declared artifacts under
        ``context.run_dir`` and returns a :class:`~aivia.core.models.run.StageResult`.
        The base implementation always raises so an unimplemented stage cannot
        silently succeed.
        """
        raise NotImplementedError(f"stage {self.name!r} does not implement run")
