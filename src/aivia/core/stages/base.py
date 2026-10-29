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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aivia.core.cost import CostTracker
from aivia.core.estimate import EstimateSource, StageEstimate
from aivia.core.models.run import Run, StageResult
from aivia.core.router import Router
from aivia.providers.protocols.storage import StorageProvider

__all__ = [
    "Artifact",
    "ArtifactDeclaration",
    "Stage",
    "StageContext",
]

ArtifactDeclaration = str
"""The path a stage declares it writes, relative to the run directory."""

Artifact = ArtifactDeclaration
"""Alias for :data:`ArtifactDeclaration`."""


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


class Stage:
    """Base class for one recipe stage.

    A subclass normally sets :attr:`name` and :attr:`artifacts` as class
    attributes and implements :meth:`run` (and, when it can be priced before a
    run, :meth:`estimate`). Both may also be supplied to the constructor, which
    is useful for parameterised stages and for tests.

    The base :meth:`run` raises :class:`NotImplementedError`; the base
    :meth:`estimate` reports a free stage so a deterministic step does not have
    to override it. ``options`` holds the per-recipe options for this stage.
    """

    name: str = ""
    artifacts: tuple[str, ...] = ()

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        artifacts: Iterable[str] | None = None,
    ) -> None:
        if name is not None:
            self.name = _normalise_name(name)
        elif self.name:
            self.name = _normalise_name(self.name)
        self.options: dict[str, Any] = dict(options or {})
        declared = self.artifacts if artifacts is None else artifacts
        self.artifacts = _normalise_artifacts(declared)

    @property
    def stage_name(self) -> str:
        """Alias for :attr:`name`."""
        return self.name

    @property
    def artifact_declarations(self) -> tuple[str, ...]:
        """Alias for :attr:`artifacts`."""
        return self.artifacts

    @property
    def artifact_paths(self) -> tuple[str, ...]:
        """Alias for :attr:`artifacts`."""
        return self.artifacts

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
