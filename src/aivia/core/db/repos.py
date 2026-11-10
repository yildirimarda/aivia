"""Repositories for the aggregates stored in the local SQLite database.

A repository owns the persistence of one aggregate. It inserts, reads, updates
and lists records without exposing SQLAlchemy statements to callers. Tables
that have a model in :mod:`aivia.core.models` return validated model instances;
the remaining tables return plain row dictionaries until their module ships a
model.

Every write runs in its own transaction and every read returns a fresh record,
so a caller cannot mutate the store by keeping a reference to a returned row.
``update`` stamps ``updated_at`` and returns ``None`` when the record does not
exist. ``list`` accepts column equality filters and returns records ordered by
their primary key, which is a time-sortable ULID for most aggregates.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    Connection,
    DateTime,
    Engine,
    RowMapping,
    Table,
    and_,
    delete,
    func,
    select,
)

from aivia.core.db.tables import (
    asset_cache,
    assets,
    compositions,
    cost_events,
    deliveries,
    embedding_cache,
    feedback,
    jobs,
    llm_cache,
    orders,
    plans,
    post_metrics,
    posts,
    ratings,
    references,
    runs,
    stage_runs,
    style_specs,
    templates,
    voice_clones,
    web_card_hits,
)
from aivia.core.ids import (
    new_asset_id,
    new_composition_id,
    new_cost_id,
    new_delivery_id,
    new_job_id,
    new_order_id,
    new_plan_id,
    new_post_id,
    new_run_id,
    new_voice_clone_id,
    ulid,
)
from aivia.core.models import (
    Asset,
    AssetCacheEntry,
    CompositionReference,
    CostEvent,
    Delivery,
    EmbeddingCacheEntry,
    Job,
    LLMCacheEntry,
    ModelBase,
    Plan,
    Post,
    Run,
    StageResult,
    UnknownFieldWarning,
    VoiceClone,
    utcnow,
)

__all__ = [
    "REPOSITORIES",
    "AssetCacheRepository",
    "AssetRepository",
    "CompositionRepository",
    "CostEventRepository",
    "DeliveryRepository",
    "EmbeddingCacheRepository",
    "FeedbackRepository",
    "JobRepository",
    "LLMCacheRepository",
    "ModelRepository",
    "OrderRepository",
    "PlanRepository",
    "PostMetricRepository",
    "PostRepository",
    "RatingRepository",
    "ReferenceRepository",
    "Repository",
    "RunRepository",
    "StageRunRepository",
    "StyleSpecRepository",
    "TableRepository",
    "TemplateRepository",
    "VoiceCloneRepository",
    "WebCardHitRepository",
    "repository_for",
]

Record = dict[str, Any]
_MIN_COUNT = 0


def _combine(
    record: ModelBase | Mapping[str, Any] | None,
    values: Mapping[str, Any],
) -> Record:
    data: Record = {}
    if isinstance(record, ModelBase):
        data.update(record.model_dump(mode="python"))
    elif record is not None:
        data.update(record)
    data.update(values)
    return data


def _prefixed(prefix: str) -> Callable[[], str]:
    return lambda: f"{prefix}_{ulid()}"


class Repository[RecordT](ABC):
    """Create, get, update and list one aggregate.

    Subclasses declare the :class:`~sqlalchemy.Table` they own, the columns that
    form its primary key and, for tables without a model, a factory for a new
    identifier. Callers never pass a connection: the repository opens a
    transaction per write and a connection per read on the configured engine.
    """

    table: ClassVar[Table]
    key_columns: ClassVar[tuple[str, ...]] = ("id",)
    id_factory: ClassVar[Callable[[], str] | None] = None

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create(
        self,
        record: ModelBase | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> RecordT:
        """Insert one record and return the stored row.

        Pass a model, a mapping or keyword values. A model fills its defaults
        and is validated before the insert; a mapping may omit the primary key
        when the repository knows how to generate one.
        """
        data = self._prepare_create(_combine(record, values))
        with self.engine.begin() as connection:
            connection.execute(self.table.insert().values(**data))
        key = tuple(data[column] for column in self.key_columns)
        stored = self.get(*key)
        if stored is None:
            raise RuntimeError(f"{type(self).__name__} could not read back its insert")
        return stored

    def create_in(
        self,
        connection: Connection,
        record: ModelBase | Mapping[str, Any] | None = None,
        **values: Any,
    ) -> RecordT:
        """Insert one record inside an existing transaction and return it.

        The caller owns the transaction, so a read that decides whether the
        insert may happen and the insert itself can be committed atomically.
        """
        data = self._prepare_create(_combine(record, values))
        connection.execute(self.table.insert().values(**data))
        key = tuple(data[column] for column in self.key_columns)
        row = (
            connection.execute(select(self.table).where(self._key_condition(key)))
            .mappings()
            .first()
        )
        if row is None:
            raise RuntimeError(f"{type(self).__name__} could not read back its insert")
        return self._to_record(row)

    def get(self, *key: Any) -> RecordT | None:
        """Return the record with *key*, or ``None`` when it does not exist."""
        with self.engine.connect() as connection:
            return self.get_in(connection, *key)

    def get_in(self, connection: Connection, *key: Any) -> RecordT | None:
        """Return the record with *key* inside an existing transaction."""
        row = (
            connection.execute(select(self.table).where(self._key_condition(key)))
            .mappings()
            .first()
        )
        return None if row is None else self._to_record(row)

    def update(self, *key: Any, **changes: Any) -> RecordT | None:
        """Apply *changes* to the record with *key* and return it.

        ``updated_at`` is refreshed on every update. An unknown record is left
        untouched and ``None`` is returned.
        """
        existing = self.get(*key)
        if existing is None:
            return None
        data = self._prepare_update(existing, changes)
        condition = self._key_condition(key)
        with self.engine.begin() as connection:
            connection.execute(self.table.update().where(condition).values(**data))
        return self.get(*key)

    def list_in(
        self,
        connection: Connection,
        *,
        limit: int | None = None,
        offset: int | None = None,
        **filters: Any,
    ) -> list[RecordT]:
        """List records inside an existing transaction.

        The filters, ordering and pagination match :meth:`list`; the caller
        supplies the connection so a read can join a write transaction.
        """
        if limit is not None and limit < _MIN_COUNT:
            raise ValueError("limit must not be negative")
        if offset is not None and offset < _MIN_COUNT:
            raise ValueError("offset must not be negative")
        statement = select(self.table)
        for name, value in filters.items():
            if name not in self.table.c:
                raise ValueError(f"{self.table.name} has no column named {name!r}")
            statement = statement.where(self.table.c[name] == value)
        statement = statement.order_by(
            *(self.table.c[column] for column in self.key_columns)
        )
        if limit is not None:
            statement = statement.limit(limit)
        if offset is not None:
            statement = statement.offset(offset)
        rows = connection.execute(statement).mappings().all()
        return [self._to_record(row) for row in rows]

    def list(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        **filters: Any,
    ) -> list[RecordT]:
        """List records matching the column equality *filters*.

        Results are ordered by the primary key so repeated calls are stable.
        """
        with self.engine.connect() as connection:
            return self.list_in(
                connection,
                limit=limit,
                offset=offset,
                **filters,
            )

    def delete(self, *key: Any) -> RecordT | None:
        """Delete the record with *key* and return it, or ``None`` when absent."""
        existing = self.get(*key)
        if existing is None:
            return None
        with self.engine.begin() as connection:
            connection.execute(delete(self.table).where(self._key_condition(key)))
        return existing

    def _key_condition(self, key: tuple[Any, ...]) -> Any:
        if len(key) != len(self.key_columns):
            expected = ", ".join(self.key_columns)
            raise ValueError(
                f"{type(self).__name__} expects the key ({expected}), "
                f"got {len(key)} value(s)"
            )
        return and_(
            *(
                self.table.c[column] == value
                for column, value in zip(self.key_columns, key, strict=True)
            )
        )

    def _clear_empty_foreign_keys(self, values: Record) -> Record:
        for name, value in values.items():
            column = self.table.c[name]
            if (
                isinstance(value, str)
                and not value
                and column.nullable
                and column.foreign_keys
            ):
                values[name] = None
        return values

    @abstractmethod
    def _prepare_create(self, data: Record) -> Record:
        """Return the column values to insert."""

    @abstractmethod
    def _prepare_update(self, existing: RecordT, changes: Mapping[str, Any]) -> Record:
        """Return the column values to update."""

    @abstractmethod
    def _to_record(self, row: RowMapping) -> RecordT:
        """Convert one database row into the repository's record type."""


class TableRepository(Repository[Record]):
    """A repository for a table whose module does not ship a model yet."""

    def _prepare_create(self, data: Record) -> Record:
        values = self._columns(data)
        if "id" in self.table.c and not values.get("id"):
            factory = type(self).id_factory
            if factory is None:
                raise ValueError(f"{type(self).__name__} needs an id")
            values["id"] = factory()
        return values

    def _prepare_update(self, existing: Record, changes: Mapping[str, Any]) -> Record:
        values = self._columns(changes)
        values["updated_at"] = utcnow()
        return values

    def _to_record(self, row: RowMapping) -> Record:
        return dict(row)

    def _columns(self, data: Mapping[str, Any]) -> Record:
        unknown = set(data) - set(self.table.c.keys())
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"{self.table.name} has no column(s) named {names}")
        return self._clear_empty_foreign_keys(dict(data))


class ModelRepository[ModelT: ModelBase](Repository[ModelT]):
    """A repository that validates and returns the aggregate's model."""

    model: ClassVar[type[ModelT]]

    def _prepare_create(self, data: Record) -> Record:
        return self._columns(self._validate(data))

    def _prepare_update(self, existing: ModelT, changes: Mapping[str, Any]) -> Record:
        data = existing.model_dump(mode="python")
        data.update(changes)
        data["updated_at"] = utcnow()
        return self._columns(self._validate(data))

    def _to_record(self, row: RowMapping) -> ModelT:
        data = dict(row)
        for name, value in data.items():
            column = self.table.c[name]
            if value is None and column.foreign_keys:
                data[name] = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnknownFieldWarning)
            return self.model.model_validate(data)

    def _validate(self, data: Mapping[str, Any]) -> Record:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnknownFieldWarning)
            dumped = self.model.model_validate(dict(data)).model_dump(mode="json")
        return {
            name: self._column_value(name, value)
            for name, value in dumped.items()
            if name in self.table.c
        }

    def _column_value(self, name: str, value: Any) -> Any:
        if isinstance(self.table.c[name].type, DateTime) and isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

    def _columns(self, data: Mapping[str, Any]) -> Record:
        return self._clear_empty_foreign_keys(
            {name: value for name, value in data.items() if name in self.table.c}
        )


class RunRepository(ModelRepository[Run]):
    """Persist pipeline, order, instruction, media and benchmark runs."""

    table = runs
    model = Run
    id_factory = new_run_id

    def find_by_idempotency_key(self, key: str) -> Run | None:
        """Return the run stored for *key*, or ``None`` when it is unused."""
        with self.engine.connect() as connection:
            return self.find_by_idempotency_key_in(connection, key)

    def find_by_idempotency_key_in(
        self,
        connection: Connection,
        key: str,
    ) -> Run | None:
        """Return the run stored for *key* inside an existing transaction."""
        text = key.strip()
        if not text:
            return None
        row = (
            connection.execute(
                select(self.table)
                .where(self.table.c.idempotency_key == text)
                .order_by(*(self.table.c[column] for column in self.key_columns))
                .limit(1)
            )
            .mappings()
            .first()
        )
        return None if row is None else self._to_record(row)


class StageRunRepository(ModelRepository[StageResult]):
    """Persist the per-stage results belonging to a run."""

    table = stage_runs
    model = StageResult
    key_columns = ("run_id", "stage")


class JobRepository(ModelRepository[Job]):
    """Persist the durable worker queue."""

    table = jobs
    model = Job
    id_factory = new_job_id


class AssetRepository(ModelRepository[Asset]):
    """Persist generated and user supplied media assets."""

    table = assets
    model = Asset
    id_factory = new_asset_id


class AssetCacheRepository(ModelRepository[AssetCacheEntry]):
    """Persist the request digest that points at a cached asset."""

    table = asset_cache
    model = AssetCacheEntry
    key_columns = ("cache_key",)


class LLMCacheRepository(ModelRepository[LLMCacheEntry]):
    """Persist request digests that point at cached LLM responses."""

    table = llm_cache
    model = LLMCacheEntry
    key_columns = ("cache_key",)

    def increment_hit_count(self, cache_key: str) -> LLMCacheEntry | None:
        """Increment one entry's hit counter in a single atomic statement.

        The addition happens in the database rather than in Python, so
        concurrent hits cannot overwrite each other's increment. An unknown
        key is left untouched and ``None`` is returned.
        """
        with self.engine.begin() as connection:
            result = connection.execute(
                self.table.update()
                .where(self.table.c.cache_key == cache_key)
                .values(
                    hit_count=self.table.c.hit_count + 1,
                    updated_at=utcnow(),
                )
            )
        if result.rowcount == 0:
            return None
        return self.get(cache_key)


class EmbeddingCacheRepository(ModelRepository[EmbeddingCacheEntry]):
    """Persist request digests that point at cached embedding vectors."""

    table = embedding_cache
    model = EmbeddingCacheEntry
    key_columns = ("cache_key",)

    def count(self) -> int:
        """Return the number of stored vectors."""
        with self.engine.connect() as connection:
            value = connection.execute(
                select(func.count()).select_from(self.table)
            ).scalar_one()
        return int(value)

    def clear(self) -> int:
        """Delete every stored vector and return how many rows were removed."""
        with self.engine.begin() as connection:
            result = connection.execute(delete(self.table))
        return int(result.rowcount or 0)

    def prune(self, max_entries: int) -> int:
        """Keep only the *max_entries* most recently written vectors."""
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise TypeError("max_entries must be an integer")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        with self.engine.begin() as connection:
            total = connection.execute(
                select(func.count()).select_from(self.table)
            ).scalar_one()
            if int(total) <= max_entries:
                return 0
            keep = (
                select(self.table.c.cache_key)
                .order_by(
                    self.table.c.updated_at.desc(),
                    self.table.c.cache_key.desc(),
                )
                .limit(max_entries)
            )
            result = connection.execute(
                delete(self.table).where(self.table.c.cache_key.not_in(keep))
            )
        return int(result.rowcount or 0)


class CompositionRepository(ModelRepository[CompositionReference]):
    """Persist references to rendered compositions."""

    table = compositions
    model = CompositionReference
    id_factory = new_composition_id


class CostEventRepository(ModelRepository[CostEvent]):
    """Persist the ledger of paid calls."""

    table = cost_events
    model = CostEvent
    id_factory = new_cost_id


class PlanRepository(ModelRepository[Plan]):
    """Persist validated assistant plans."""

    table = plans
    model = Plan
    id_factory = new_plan_id


class DeliveryRepository(ModelRepository[Delivery]):
    """Persist delivery attempts and their confirmations."""

    table = deliveries
    model = Delivery
    id_factory = new_delivery_id


class PostRepository(ModelRepository[Post]):
    """Persist posts and their publication state."""

    table = posts
    model = Post
    id_factory = new_post_id


class TemplateRepository(TableRepository):
    """Persist reusable plan and composition templates."""

    table = templates
    id_factory = _prefixed("template")


class ReferenceRepository(TableRepository):
    """Persist user supplied reference media."""

    table = references
    id_factory = _prefixed("reference")


class StyleSpecRepository(TableRepository):
    """Persist structured style descriptions extracted from references."""

    table = style_specs
    id_factory = _prefixed("style")


class FeedbackRepository(TableRepository):
    """Persist likes, dislikes and comments about a subject."""

    table = feedback
    id_factory = _prefixed("feedback")


class RatingRepository(TableRepository):
    """Persist the single rating a subject can carry."""

    table = ratings
    id_factory = _prefixed("rating")


class OrderRepository(TableRepository):
    """Persist personalized orders and their lifecycle."""

    table = orders
    id_factory = new_order_id

    def find_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        """Return the order stored for *key*, or ``None`` when it is unused."""
        with self.engine.connect() as connection:
            return self.find_by_idempotency_key_in(connection, key)

    def find_by_idempotency_key_in(
        self,
        connection: Connection,
        key: str,
    ) -> dict[str, Any] | None:
        """Return the order stored for *key* inside an existing transaction."""
        text = key.strip()
        if not text:
            return None
        row = (
            connection.execute(
                select(self.table)
                .where(self.table.c.idempotency_key == text)
                .order_by(*(self.table.c[column] for column in self.key_columns))
                .limit(1)
            )
            .mappings()
            .first()
        )
        return None if row is None else dict(row)

    def delete(self, *key: Any) -> dict[str, Any] | None:
        """Delete the record with *key* and return it, or ``None`` when absent."""
        existing = self.get(*key)
        if existing is None:
            return None
        with self.engine.begin() as connection:
            connection.execute(delete(self.table).where(self._key_condition(key)))
        return existing

    def cost_for_order(self, order_id: str) -> float:
        """Return the order's cost as the sum of its runs' costs.

        Runs are linked to an order through ``runs.order_id``. An order with no
        linked runs costs nothing.
        """
        with self.engine.connect() as connection:
            total = connection.execute(
                select(
                    func.coalesce(func.sum(runs.c.cost_usd), 0.0),
                ).where(runs.c.order_id == order_id)
            ).scalar_one()
        return float(total)

    def recompute_order_cost(self, order_id: str) -> dict[str, Any] | None:
        """Store the order's cost as the sum of its runs and return the order.

        Returns ``None`` when the order does not exist.
        """
        return self.update(order_id, cost_usd=self.cost_for_order(order_id))


class PostMetricRepository(TableRepository):
    """Persist periodic snapshots of a published post's metrics."""

    table = post_metrics
    id_factory = _prefixed("metric")


class VoiceCloneRepository(ModelRepository[VoiceClone]):
    """Persist the provider voice ids registered for an owner."""

    table = voice_clones
    model = VoiceClone
    id_factory = new_voice_clone_id

    def delete(self, *key: Any) -> VoiceClone | None:
        """Delete the record with *key* and return it, or ``None`` when absent."""
        existing = self.get(*key)
        if existing is None:
            return None
        with self.engine.begin() as connection:
            connection.execute(delete(self.table).where(self._key_condition(key)))
        return existing


class WebCardHitRepository(TableRepository):
    """Persist anonymous views of a delivered web card."""

    table = web_card_hits
    id_factory = _prefixed("hit")


REPOSITORIES: Mapping[str, type[Repository[Any]]] = {
    repository.table.name: repository
    for repository in (
        RunRepository,
        StageRunRepository,
        JobRepository,
        AssetRepository,
        AssetCacheRepository,
        LLMCacheRepository,
        EmbeddingCacheRepository,
        CompositionRepository,
        CostEventRepository,
        PlanRepository,
        TemplateRepository,
        ReferenceRepository,
        StyleSpecRepository,
        FeedbackRepository,
        RatingRepository,
        OrderRepository,
        DeliveryRepository,
        PostRepository,
        PostMetricRepository,
        VoiceCloneRepository,
        WebCardHitRepository,
    )
}
"""Every repository keyed by the table it owns."""


def repository_for(
    table: str | Table,
    engine: Engine,
) -> Repository[Any]:
    """Return the repository that owns *table*, bound to *engine*."""
    name = table if isinstance(table, str) else table.name
    try:
        repository = REPOSITORIES[name]
    except KeyError as exc:
        raise ValueError(f"no repository for table {name!r}") from exc
    return repository(engine)
