"""Human ratings drive the best quality and weighted router strategies.

Ratings are captured into the ``ratings`` table by the analytics module and
consumed by the router through a name-to-score lookup. Capturing a rating for a
provider must flip which provider the ``best_quality`` and ``weighted_ab``
strategies select, proving the ratings drive the order rather than static
ranks or configured weights alone.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine

from aivia.analytics import all_ratings, capture_rating, rating_for
from aivia.core.config import CapabilityChainConfig, CapabilityStrategy
from aivia.core.db import create_database_engine, upgrade_database
from aivia.core.models import ProviderMeta
from aivia.core.router import (
    MissingQualityRankWarning,
    Router,
)
from aivia.providers.base import Provider


class RatedProvider(Provider):
    """A provider with a name, the image_gen capability and an optional rank."""

    def __init__(self, name: str, quality_rank: int | None = None) -> None:
        super().__init__(
            ProviderMeta.model_validate(
                {
                    "name": name,
                    "capabilities": ["image_gen"],
                    "quality_rank": quality_rank,
                }
            )
        )


@pytest.fixture
def engine() -> Iterator[Engine]:
    created = create_database_engine(url="sqlite+pysqlite:///:memory:")
    upgrade_database(engine=created)
    try:
        yield created
    finally:
        created.dispose()


def _router(
    engine: Engine,
    *providers: RatedProvider,
    strategy: CapabilityStrategy,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
) -> Router:
    ratings = {
        provider.name: score
        for provider in providers
        if (score := rating_for(engine, provider.name)) is not None
    }
    return Router(
        providers={provider.name: provider for provider in providers},
        config=CapabilityChainConfig(
            strategy=strategy,
            chain=[provider.name for provider in providers],
            weights=weights or {},
        ),
        seed=seed,
        ratings=ratings or None,
    )


def test_capture_rating_round_trip(engine: Engine) -> None:
    stored = capture_rating(engine, "image-a", 4.5, rater="alice", comment="solid")
    assert stored["subject_kind"] == "provider"
    assert stored["subject_id"] == "image-a"
    assert stored["score"] == 4.5
    assert stored["rater"] == "alice"
    assert rating_for(engine, "image-a") == 4.5


def test_capture_rating_upserts_instead_of_duplicating(engine: Engine) -> None:
    capture_rating(engine, "image-a", 2.0)
    capture_rating(engine, "image-a", 5.0)
    assert rating_for(engine, "image-a") == 5.0
    assert all_ratings(engine) == {"image-a": 5.0}


def test_rating_for_returns_none_when_unrated(engine: Engine) -> None:
    assert rating_for(engine, "never-rated") is None


def test_best_quality_rating_beats_static_rank(engine: Engine) -> None:
    low = RatedProvider("low", quality_rank=1)
    high = RatedProvider("high", quality_rank=9)
    router = _router(engine, low, high, strategy=CapabilityStrategy.BEST_QUALITY)
    assert router.select("image_gen") is high

    capture_rating(engine, "low", 9.5)
    router = _router(engine, low, high, strategy=CapabilityStrategy.BEST_QUALITY)
    assert router.select("image_gen") is low


def test_best_quality_still_uses_static_rank_when_unrated(engine: Engine) -> None:
    low = RatedProvider("low", quality_rank=1)
    high = RatedProvider("high", quality_rank=9)
    router = _router(engine, low, high, strategy=CapabilityStrategy.BEST_QUALITY)
    assert router.select("image_gen") is high


def test_best_quality_warns_for_every_unrated_unranked_candidate(
    engine: Engine,
) -> None:
    first = RatedProvider("first")
    second = RatedProvider("second")
    router = _router(engine, first, second, strategy=CapabilityStrategy.BEST_QUALITY)

    with pytest.warns(MissingQualityRankWarning) as caught:
        selection = router.select_with_record("image_gen")

    assert len(caught) == 2
    assert selection.quality_rank is None


def test_weighted_ab_shifts_toward_the_highly_rated_provider(engine: Engine) -> None:
    first = RatedProvider("first")
    second = RatedProvider("second")
    router = _router(
        engine,
        first,
        second,
        strategy=CapabilityStrategy.WEIGHTED_AB,
        weights={"first": 1.0, "second": 1.0},
        seed=3,
    )
    assert router.select("image_gen") is first

    capture_rating(engine, "second", 5.0)
    router = _router(
        engine,
        first,
        second,
        strategy=CapabilityStrategy.WEIGHTED_AB,
        weights={"first": 1.0, "second": 1.0},
        seed=3,
    )
    assert router.select("image_gen") is second
