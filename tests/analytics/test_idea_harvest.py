"""Harvesting comments from read-capable accounts into manual queue topic suggestions.

Idea harvesting is opted in through the ``analytics`` config section, reads
comments through a read-capable API, requires the ``read`` scope, deduplicates
and clusters comments into topic suggestions, and appends them to the manual
queue.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from aivia.analytics.harvest import (
    ANALYTICS_SECTION_KEY,
    IdeaHarvest,
    IdeaHarvestConfig,
    register_analytics_section,
)
from aivia.core.config.registry import SectionRegistry
from aivia.core.config.schedule import PublishConfig
from aivia.core.config.sections import register_core_sections
from aivia.core.errors import ConfigError
from aivia.publish.account_reference import (
    CAPABILITY_READ,
    MissingScopeError,
    ReferenceAccount,
    ReferenceAccountSource,
    TopPost,
)
from aivia.publish.accounts import AccountMap
from aivia.research.modes.manual import ManualMode


def _publish(**accounts: object) -> PublishConfig:
    return PublishConfig.from_mapping({"accounts": dict(accounts)})


def _account(
    platform: str = "x",
    language: str = "en",
    scopes: Sequence[str] | None = None,
) -> dict[str, object]:
    return {
        "credential_refs": ["TOKEN"],
        "options": {
            "scopes": list(scopes) if scopes is not None else ["read", "publish"]
        },
    }


class FakeSource(ReferenceAccountSource):
    """A read-capable API fake recording every top_posts call."""

    def __init__(self) -> None:
        self._posts_by_account: dict[str, list[TopPost]] = {}
        self.calls: list[tuple[str, int]] = []

    async def top_posts(
        self, account: ReferenceAccount, *, limit: int = 10
    ) -> list[TopPost]:
        self.calls.append((account.key, limit))
        return list(self._posts_by_account.get(account.key, []))


def _posts(*texts: str) -> list[TopPost]:
    return [
        TopPost(
            post_id=f"post-{index}",
            text=text,
            url=f"https://example.com/post-{index}",
            score=90.0 - index * 5,
        )
        for index, text in enumerate(texts, start=1)
    ]


def _harvest(
    platform: str = "x",
    language: str = "en",
    *,
    enabled: bool = True,
    queue: ManualMode | None = None,
    scopes: Sequence[str] | None = None,
) -> tuple[IdeaHarvest, FakeSource]:
    publish = _publish(**{f"{platform}/{language}": _account(scopes=scopes)})
    source = FakeSource()
    harvest = IdeaHarvest(
        AccountMap(publish),
        source,
        config=IdeaHarvestConfig(enabled=enabled),
        queue=queue,
    )
    return harvest, source


# ---------------------------------------------------------------------------
# Opt in
# ---------------------------------------------------------------------------


def test_not_opted_in_appends_nothing_and_reads_nothing() -> None:
    queue = ManualMode()
    harvest, source = _harvest(enabled=False, queue=queue)
    source._posts_by_account["x/en"] = _posts("a comment")

    added = asyncio.run(harvest.harvest("x", "en"))

    assert added == []
    assert queue.empty
    assert source.calls == []


def test_opted_in_appends_topic_suggestions() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts(
        "the quick brown fox", "bright red sunset"
    )

    added = asyncio.run(harvest.harvest("x", "en"))

    assert [candidate.title for candidate in added] == [
        "the quick brown fox",
        "bright red sunset",
    ]
    assert queue.pending == 2
    assert all(candidate.source_urls for candidate in added)
    assert [candidate.signal for candidate in added] == [85.0, 80.0]
    assert source.calls == [("x/en", 10)]


def test_suggestions_are_tagged_as_idea_harvest_before_enqueue() -> None:
    harvest, source = _harvest()
    source._posts_by_account["x/en"] = _posts("a comment")

    suggestions = harvest._suggest(source._posts_by_account["x/en"])

    assert len(suggestions) == 1
    assert suggestions[0].mode == "idea_harvest"
    assert suggestions[0].title == "a comment"


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


def test_duplicate_comments_dedupe_to_one_suggestion() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts(
        "the quick brown fox", "the quick brown fox"
    )

    added = asyncio.run(harvest.harvest("x", "en"))

    assert len(added) == 1
    assert queue.pending == 1
    assert source.calls == [("x/en", 10)]


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


def test_similar_comments_cluster_into_one_suggestion() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts(
        "the quick brown fox", "the quick brown dog"
    )

    added = asyncio.run(harvest.harvest("x", "en"))

    assert len(added) == 1
    assert queue.pending == 1


def test_dissimilar_comments_stay_separate_suggestions() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts(
        "the quick brown fox", "bright red sunset"
    )

    added = asyncio.run(harvest.harvest("x", "en"))

    assert len(added) == 2
    assert queue.pending == 2


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_missing_read_scope_raises_a_clear_error() -> None:
    publish = _publish(**{"x/en": _account(scopes=["publish"])})
    harvest = IdeaHarvest(
        AccountMap(publish), FakeSource(), config=IdeaHarvestConfig(enabled=True)
    )

    with pytest.raises(MissingScopeError) as caught:
        asyncio.run(harvest.harvest("x", "en"))

    error = caught.value
    assert isinstance(error, ConfigError)
    assert "not authorised to read comments" in str(error)
    assert error.details["account"] == "x/en"
    assert error.details["required_scope"] == CAPABILITY_READ
    assert error.details["scopes"] == ["publish"]
    assert "x/en" in str(error)


def test_account_without_any_scopes_raises() -> None:
    publish = _publish(**{"x/en": _account(scopes=[])})
    harvest = IdeaHarvest(
        AccountMap(publish), FakeSource(), config=IdeaHarvestConfig(enabled=True)
    )

    with pytest.raises(MissingScopeError) as caught:
        asyncio.run(harvest.harvest("x", "en"))

    assert caught.value.details["scopes"] == []


def test_unknown_account_raises_config_error() -> None:
    harvest, _source = _harvest()

    with pytest.raises(ConfigError):
        asyncio.run(harvest.harvest("tiktok", "en"))


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_comments_are_cached_and_not_refetched() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts("a comment")

    asyncio.run(harvest.harvest("x", "en"))
    asyncio.run(harvest.harvest("x", "en"))

    assert source.calls == [("x/en", 10)]


def test_different_limits_refetch() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts("a", "b")

    asyncio.run(harvest.harvest("x", "en", limit=3))
    asyncio.run(harvest.harvest("x", "en", limit=5))

    assert source.calls == [("x/en", 3), ("x/en", 5)]


# ---------------------------------------------------------------------------
# Cap
# ---------------------------------------------------------------------------


def test_max_suggestions_caps_appended_suggestions() -> None:
    queue = ManualMode()
    harvest, source = _harvest(queue=queue)
    source._posts_by_account["x/en"] = _posts(
        *[
            f"topic {word}"
            for word in [
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
                "iota",
                "kappa",
                "lambda",
                "mu",
            ]
        ]
    )

    added = asyncio.run(harvest.harvest("x", "en"))

    assert len(added) == 10
    assert queue.pending == 10


# ---------------------------------------------------------------------------
# Config section
# ---------------------------------------------------------------------------


def test_analytics_section_registers() -> None:
    sections = SectionRegistry()
    register_analytics_section(sections)

    assert "analytics" in sections


def test_analytics_section_is_idempotent() -> None:
    sections = SectionRegistry()
    register_analytics_section(sections)
    register_analytics_section(sections)

    assert len(sections) == 1


def test_analytics_opt_in_config_parses() -> None:
    sections = SectionRegistry()
    register_core_sections(sections)
    register_analytics_section(sections)

    parsed = sections.parse({"analytics": {"enabled": True}})

    analytics = parsed.analytics  # type: ignore[attr-defined]
    assert isinstance(analytics, IdeaHarvestConfig)
    assert analytics.enabled is True
    assert analytics.max_suggestions == 10


def test_analytics_is_an_installed_module() -> None:
    from aivia.core.config.installed import installed_module_names

    assert "analytics" in installed_module_names()


def test_analytics_section_key() -> None:
    assert ANALYTICS_SECTION_KEY == "analytics"
