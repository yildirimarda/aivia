"""Topic source protocol, response models, registry and deterministic mock."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from pydantic import ValidationError

from aivia.core.errors import ConfigError
from aivia.core.models import ProviderMeta
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols import (
    CAPABILITY_SOURCE,
    CapabilityProvider,
    SourceCandidate,
    SourceProvider,
    SourceRequest,
    SourceResponse,
    SourceResult,
)
from aivia.providers.sources import (
    MockSource,
    MockSourceProvider,
    Source,
    SourceProviderBase,
    SourceRegistry,
    register_source,
    resolve_source,
    source_registry,
)


class RecordingSource(SourceProviderBase):
    """A tiny source that records requests and returns one candidate."""

    default_metadata = ProviderMeta(
        name="recording",
        capabilities=[CAPABILITY_SOURCE],
    )

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=[
                TopicCandidate(
                    title=f"Recorded {request.query}",
                    source_urls=["https://example.com/recorded"],
                )
            ],
        )


def test_capability_name_and_aliases_are_stable() -> None:
    assert CAPABILITY_SOURCE == "source"
    assert SourceCandidate is TopicCandidate
    assert SourceResult is TopicCandidate
    assert Source is SourceProviderBase
    assert MockSource is MockSourceProvider


def test_source_request_normalizes_aliases_and_bounds() -> None:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    request = SourceRequest.model_validate(
        {
            "brief": "  ocean tides  ",
            "lang": "EN_us",
            "max_results": 5,
            "topic_mode": "  trending  ",
            "since": start,
            "until": start + timedelta(days=2),
            "timeout": 2.5,
            "metadata": {"profile": "science"},
        }
    )

    assert request.query == "ocean tides"
    assert request.language == "en-us"
    assert request.limit == 5
    assert request.mode == "trending"
    assert request.window_start == start
    assert request.window_end == start + timedelta(days=2)
    assert request.timeout_seconds == 2.5
    assert request.metadata == {"profile": "science"}

    naive = SourceRequest(window_start=datetime(2026, 1, 1))
    assert naive.window_start == datetime(2026, 1, 1, tzinfo=UTC)

    assert SourceRequest().query == ""
    assert SourceRequest(language="  ").language is None


@pytest.mark.parametrize(
    "values",
    [
        {"limit": 0},
        {"limit": 201},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
        {"unknown": True},
    ],
)
def test_source_request_rejects_invalid_values(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SourceRequest(**values)


def test_source_request_rejects_an_inverted_window() -> None:
    with pytest.raises(ValidationError, match="window_start"):
        SourceRequest(
            window_start=datetime(2026, 2, 1, tzinfo=UTC),
            window_end=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_source_response_wraps_validated_topic_candidates() -> None:
    response = SourceResponse.model_validate(
        {
            "source": "mock-source",
            "query": "  ocean  ",
            "results": [
                {
                    "title": "  Tides explained  ",
                    "summary": "A short summary",
                    "sources": ["https://example.com/tides"],
                    "signal": 4.0,
                },
                {
                    "title": "Moon and tides",
                    "urls": [
                        "https://example.org/moon",
                        "https://example.com/tides",
                    ],
                },
            ],
        }
    )

    assert response.provider == "mock-source"
    assert response.source == "mock-source"
    assert response.query == "ocean"
    assert response.count == 2
    assert response.topics == response.candidates
    assert response.results == response.candidates
    assert response.titles == ["Tides explained", "Moon and tides"]
    assert response.urls == [
        "https://example.com/tides",
        "https://example.org/moon",
    ]
    assert isinstance(response.first, TopicCandidate)
    assert response.first is response.candidates[0]
    assert response.first.signal == 4.0


def test_source_response_allows_duplicate_candidates() -> None:
    response = SourceResponse.model_validate(
        {
            "candidates": [
                {"title": "Same topic"},
                {"title": "Same topic"},
            ]
        }
    )

    assert response.count == 2
    assert response.first is not None


def test_source_response_rejects_an_invalid_candidate() -> None:
    with pytest.raises(ValidationError):
        SourceResponse.model_validate({"candidates": [{"title": "   "}]})


def test_empty_source_response_has_no_first_candidate() -> None:
    response = SourceResponse()

    assert response.count == 0
    assert response.first is None
    assert response.titles == []
    assert response.urls == []


def test_mock_source_is_a_deterministic_protocol_provider() -> None:
    provider = MockSourceProvider()
    request = SourceRequest(query="ocean", limit=2, mode="trending")

    assert isinstance(provider, SourceProvider)
    assert isinstance(provider, CapabilityProvider)
    assert provider.supports(CAPABILITY_SOURCE)

    first = asyncio.run(provider.discover(request))
    second = asyncio.run(provider.discover(request))
    fetched = asyncio.run(provider.fetch(request))

    assert first == second == fetched
    assert first.provider == "mock-source"
    assert first.query == "ocean"
    assert first.count == 2
    assert all("ocean" in title for title in first.titles)
    assert all(candidate.mode == "trending" for candidate in first.candidates)
    assert [candidate.rank for candidate in first.candidates] == [0, 1]
    assert all(candidate.source_urls for candidate in first.candidates)
    assert len(provider.calls) == 3
    assert provider.total_cost_usd == 0.0


def test_mock_source_respects_the_limit_and_handles_an_empty_query() -> None:
    provider = MockSourceProvider()

    response = asyncio.run(provider.discover(SourceRequest(limit=1)))

    assert response.count == 1
    assert response.first is not None
    assert "mock" in response.first.title.lower()


def test_base_source_fetch_and_collect_delegate_to_discover() -> None:
    source = RecordingSource()
    request = SourceRequest(query="history")

    fetched = asyncio.run(source.fetch(request))
    collected = asyncio.run(source.collect(request))

    assert fetched.provider == "recording"
    assert fetched.titles == ["Recorded history"]
    assert [candidate.title for candidate in collected] == ["Recorded history"]
    assert source.requests == [request, request]


def test_source_registry_registers_resolves_and_rejects_duplicates() -> None:
    registry = SourceRegistry()

    registry.register_source("recording", RecordingSource)

    assert registry.resolve("recording") is RecordingSource
    assert registry.resolve_source("recording") is RecordingSource
    assert registry.get_source("recording") is RecordingSource
    assert registry.source_names() == ("recording",)

    with pytest.raises(ConfigError, match="already registered"):
        registry.register_source("recording", RecordingSource)


def test_source_registry_supports_the_decorator_form() -> None:
    registry = SourceRegistry()

    @registry.register_source("decorated")
    class Decorated(RecordingSource):
        default_metadata = ProviderMeta(
            name="decorated",
            capabilities=[CAPABILITY_SOURCE],
        )

    assert registry.resolve("decorated") is Decorated


def test_source_registry_unknown_name_lists_the_registered_sources() -> None:
    registry = SourceRegistry()
    registry.register_source("recording", RecordingSource)

    with pytest.raises(ConfigError) as error:
        registry.resolve_source("wikipedia")

    assert "wikipedia" in str(error.value)
    assert "recording" in str(error.value)


def test_shared_source_registry_contains_the_mock_source() -> None:
    assert resolve_source("mock-source") is MockSourceProvider
    assert resolve_source("mock") is MockSourceProvider
    assert cast(Any, register_source).__self__ is source_registry
