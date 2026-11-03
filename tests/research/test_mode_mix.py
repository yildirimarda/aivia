"""Mode mix: weighted combination, seeded determinism and per-mode tags."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.research import (
    DEFAULT_SEED,
    MixedMode,
    MixMode,
    ModeMix,
    ModeMixer,
    TopicModeBase,
    TopicModeRegistry,
    combine_modes,
    mix_modes,
    normalize_weights,
    resolve_mode,
    weighted_interleave,
)


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def _candidate(title: str, mode: str = "") -> Any:
    from aivia.core.models.topic import TopicCandidate

    return TopicCandidate(title=title, mode=mode)


class RecordingMode(TopicModeBase):
    """A mode returning fixed titles and recording every request."""

    def __init__(
        self,
        name: str,
        titles: list[str],
        *,
        fail: bool = False,
        prefix: str | None = None,
    ) -> None:
        super().__init__(name)
        self.titles = list(titles)
        self.fail = fail
        self.prefix = prefix
        self.requests: list[SourceRequest] = []

    async def discover(self, request: SourceRequest) -> SourceResponse:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(f"boom: {self.name}")
        return SourceResponse(
            provider=self.name,
            query=request.query,
            candidates=[
                _candidate(
                    title,
                    "wrong" if self.prefix is None else self.prefix,
                )
                for title in self.titles
            ],
        )


def test_the_mode_is_registered_and_aliases_share_one_class() -> None:
    assert resolve_mode("mix") is ModeMix
    assert resolve_mode("mixed") is ModeMix
    assert ModeMix is MixedMode is MixMode is ModeMixer
    assert ModeMix().name == "mix"
    assert DEFAULT_SEED == 0


def test_weights_combine_the_streams_and_tag_every_candidate() -> None:
    alpha = RecordingMode("alpha", ["a1", "a2", "a3"])
    beta = RecordingMode("beta", ["b1", "b2", "b3"])
    mode = ModeMix(
        {"alpha": alpha, "beta": beta},
        weights={"alpha": 2, "beta": 1},
        seed=7,
    )

    response = _run(mode.discover(SourceRequest(query="history")))

    assert response.titles == ["a1", "b1", "a2", "a3", "b2", "b3"]
    assert [candidate.mode for candidate in response.candidates] == [
        "alpha",
        "beta",
        "alpha",
        "alpha",
        "beta",
        "beta",
    ]
    assert [candidate.rank for candidate in response.candidates] == list(range(6))
    assert [request.mode for request in alpha.requests] == ["alpha"]
    assert [request.mode for request in beta.requests] == ["beta"]
    assert response.provider == "mix"
    assert response.query == "history"
    assert response.metadata["mode"] == "mix"
    assert response.metadata["mode_weights"] == {"alpha": 2.0, "beta": 1.0}
    assert response.metadata["mode_counts"] == {"alpha": 3, "beta": 3}
    assert response.metadata["found_count"] == 6
    assert response.metadata["served_count"] == 6


def test_the_same_seed_repeats_and_another_seed_can_reorder() -> None:
    def build(seed: int) -> ModeMix:
        return ModeMix(
            {
                name: RecordingMode(name, [name])
                for name in ("alpha", "beta", "gamma", "delta", "epsilon")
            },
            seed=seed,
        )

    first = _run(build(1).discover(SourceRequest()))
    second = _run(build(1).discover(SourceRequest()))
    other = _run(build(2).discover(SourceRequest()))

    assert first.titles == second.titles
    assert first.titles != other.titles


def test_weights_control_the_served_mix_under_a_limit() -> None:
    alpha = RecordingMode("alpha", [f"a{index}" for index in range(10)])
    beta = RecordingMode("beta", [f"b{index}" for index in range(10)])
    mode = ModeMix(
        {"alpha": alpha, "beta": beta},
        weights={"alpha": 2, "beta": 1},
        limit=9,
        seed=3,
    )

    response = _run(mode.discover(SourceRequest()))

    assert len(response.candidates) == 9
    assert response.metadata["mode_counts"] == {"alpha": 6, "beta": 3}
    assert response.titles == [
        "a0",
        "b0",
        "a1",
        "a2",
        "b1",
        "a3",
        "a4",
        "b2",
        "a5",
    ]


def test_zero_weight_modes_are_not_run() -> None:
    alpha = RecordingMode("alpha", ["a1"])
    beta = RecordingMode("beta", ["b1"])
    mode = ModeMix({"alpha": alpha, "beta": beta}, weights={"alpha": 1, "beta": 0})

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["a1"]
    assert beta.requests == []
    assert mode.shares == {"alpha": 1.0, "beta": 0.0}


def test_candidates_can_be_deduplicated_across_modes() -> None:
    alpha = RecordingMode("alpha", ["Same topic"])
    beta = RecordingMode("beta", [" same   TOPIC "])
    mode = ModeMix({"alpha": alpha, "beta": beta}, dedupe=True, seed=1)

    response = _run(mode.discover(SourceRequest()))

    assert response.titles == ["Same topic"]
    assert response.metadata["skipped_duplicates"] == 1


def test_a_failing_mode_propagates_unless_errors_are_tolerated() -> None:
    alpha = RecordingMode("alpha", ["a1"])
    beta = RecordingMode("beta", [], fail=True)

    with pytest.raises(RuntimeError, match="boom: beta"):
        _run(ModeMix({"alpha": alpha, "beta": beta}).discover(SourceRequest()))

    response = _run(
        ModeMix(
            {"alpha": alpha, "beta": beta},
            continue_on_error=True,
        ).discover(SourceRequest())
    )

    assert response.titles == ["a1"]
    assert response.metadata["failed_modes"] == ["beta"]
    assert response.metadata["errors"][0]["mode"] == "beta"


class AlphaMode(TopicModeBase):
    name = "alpha"

    async def discover(self, request: SourceRequest) -> SourceResponse:
        return SourceResponse(candidates=[_candidate("a1")])


class BetaMode(TopicModeBase):
    name = "beta"

    async def discover(self, request: SourceRequest) -> SourceResponse:
        return SourceResponse(candidates=[_candidate("b1")])


def test_modes_accept_entries_weights_and_a_custom_registry() -> None:
    registry = TopicModeRegistry()
    registry.register("alpha", AlphaMode)

    mode = ModeMix(
        [("alpha", "alpha", 3)],
        registry=registry,
    )

    assert mode.names == ("alpha",)
    assert mode.weights == {"alpha": 3.0}
    assert isinstance(mode.modes[0], AlphaMode)


def test_a_profile_supplies_the_mode_names_and_weights() -> None:
    registry = TopicModeRegistry()
    registry.register("alpha", AlphaMode)
    registry.register("beta", BetaMode)

    mode = ModeMix.from_profile(
        {"topic_modes": {"alpha": 0.75, "beta": 0.25}},
        registry=registry,
    )

    assert mode.names == ("alpha", "beta")
    assert mode.weights == {"alpha": 0.75, "beta": 0.25}


@pytest.mark.parametrize(
    "weights",
    [
        {"alpha": -1},
        {"alpha": float("inf")},
        {"alpha": "many"},
        {"missing": 1},
    ],
)
def test_invalid_weights_fail_clearly(weights: dict[str, Any]) -> None:
    alpha = RecordingMode("alpha", ["a1"])

    with pytest.raises(ConfigError):
        ModeMix({"alpha": alpha}, weights=weights)


def test_invalid_names_seeds_limits_and_duplicates_are_rejected() -> None:
    alpha = RecordingMode("alpha", ["a1"])

    with pytest.raises(ConfigError, match="at least one topic mode"):
        _run(ModeMix({}).discover(SourceRequest()))
    with pytest.raises(ConfigError, match="positive weight"):
        ModeMix({"alpha": alpha}, weights={"alpha": 0})
    with pytest.raises(ConfigError, match="duplicate topic mode"):
        ModeMix([("alpha", alpha), ("alpha", alpha)])
    with pytest.raises(ConfigError, match="seed"):
        ModeMix({"alpha": alpha}, seed="1")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="limit"):
        ModeMix({"alpha": alpha}, limit=-1)


def test_normalize_weights_is_relative_by_default_and_can_require_shares() -> None:
    assert normalize_weights({"alpha": 3, "beta": 1}) == {
        "alpha": 3.0,
        "beta": 1.0,
    }
    assert normalize_weights({"alpha": 0.75, "beta": 0.25}, require_sum=True) == {
        "alpha": 0.75,
        "beta": 0.25,
    }
    with pytest.raises(ConfigError, match="sum to 1"):
        normalize_weights({"alpha": 2, "beta": 1}, require_sum=True)


def test_weighted_interleave_works_without_a_mode_class() -> None:
    combined = weighted_interleave(
        {
            "alpha": [{"title": "a1"}, {"title": "a2"}],
            "beta": [{"title": "b1"}],
        },
        {"alpha": 1, "beta": 1},
        seed=4,
    )

    assert [candidate.title for candidate in combined] == ["a1", "b1", "a2"]
    assert [candidate.mode for candidate in combined] == ["alpha", "beta", "alpha"]


def test_the_helpers_return_a_list_like_response() -> None:
    alpha = RecordingMode("alpha", ["a1"])
    beta = RecordingMode("beta", ["b1"])

    response = _run(
        mix_modes(
            {"alpha": alpha, "beta": beta},
            {"alpha": 1, "beta": 1},
            seed=1,
            query="news",
        )
    )
    alias = _run(
        combine_modes(
            {"alpha": alpha, "beta": beta},
            {"alpha": 1, "beta": 1},
            seed=1,
        )
    )

    assert response.query == "news"
    assert response.titles == ["a1", "b1"]
    assert list(response) == response.candidates
    assert response[0].title == "a1"
    assert len(response) == 2
    assert response == response.candidates
    assert alias.titles == response.titles
