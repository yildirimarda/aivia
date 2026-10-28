"""``aivia providers``: listing registered types and live health checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest

from aivia.cli import main
from aivia.cli.providers import LIVE_BUDGET_ENV
from aivia.providers import (
    HealthStatus,
    MockImageProvider,
    MockLLMProvider,
    MockProvider,
    Provider,
    ProviderHealth,
    ProviderRegistry,
)


class _UnhealthyProvider(Provider):
    default_metadata: ClassVar[Mapping[str, Any]] = {
        "name": "broken",
        "capabilities": ["llm"],
    }

    async def check_health(self) -> ProviderHealth:
        return ProviderHealth(status=HealthStatus.UNHEALTHY, message="service down")


class _NeedsArgumentsProvider(Provider):
    def __init__(self, required: object) -> None:
        super().__init__({"name": "needs-arguments", "capabilities": ["llm"]})
        self.required = required


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> ProviderRegistry:
    """Use a small isolated registry instead of the process-wide one."""
    isolated = ProviderRegistry()
    isolated.register("mock", MockProvider)
    isolated.register("mock-llm", MockLLMProvider)
    isolated.register("mock-image", MockImageProvider)
    monkeypatch.setattr("aivia.cli.providers.provider_registry", isolated)
    return isolated


def test_list_prints_registered_types_and_capabilities(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["providers", "list"]) == 0

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "NAME\tCAPABILITIES"
    assert "mock\t" in output
    assert "mock-image\timage_gen" in output
    assert "mock-llm\tllm" in output


def test_list_filters_by_capability(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["providers", "list", "--capability", "image_gen"]) == 0

    lines = capsys.readouterr().out.splitlines()
    names = {line.split("\t", 1)[0] for line in lines[1:]}
    assert names == {"mock", "mock-image"}


def test_list_uses_static_metadata_without_constructing_the_provider(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructed: list[bool] = []

    class _Static(Provider):
        default_metadata: ClassVar[Mapping[str, Any]] = {
            "name": "static",
            "capabilities": ["tts"],
        }

        def __init__(self) -> None:
            constructed.append(True)
            super().__init__()

    registry.register("static", _Static)

    assert main(["providers", "list"]) == 0

    assert "static\ttts" in capsys.readouterr().out
    assert constructed == []


def test_check_requires_live_mode(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["providers", "check", "--budget", "1"]) == 1

    error = capsys.readouterr().err
    assert "requires live mode" in error
    assert "--live" in error


def test_check_requires_a_budget_cap(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LIVE_BUDGET_ENV, raising=False)

    assert main(["providers", "check", "--live"]) == 1

    error = capsys.readouterr().err
    assert "requires a budget cap" in error
    assert LIVE_BUDGET_ENV in error


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_check_rejects_a_non_positive_budget(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
    budget: str,
) -> None:
    assert main(["providers", "check", "--live", "--budget", budget]) == 1

    assert "positive amount" in capsys.readouterr().err


def test_check_runs_health_checks_under_the_budget(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["providers", "check", "--live", "--budget", "2.50"]) == 0

    output = capsys.readouterr().out
    assert "mock: healthy" in output
    assert "mock-image: healthy" in output
    assert "mock-llm: healthy" in output
    assert "checked 3 provider(s) with a $2.50 budget cap: 0 failed" in output


def test_check_reads_the_budget_from_the_environment(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LIVE_BUDGET_ENV, "1.25")

    assert main(["providers", "check", "--live"]) == 0

    assert "$1.25 budget cap" in capsys.readouterr().out


def test_check_accepts_an_explicit_provider_name(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["providers", "check", "--live", "--budget", "1", "mock-llm"]) == 0

    output = capsys.readouterr().out
    assert "mock-llm: healthy" in output
    assert "mock-image" not in output


def test_check_rejects_an_unknown_provider(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["providers", "check", "--live", "--budget", "1", "missing"]) == 1

    assert "unknown provider" in capsys.readouterr().err


def test_check_fails_when_a_provider_is_unhealthy(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.register("broken", _UnhealthyProvider)

    assert main(["providers", "check", "--live", "--budget", "1", "broken"]) == 1

    output = capsys.readouterr().out
    assert "broken: unhealthy" in output
    assert "service down" in output


def test_check_reports_a_provider_that_cannot_be_constructed(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.register("needs-arguments", _NeedsArgumentsProvider)

    assert main(["providers", "check", "--live", "--budget", "1"]) == 1

    error = capsys.readouterr().err
    assert "needs-arguments: error:" in error
    assert "TypeError" in error


def test_check_closes_each_provider_after_the_probe(
    registry: ProviderRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    created: list[Provider] = []

    class _Tracking(Provider):
        default_metadata: ClassVar[Mapping[str, Any]] = {
            "name": "tracking",
            "capabilities": ["llm"],
        }

        def __init__(self) -> None:
            super().__init__()
            created.append(self)

    registry.register("tracking", _Tracking)

    assert main(["providers", "check", "--live", "--budget", "1", "tracking"]) == 0

    assert len(created) == 1
    assert created[0].closed
    assert "tracking: healthy" in capsys.readouterr().out
