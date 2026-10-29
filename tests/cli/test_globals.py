"""Global CLI options accepted before or after every command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.cli import CommandRegistry, GlobalOptions, main
from aivia.cli.globals import apply_global_options, build_global_parser

GLOBAL_FLAGS = [
    "--config",
    "/tmp/aivia-config.yaml",
    "--env",
    "staging",
    "--profile",
    "short",
    "--set",
    "pipeline.timezone=UTC",
    "--set",
    "video.fps=24",
    "--estimate",
    "--dry-run",
    "--yes",
    "--no-cache",
    "--live",
    "--verbose",
]


@pytest.fixture(autouse=True)
def _isolate_global_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aivia.cli.root.configure_logging", lambda level: None)


def inspect_registry(seen: list[argparse.Namespace]) -> CommandRegistry:
    commands = CommandRegistry()

    @commands.command("inspect", help="Inspect the parsed global options")
    def inspect(args: argparse.Namespace) -> int:
        seen.append(args)
        return 0

    return commands


def expected_options() -> GlobalOptions:
    return GlobalOptions(
        config="/tmp/aivia-config.yaml",
        environment="staging",
        profile="short",
        set_values=("pipeline.timezone=UTC", "video.fps=24"),
        estimate=True,
        dry_run=True,
        yes=True,
        no_cache=True,
        live=True,
        verbose=True,
    )


def test_global_flags_before_the_command_reach_the_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)

    assert main([*GLOBAL_FLAGS, "inspect"], registry=commands) == 0

    assert len(seen) == 1
    args = seen[0]
    assert args.global_options == expected_options()
    assert args.config == "/tmp/aivia-config.yaml"
    assert args.environment == "staging"
    assert args.profile == "short"
    assert args.set_values == ["pipeline.timezone=UTC", "video.fps=24"]
    assert args.estimate is True
    assert args.dry_run is True
    assert args.yes is True
    assert args.no_cache is True
    assert args.live is True
    assert args.verbose is True
    assert capsys.readouterr().err == ""


def test_global_flags_after_the_command_reach_the_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)

    assert main(["inspect", *GLOBAL_FLAGS], registry=commands) == 0

    assert seen[0].global_options == expected_options()
    assert capsys.readouterr().err == ""


def test_global_flags_default_to_off() -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)

    assert main(["inspect"], registry=commands) == 0

    args = seen[0]
    assert args.global_options == GlobalOptions()
    assert args.config is None
    assert args.environment is None
    assert args.profile is None
    assert args.set_values is None
    assert args.estimate is False
    assert args.dry_run is False
    assert args.yes is False
    assert args.no_cache is False
    assert args.live is False
    assert args.verbose is False


def test_repeated_set_flags_keep_their_order() -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)

    assert (
        main(
            ["inspect", "--set", "b=2", "--set", "a=1", "--set", "c=3"],
            registry=commands,
        )
        == 0
    )

    assert seen[0].set_values == ["b=2", "a=1", "c=3"]
    assert seen[0].global_options.set_values == ("b=2", "a=1", "c=3")


def test_global_options_expose_resolution_keywords() -> None:
    options = expected_options()

    assert options.resolution_options() == {
        "config_file": "/tmp/aivia-config.yaml",
        "environment": "staging",
        "profile": "short",
        "set_values": ["pipeline.timezone=UTC", "video.fps=24"],
    }


def test_global_resolution_flags_apply_around_a_nested_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
schema_version: 1
profiles:
  short:
    overrides:
      pipeline:
        timezone: Europe/Istanbul
""".lstrip(),
        encoding="utf-8",
    )

    code = main(
        [
            "--profile",
            "short",
            "config",
            "show",
            "--set",
            "pipeline.timezone=UTC",
            "--config",
            str(config),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    data = cast(dict[str, Any], yaml.safe_load(captured.out))
    assert data["pipeline"]["timezone"] == "UTC"


def test_global_dry_run_is_honoured_by_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    assert main(["--dry-run", "init"]) == 0

    assert not (xdg / "aivia").exists()
    assert "Dry run: no files written." in capsys.readouterr().out


def test_global_yes_is_honoured_by_a_costly_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    config = tmp_path / "config.yaml"
    config.write_text(
        """
providers:
  text:
    type: mock-llm
formats:
  sample:
    stages:
      - stage: write
    capabilities: [llm]
budget:
  confirm_above: 0.000001
""".lstrip(),
        encoding="utf-8",
    )

    code = main(
        [
            "--yes",
            "run",
            "--format",
            "sample",
            "--config",
            str(config),
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert "execution is not available" in captured.err
    assert "non-interactive runs require --yes" not in captured.err


def test_global_live_is_honoured_by_providers_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--live", "providers", "check", "--budget", "1", "mock-llm"]) == 0
    assert "budget cap" in capsys.readouterr().out


def test_verbose_configures_diagnostic_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)
    levels: list[int | str] = []

    def record(level: int | str) -> None:
        levels.append(level)

    monkeypatch.setattr("aivia.cli.root.configure_logging", record)

    assert main(["--verbose", "inspect"], registry=commands) == 0

    assert levels == [logging.DEBUG]
    assert seen[0].verbose is True


def test_root_help_lists_every_global_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--help"], registry=CommandRegistry()) == 0

    output = capsys.readouterr().out
    for flag in (
        "--config",
        "--env",
        "--profile",
        "--set",
        "--estimate",
        "--dry-run",
        "--yes",
        "--no-cache",
        "--live",
        "--verbose",
    ):
        assert flag in output


def test_command_help_includes_the_global_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)

    assert main(["inspect", "--help"], registry=commands) == 0

    output = capsys.readouterr().out
    assert "--no-cache" in output
    assert "--verbose" in output


def test_global_parser_only_returns_explicit_flags() -> None:
    provided, remaining = build_global_parser().parse_known_args(
        ["run", "--format", "sample", "--set", "a=1"]
    )

    assert provided.set_values == ["a=1"]
    assert not hasattr(provided, "config")
    assert remaining == ["run", "--format", "sample"]


def test_apply_global_options_keeps_command_defaults_when_flags_are_absent() -> None:
    args = argparse.Namespace(command="run", estimate=True, yes=False)
    provided = argparse.Namespace()

    apply_global_options(args, provided)

    assert args.estimate is True
    assert args.yes is False
    assert args.global_options == GlobalOptions(estimate=True)


def test_global_option_values_are_trimmed_and_empty_strings_are_ignored() -> None:
    seen: list[argparse.Namespace] = []
    commands = inspect_registry(seen)

    assert (
        main(
            ["inspect", "--config", "  ", "--profile", " short "],
            registry=commands,
        )
        == 0
    )

    assert seen[0].global_options.config is None
    assert seen[0].global_options.profile == "short"
