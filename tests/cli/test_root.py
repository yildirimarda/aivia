"""Root CLI contract: version flag, subcommand registry and dispatch."""

from __future__ import annotations

import argparse

import pytest

from aivia import __version__
from aivia.cli import (
    Command,
    CommandRegistry,
    build_parser,
    main,
    version_string,
)


def test_version_flag_prints_the_installed_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"aivia {__version__}"


def test_short_version_flag_is_an_alias(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["-V"]) == 0
    assert capsys.readouterr().out.strip() == version_string()


def test_no_arguments_keeps_printing_the_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([], registry=CommandRegistry()) == 0
    assert capsys.readouterr().out.strip() == version_string()


def test_help_lists_registered_commands(capsys: pytest.CaptureFixture[str]) -> None:
    commands = CommandRegistry()
    commands.register(
        Command(name="example", help="Run the example command", handler=lambda _: 0)
    )

    assert main(["--help"], registry=commands) == 0
    output = capsys.readouterr().out

    assert "COMMAND" in output
    assert "example" in output
    assert "Run the example command" in output


def test_registered_command_receives_parsed_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands = CommandRegistry()

    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--name", default="world")

    @commands.command("hello", help="Say hello", configure=configure)
    def hello(args: argparse.Namespace) -> int:
        print(f"hello {args.name}")
        return 0

    assert main(["hello", "--name", "aivia"], registry=commands) == 0
    assert capsys.readouterr().out.strip() == "hello aivia"


def test_command_return_code_becomes_the_exit_code() -> None:
    commands = CommandRegistry()
    commands.register_command("fail", lambda _: 3, help="Fail on purpose")

    assert main(["fail"], registry=commands) == 3


def test_handler_returning_none_exits_successfully() -> None:
    commands = CommandRegistry()
    commands.register(Command(name="quiet", handler=lambda _: None))

    assert main(["quiet"], registry=commands) == 0


def test_unknown_command_fails_with_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["missing"], registry=CommandRegistry()) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_registry_rejects_duplicate_command_names() -> None:
    commands = CommandRegistry()
    commands.register(Command(name="same", handler=lambda _: 0))

    with pytest.raises(ValueError, match="already registered"):
        commands.register(Command(name="same", handler=lambda _: 0))


@pytest.mark.parametrize("name", ["", "   ", "-flag", "two words"])
def test_registry_rejects_malformed_command_names(name: str) -> None:
    commands = CommandRegistry()

    with pytest.raises(ValueError, match="command name"):
        commands.register(Command(name=name, handler=lambda _: 0))


def test_registry_looks_up_commands_and_lists_them_deterministically() -> None:
    commands = CommandRegistry()
    commands.register(Command(name="zulu", handler=lambda _: 0))
    commands.register(Command(name="alpha", handler=lambda _: 0))

    assert commands.names() == ("alpha", "zulu")
    assert "alpha" in commands
    assert "missing" not in commands
    assert len(commands) == 2
    assert commands.get("alpha").name == "alpha"
    with pytest.raises(KeyError, match="unknown command: missing"):
        commands.get("missing")


def test_build_parser_includes_registered_commands() -> None:
    commands = CommandRegistry()
    commands.register(Command(name="example", handler=lambda _: 0))

    parser = build_parser(commands)
    args = parser.parse_args(["example"])

    assert args.command == "example"
    assert args._command.name == "example"
