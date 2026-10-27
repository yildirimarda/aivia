"""Secrets loader contract: environment, file, keyring and token store key."""

from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path

import pytest

from aivia.core.errors import ConfigError
from aivia.core.secrets import (
    DEFAULT_SECRETS_FILENAME,
    ENV_PREFIX,
    TOKEN_STORE_KEY,
    TOKEN_STORE_KEY_ENV,
    MissingSecretError,
    SecretsLoader,
    default_secrets_file,
)


class FakeKeyring:
    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self.values = dict(values or {})
        self.calls: list[tuple[str, str]] = []

    def get_password(self, service_name: str, username: str) -> str | None:
        self.calls.append((service_name, username))
        return self.values.get(username)


class BrokenKeyring:
    def get_password(self, service_name: str, username: str) -> str | None:
        raise RuntimeError("keyring unavailable")


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower()


def _loader(tmp_path: Path, **kwargs: object) -> SecretsLoader:
    options: dict[str, object] = {
        "environ": {"XDG_CONFIG_HOME": str(tmp_path)},
    }
    options.update(kwargs)
    return SecretsLoader(**options)  # type: ignore[arg-type]


def test_environment_is_checked_before_file_and_keyring(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=file\n", encoding="utf-8")
    keyring = FakeKeyring({"API_KEY": "keyring"})

    loader = SecretsLoader(
        secrets_file=secrets,
        environ={ENV_PREFIX + "API_KEY": "environment"},
        keyring=keyring,
    )

    assert loader.get("API_KEY") == "environment"
    assert keyring.calls == []


def test_prefixed_environment_variable_wins_over_the_bare_name() -> None:
    loader = SecretsLoader(
        environ={
            ENV_PREFIX + "API_KEY": "prefixed",
            "API_KEY": "bare",
        }
    )

    assert loader.get("API_KEY") == "prefixed"


def test_bare_environment_variable_is_accepted() -> None:
    loader = SecretsLoader(environ={"API_KEY": "bare"})

    assert loader.get("API_KEY") == "bare"


def test_file_is_checked_before_keyring(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=file\n", encoding="utf-8")
    keyring = FakeKeyring({"API_KEY": "keyring"})

    loader = SecretsLoader(secrets_file=secrets, environ={}, keyring=keyring)

    assert loader.get("API_KEY") == "file"
    assert keyring.calls == []


def test_keyring_is_used_when_environment_and_file_are_missing(
    tmp_path: Path,
) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("OTHER=file\n", encoding="utf-8")
    keyring = FakeKeyring({"API_KEY": "keyring"})

    loader = SecretsLoader(secrets_file=secrets, environ={}, keyring=keyring)

    assert loader.get("API_KEY") == "keyring"
    assert keyring.calls == [("aivia", "API_KEY")]


def test_keyring_service_is_configurable(tmp_path: Path) -> None:
    keyring = FakeKeyring({"API_KEY": "keyring"})

    loader = _loader(
        tmp_path,
        keyring=keyring,
        keyring_service="aivia-test",
    )

    assert loader.get("API_KEY") == "keyring"
    assert keyring.calls == [("aivia-test", "API_KEY")]


def test_a_broken_keyring_falls_back_to_missing(tmp_path: Path) -> None:
    loader = _loader(tmp_path, keyring=BrokenKeyring())

    assert loader.get_optional("API_KEY") is None
    with pytest.raises(MissingSecretError):
        loader.get("API_KEY")


def test_keyring_is_declared_as_an_optional_extra() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    project = pyproject["project"]
    core = {_requirement_name(dep) for dep in project["dependencies"]}
    extras = {
        name: {_requirement_name(dep) for dep in deps}
        for name, deps in project.get("optional-dependencies", {}).items()
    }

    assert "keyring" not in core
    assert "keyring" in extras.get("keyring", set())


def test_an_installed_keyring_is_discovered_without_an_explicit_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keyring = FakeKeyring({"API_KEY": "keyring"})
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    loader = _loader(tmp_path)

    assert loader.get("API_KEY") == "keyring"
    assert keyring.calls == [("aivia", "API_KEY")]


def test_a_missing_keyring_extra_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)

    loader = _loader(tmp_path)

    assert loader.get_optional("API_KEY") is None


def test_file_supports_comments_exports_quotes_and_prefixed_keys(
    tmp_path: Path,
) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "AIVA_OPENAI_API_KEY=from-file",
                'QUOTED="value with spaces"',
                "SINGLE='single value'",
                "export EXPORTED=yes",
                "EMPTY=",
            ]
        ),
        encoding="utf-8",
    )
    keyring = FakeKeyring({"EMPTY": "keyring"})
    loader = SecretsLoader(secrets_file=secrets, environ={}, keyring=keyring)

    assert loader.get("OPENAI_API_KEY") == "from-file"
    assert loader.get("AIVA_OPENAI_API_KEY") == "from-file"
    assert loader.get("QUOTED") == "value with spaces"
    assert loader.get("SINGLE") == "single value"
    assert loader.get("EXPORTED") == "yes"
    assert loader.get("EMPTY") == "keyring"


def test_default_secrets_file_uses_the_xdg_config_directory(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config"
    secrets = config_root / "aivia" / DEFAULT_SECRETS_FILENAME
    secrets.parent.mkdir(parents=True)
    secrets.write_text("API_KEY=default\n", encoding="utf-8")
    environ = {"XDG_CONFIG_HOME": str(config_root)}

    assert default_secrets_file(environ) == secrets
    assert SecretsLoader(environ=environ).get("API_KEY") == "default"


def test_secrets_file_can_be_selected_by_environment(tmp_path: Path) -> None:
    secrets = tmp_path / "custom.env"
    secrets.write_text("API_KEY=custom\n", encoding="utf-8")

    loader = SecretsLoader(environ={ENV_PREFIX + "SECRETS_FILE": str(secrets)})

    assert loader.get("API_KEY") == "custom"


def test_an_explicit_missing_secrets_file_is_an_error(tmp_path: Path) -> None:
    missing = tmp_path / "absent.env"
    loader = SecretsLoader(secrets_file=missing, environ={})

    with pytest.raises(ConfigError) as excinfo:
        loader.get("API_KEY")

    assert str(missing) in str(excinfo.value)


def test_an_invalid_file_line_names_the_line_but_not_its_content(
    tmp_path: Path,
) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text("API_KEY=file\nnot a pair\n", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    with pytest.raises(ConfigError) as excinfo:
        loader.get("API_KEY")

    message = str(excinfo.value)
    assert "line 2" in message
    assert "not a pair" not in message


def test_missing_error_names_only_the_missing_reference(tmp_path: Path) -> None:
    loader = _loader(
        tmp_path,
        environ={
            "XDG_CONFIG_HOME": str(tmp_path),
            ENV_PREFIX + "OTHER_SECRET": "another-value",
        },
    )

    with pytest.raises(MissingSecretError) as excinfo:
        loader.get("MISSING_TOKEN")

    error = excinfo.value
    assert error.reference == "MISSING_TOKEN"
    assert error.details == {"reference": "MISSING_TOKEN"}
    assert str(error) == "missing secret: MISSING_TOKEN"
    assert "another-value" not in str(error)
    assert "OTHER_SECRET" not in str(error)


def test_an_empty_reference_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _loader(tmp_path).get("   ")


def test_token_store_key_is_exposed_from_the_environment() -> None:
    loader = SecretsLoader(environ={TOKEN_STORE_KEY_ENV: "encryption-key"})

    assert loader.get_token_store_key() == "encryption-key"
    assert loader.token_store_key == "encryption-key"


def test_token_store_key_can_come_from_the_secrets_file(tmp_path: Path) -> None:
    secrets = tmp_path / "secrets.env"
    secrets.write_text(f"{TOKEN_STORE_KEY}=file-key\n", encoding="utf-8")
    loader = SecretsLoader(secrets_file=secrets, environ={})

    assert loader.get_token_store_key() == "file-key"


def test_missing_token_store_key_names_the_token_store_reference(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path)

    with pytest.raises(MissingSecretError) as excinfo:
        _ = loader.token_store_key

    assert excinfo.value.reference == TOKEN_STORE_KEY
    assert TOKEN_STORE_KEY in str(excinfo.value)
