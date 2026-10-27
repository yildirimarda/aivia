"""Layered secret loading from the environment, a file and the OS keyring."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from aivia.core.errors import ConfigError

__all__ = [
    "DEFAULT_SECRETS_FILENAME",
    "ENV_PREFIX",
    "TOKEN_STORE_KEY",
    "TOKEN_STORE_KEY_ENV",
    "TOKEN_STORE_KEY_REFERENCE",
    "KeyringBackend",
    "MissingSecretError",
    "SecretsLoader",
    "default_secrets_file",
]

ENV_PREFIX = "AIVA_"
DEFAULT_SECRETS_FILENAME = "secrets.env"

TOKEN_STORE_KEY = "TOKEN_STORE_KEY"
TOKEN_STORE_KEY_REFERENCE = TOKEN_STORE_KEY
TOKEN_STORE_KEY_ENV = f"{ENV_PREFIX}{TOKEN_STORE_KEY}"


class MissingSecretError(ConfigError):
    """A requested secret is not present in any configured source.

    The message and details contain only the reference that was requested,
    never a secret value or the names of unrelated references.
    """

    def __init__(self, reference: str) -> None:
        super().__init__(
            f"missing secret: {reference}",
            details={"reference": reference},
        )
        self.reference = reference


class KeyringBackend(Protocol):
    """The subset of the keyring API the loader needs."""

    def get_password(self, service_name: str, username: str) -> str | None:
        """Return the stored password or ``None`` when it is absent."""


def default_secrets_file(environ: Mapping[str, str] | None = None) -> Path:
    """Return the secrets file path used when none is passed explicitly."""
    env = os.environ if environ is None else environ
    xdg = env.get("XDG_CONFIG_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return root / "aivia" / DEFAULT_SECRETS_FILENAME


class SecretsLoader:
    """Resolve secret references in a fixed, predictable order.

    Lookup order is environment, then the secrets file, then the OS keyring.
    A reference is a name such as ``OPENAI_API_KEY``; environment variables use
    the :data:`ENV_PREFIX` (``AIVA_OPENAI_API_KEY``). Values are never included
    in errors or log messages.
    """

    def __init__(
        self,
        *,
        secrets_file: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        keyring: KeyringBackend | None = None,
        keyring_service: str = "aivia",
    ) -> None:
        self._environ = os.environ if environ is None else environ
        self._secrets_file = (
            Path(secrets_file).expanduser() if secrets_file is not None else None
        )
        self._file_required = self._secrets_file is not None or bool(
            self._environ.get(f"{ENV_PREFIX}SECRETS_FILE")
        )
        self._keyring = keyring
        self._keyring_checked = keyring is not None
        self._keyring_service = keyring_service

    def get(self, reference: str) -> str:
        """Return the value for *reference* or raise :class:`MissingSecretError`."""
        value = self.get_optional(reference)
        if value is None:
            raise MissingSecretError(_normalise_reference(reference))
        return value

    def get_optional(self, reference: str) -> str | None:
        """Return the value for *reference*, or ``None`` when it is missing."""
        normalised = _normalise_reference(reference)
        if not normalised:
            raise ConfigError("secret reference must not be empty")

        value = self._from_environment(reference)
        if value:
            return value

        value = self._from_file(normalised)
        if value:
            return value

        return self._from_keyring(normalised)

    def get_token_store_key(self) -> str:
        """Return the key used to encrypt stored publish tokens."""
        return self.get(TOKEN_STORE_KEY_REFERENCE)

    @property
    def token_store_key(self) -> str:
        """The key used to encrypt stored publish tokens."""
        return self.get_token_store_key()

    def _from_environment(self, reference: str) -> str | None:
        for name in _environment_names(reference):
            value = self._environ.get(name)
            if value:
                return value
        return None

    def _from_file(self, reference: str) -> str | None:
        path = self._resolve_file()
        if not path.exists():
            if self._file_required:
                raise ConfigError(f"secrets file not found: {path}")
            return None
        return _parse_secrets_file(path).get(reference)

    def _resolve_file(self) -> Path:
        if self._secrets_file is not None:
            return self._secrets_file
        configured = self._environ.get(f"{ENV_PREFIX}SECRETS_FILE")
        if configured:
            return Path(configured).expanduser()
        return default_secrets_file(self._environ)

    def _from_keyring(self, reference: str) -> str | None:
        backend = self._get_keyring()
        if backend is None:
            return None
        try:
            value = backend.get_password(self._keyring_service, reference)
        except Exception:
            return None
        return value or None

    def _get_keyring(self) -> KeyringBackend | None:
        if self._keyring_checked:
            return self._keyring
        self._keyring_checked = True
        try:
            module = importlib.import_module("keyring")
        except ImportError:
            return None
        self._keyring = cast(KeyringBackend, module)
        return self._keyring


def _normalise_reference(reference: str) -> str:
    reference = reference.strip()
    if reference.startswith(ENV_PREFIX):
        return reference[len(ENV_PREFIX) :]
    return reference


def _environment_names(reference: str) -> tuple[str, ...]:
    reference = reference.strip()
    if reference.startswith(ENV_PREFIX):
        return (reference,)
    return (f"{ENV_PREFIX}{reference}", reference)


def _parse_secrets_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read secrets file: {path}") from exc

    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ConfigError(f"invalid secrets file line {number}: {path}")
        key, value = line.split("=", 1)
        reference = _normalise_reference(key)
        if not reference:
            raise ConfigError(f"invalid secrets file line {number}: {path}")
        values[reference] = _unquote(value.strip())
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
