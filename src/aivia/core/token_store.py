"""Encrypted, auto-refreshing token store for publish credentials.

Publish tokens are sensitive credentials: they are encrypted at rest with the
key from :meth:`aivia.core.secrets.SecretsLoader.get_token_store_key`, never
appear in log messages or error text, and are refreshed automatically through a
caller-supplied callback when they expire or are absent.
"""

from __future__ import annotations

import base64
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from aivia.core.errors import ConfigError
from aivia.core.redact import REDACTED, redact, register_secret
from aivia.core.secrets import SecretsLoader

__all__ = [
    "StoredToken",
    "TokenError",
    "TokenRefreshError",
    "TokenStore",
]


def _derive_key(secret: str) -> bytes:
    """Derive a Fernet-compatible key from the token store secret."""
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _normalise(value: str) -> str:
    return str(value).strip().lower()


class TokenError(ConfigError):
    """A publish token is missing, invalid or cannot be decrypted."""

    code = "token"


class TokenRefreshError(TokenError):
    """The refresh callback failed for a platform account."""


@dataclass(frozen=True, slots=True)
class StoredToken:
    """A single publish token, encrypted at rest.

    The plaintext value is never part of the string representation and the
    ciphertext is the only thing persisted, so neither leaks into logs.
    """

    platform: str
    account: str
    encrypted: str
    expires_at: float | None = None
    refreshed_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (
            "StoredToken(platform="
            f"{self.platform!r}, account={self.account!r}, "
            f"expires_at={self.expires_at!r}, refreshed_at={self.refreshed_at!r})"
        )

    def is_expired(self, *, clock: Callable[[], float] = time.time) -> bool:
        """Whether the stored token is past its expiry."""
        if self.expires_at is None:
            return False
        return clock() >= self.expires_at


class TokenStore:
    """Encrypt, store and refresh publish tokens.

    The encryption key is derived from the token store key exposed by *secrets*.
    Tokens are encrypted with Fernet at rest. The plaintext value is never
    logged: it is redacted from messages and details, and each stored value is
    registered as a secret so the process-wide redaction helper masks it too.

    *refresh* is an optional async callback ``refresh(platform, account) ->
    str`` that obtains a fresh token. When :meth:`get` finds no stored token or
    an expired one, it calls the refresh callback, encrypts the result and
    returns it. Without a callback, a missing or expired token raises
    :class:`TokenError`.
    """

    def __init__(
        self,
        secrets: SecretsLoader,
        *,
        refresh: Callable[[str, str], Awaitable[str]] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secrets = secrets
        self._refresh = refresh
        self._clock = clock
        self._fernet = Fernet(_derive_key(secrets.get_token_store_key()))
        self._tokens: dict[tuple[str, str], StoredToken] = {}

    async def get(self, platform: str, account: str) -> str:
        """Return the decrypted token for *platform*/*account*.

        Refreshes automatically when no token is stored or the stored token has
        expired. The returned value is registered as a secret so it is redacted
        from any later log output.
        """
        key = (_normalise(platform), _normalise(account))
        stored = self._tokens.get(key)
        if stored is not None and not stored.is_expired(clock=self._clock):
            token = self._decrypt(stored.encrypted)
            register_secret(token)
            return token
        if self._refresh is None:
            raise TokenError(
                "no publish token stored and no refresh callback",
                details={"platform": key[0], "account": key[1]},
            )
        try:
            token = await self._refresh(key[0], key[1])
        except TokenError:
            raise
        except Exception as exc:
            raise TokenRefreshError(
                "failed to refresh publish token",
                details={"platform": key[0], "account": key[1]},
            ) from exc
        self.store(key[0], key[1], token)
        return token

    def store(
        self,
        platform: str,
        account: str,
        token: str,
        *,
        expires_in_seconds: float | None = None,
    ) -> None:
        """Store *token* for *platform*/*account*, encrypted at rest.

        The plaintext is registered as a secret so it is redacted from logs and
        error messages. *expires_in_seconds*, when given, makes :meth:`get`
        refresh automatically once the deadline passes.
        """
        if not token or not str(token).strip():
            raise TokenError(
                "publish token must not be blank",
                details={"platform": platform, "account": account},
            )
        text = str(token).strip()
        expires_at: float | None = None
        if expires_in_seconds is not None:
            expires_at = self._clock() + float(expires_in_seconds)
        self._tokens[(_normalise(platform), _normalise(account))] = StoredToken(
            platform=_normalise(platform),
            account=_normalise(account),
            encrypted=self._encrypt(text),
            expires_at=expires_at,
            refreshed_at=self._clock(),
        )
        register_secret(text)

    def has(self, platform: str, account: str) -> bool:
        """Whether a valid (non-expired) token is stored."""
        stored = self._tokens.get((_normalise(platform), _normalise(account)))
        return stored is not None and not stored.is_expired(clock=self._clock)

    def clear(self, platform: str, account: str) -> None:
        """Remove any stored token for *platform*/*account*."""
        self._tokens.pop((_normalise(platform), _normalise(account)), None)

    def entries(self) -> Mapping[tuple[str, str], StoredToken]:
        """Return a copy of the stored (encrypted) tokens."""
        return dict(self._tokens)

    def _encrypt(self, text: str) -> str:
        return self._fernet.encrypt(text.encode("utf-8")).decode("ascii")

    def _decrypt(self, encrypted: str) -> str:
        try:
            return self._fernet.decrypt(encrypted.encode("ascii")).decode("utf-8")
        except InvalidToken as exc:
            raise TokenError(
                "stored publish token could not be decrypted",
                details={"reason": "invalid_token"},
            ) from exc

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tokens={len(self._tokens)}, "
            f"refresh={self._refresh is not None})"
        )

    def __str__(self) -> str:
        return redact(self.__repr__())


def _never_log(value: Any) -> str:
    """Return a placeholder so a token value is never rendered in a message."""
    return REDACTED
