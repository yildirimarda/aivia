"""Encrypted, auto-refreshing publish tokens that are never logged.

Covers the token store (encryption at rest with the secrets-loader key,
automatic refresh, redaction of token values) and the per-platform error
taxonomy that normalizes platform HTTP errors into the shared taxonomy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator

import pytest

from aivia.core.errors import (
    AuthError,
    InvalidRequestError,
    QuotaError,
    RateLimitError,
    RetryableError,
    TimeoutError,
)
from aivia.core.publish_errors import (
    PLATFORM_ERROR_MAP,
    classify_error,
    default_status_map,
    error_for_result,
)
from aivia.core.redact import clear_secrets, redact
from aivia.core.secrets import SecretsLoader
from aivia.core.token_store import (
    StoredToken,
    TokenError,
    TokenRefreshError,
    TokenStore,
)


def _loader(
    *, key: str = "store-key-1", environ: dict[str, str] | None = None
) -> SecretsLoader:
    env = {"XDG_CONFIG_HOME": "/tmp/opencode/xdg"}
    if environ is not None:
        env.update(environ)
    env["AIVA_TOKEN_STORE_KEY"] = key
    return SecretsLoader(environ=env)


def _store(
    *,
    key: str = "store-key-1",
    clock: Callable[[], float] = time.time,
    refresh: Callable[[str, str], Awaitable[str]] | None = None,
) -> TokenStore:
    return TokenStore(_loader(key=key), refresh=refresh, clock=clock)


def _clear_secrets() -> None:
    clear_secrets()


@pytest.fixture(autouse=True)
def _reset_secrets() -> Iterator[None]:
    clear_secrets()
    yield
    clear_secrets()
    PLATFORM_ERROR_MAP.clear()


# ---------------------------------------------------------------- round trip


def test_store_round_trips_a_token_encrypted_at_rest() -> None:
    store = _store()
    store.store("youtube", "acct-1", "plain-secret-token", expires_in_seconds=3600)

    entries = store.entries()
    assert len(entries) == 1
    stored = entries[("youtube", "acct-1")]
    assert isinstance(stored, StoredToken)
    assert stored.encrypted != "plain-secret-token"
    assert asyncio.run(store.get("youtube", "acct-1")) == "plain-secret-token"


def test_store_requires_non_blank_token() -> None:
    store = _store()
    with pytest.raises(TokenError, match="must not be blank"):
        store.store("youtube", "acct-1", "   ")
    with pytest.raises(TokenError, match="must not be blank"):
        store.store("youtube", "acct-1", "")


def test_get_missing_token_without_refresh_raises() -> None:
    store = _store()
    with pytest.raises(TokenError) as excinfo:
        asyncio.run(store.get("youtube", "acct-1"))
    assert excinfo.value.details == {"platform": "youtube", "account": "acct-1"}


def test_get_expired_token_raises_without_refresh() -> None:
    now = [1000.0]

    def clock() -> float:
        return now[0]

    store = _store(clock=clock)
    store.store("youtube", "acct-1", "token", expires_in_seconds=10)
    now[0] = 2000.0  # past the deadline
    with pytest.raises(TokenError):
        asyncio.run(store.get("youtube", "acct-1"))


# ------------------------------------------------------------------ auto refresh


def test_expired_token_triggers_refresh() -> None:
    refreshed: list[str] = []

    async def refresh(platform: str, account: str) -> str:
        refreshed.append(f"{platform}/{account}")
        return "fresh-token"

    clock = time.time
    store = _store(clock=clock, refresh=refresh)
    store.store("youtube", "acct-1", "stale-token", expires_in_seconds=10)
    store._clock = lambda: clock() + 100

    token = asyncio.run(store.get("youtube", "acct-1"))
    assert token == "fresh-token"
    assert refreshed == ["youtube/acct-1"]
    # the refreshed token is stored and reused without another refresh call
    assert asyncio.run(store.get("youtube", "acct-1")) == "fresh-token"
    assert refreshed == ["youtube/acct-1"]


def test_missing_token_triggers_refresh() -> None:
    calls: list[tuple[str, str]] = []

    async def refresh(platform: str, account: str) -> str:
        calls.append((platform, account))
        return f"token-for-{account}"

    store = _store(refresh=refresh)
    assert asyncio.run(store.get("instagram", "acct-2")) == "token-for-acct-2"
    assert calls == [("instagram", "acct-2")]
    # subsequent gets reuse the stored token
    assert asyncio.run(store.get("instagram", "acct-2")) == "token-for-acct-2"
    assert calls == [("instagram", "acct-2")]


def test_refresh_failure_raises_token_refresh_error() -> None:
    async def refresh(platform: str, account: str) -> str:
        raise RuntimeError("refresh failed")

    store = _store(refresh=refresh)
    with pytest.raises(TokenRefreshError) as excinfo:
        asyncio.run(store.get("youtube", "acct-1"))
    assert excinfo.value.details == {"platform": "youtube", "account": "acct-1"}
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ------------------------------------------------------------------- never log


def test_token_never_appears_in_store_repr() -> None:
    store = _store()
    store.store("youtube", "acct-1", "test-token")
    text = repr(store) + str(store)
    assert "test-token" not in text
    assert "super-secret" not in text


def test_stored_token_never_reveals_plaintext() -> None:
    store = _store()
    store.store("youtube", "acct-1", "test-token")
    stored = store.entries()[("youtube", "acct-1")]
    assert "test-token" not in repr(stored)
    assert stored.encrypted != "test-token"


def test_registered_token_is_redacted() -> None:
    store = _store()
    store.store("youtube", "acct-1", "test-token")
    assert "test-token" not in redact("token=test-token")
    assert "[REDACTED]" in redact("token=test-token")


# ----------------------------------------------------------- key isolation


def test_different_keys_produce_different_ciphertext() -> None:
    store_a = _store(key="key-a")
    store_b = _store(key="key-b")
    store_a.store("youtube", "acct-1", "same-token")
    store_b.store("youtube", "acct-1", "same-token")
    entries_a = store_a.entries()[("youtube", "acct-1")]
    entries_b = store_b.entries()[("youtube", "acct-1")]
    assert entries_a.encrypted != entries_b.encrypted
    # a store built with the wrong key cannot decrypt the other store's token
    assert asyncio.run(store_b.get("youtube", "acct-1")) == "same-token"


def test_store_key_comes_from_secrets_loader() -> None:
    store_a = _store(key="loader-key")
    store_a.store("youtube", "acct-1", "same-token")
    ciphertext = store_a.entries()[("youtube", "acct-1")].encrypted

    # a store built from a loader with the same key decrypts the token
    store_b = TokenStore(_loader(key="loader-key"), clock=time.time)
    store_b._tokens[("youtube", "acct-1")] = StoredToken(
        platform="youtube", account="acct-1", encrypted=ciphertext
    )
    assert asyncio.run(store_b.get("youtube", "acct-1")) == "same-token"

    # a store built from a loader with a different key cannot decrypt it
    store_c = TokenStore(_loader(key="other-key"), clock=time.time)
    store_c._tokens[("youtube", "acct-1")] = StoredToken(
        platform="youtube", account="acct-1", encrypted=ciphertext
    )
    with pytest.raises(TokenError):
        asyncio.run(store_c.get("youtube", "acct-1"))


# ------------------------------------------------------- has / clear / entries


def test_has_clear_and_entries() -> None:
    store = _store()
    assert not store.has("youtube", "acct-1")
    assert store.entries() == {}
    store.store("youtube", "acct-1", "token-1")
    assert store.has("youtube", "acct-1")
    store.clear("youtube", "acct-1")
    assert not store.has("youtube", "acct-1")
    assert store.entries() == {}


def test_normalises_platform_and_account() -> None:
    store = _store()
    store.store("YouTube", " ACCT-1 ", "token")
    assert store.has("youtube", "acct-1")
    assert asyncio.run(store.get("YouTube", "ACCT-1")) == "token"


# ---------------------------------------------------- error taxonomy


def test_default_status_map_covers_shared_taxonomy() -> None:
    mapping = default_status_map()
    assert mapping[401] is AuthError
    assert mapping[403] is AuthError
    assert mapping[429] is RateLimitError
    assert mapping[400] is InvalidRequestError
    assert mapping[422] is InvalidRequestError
    assert mapping[500] is RetryableError
    assert mapping[503] is RetryableError
    assert mapping[504] is TimeoutError
    assert mapping[402] is QuotaError


def test_classify_error_maps_http_status_to_shared_taxonomy() -> None:
    assert isinstance(classify_error("youtube", 401), AuthError)
    assert isinstance(classify_error("youtube", 403), AuthError)
    assert isinstance(classify_error("youtube", 429), RateLimitError)
    assert isinstance(classify_error("youtube", 400), InvalidRequestError)
    assert isinstance(classify_error("youtube", 500), RetryableError)
    assert isinstance(classify_error("youtube", 504), TimeoutError)
    assert isinstance(classify_error("youtube", 402), QuotaError)


def test_classify_error_is_scoped_with_platform() -> None:
    error = classify_error("YouTube", 401, body={"error": "invalid credentials"})
    assert isinstance(error, AuthError)
    assert error.details["platform"] == "youtube"
    assert error.details["status"] == 401
    assert "invalid credentials" in str(error)


def test_classify_error_unknown_status_defaults_to_invalid_request() -> None:
    error = classify_error("tiktok", 418)
    assert isinstance(error, InvalidRequestError)
    assert error.details["platform"] == "tiktok"
    assert error.details["status"] == 418


def test_classify_error_unknown_transient_status_is_retryable() -> None:
    error = classify_error("x", 599, body="rate limited, try again later")
    assert isinstance(error, RetryableError)


def test_platform_error_map_overrides_default() -> None:
    PLATFORM_ERROR_MAP["instagram"] = {400: RateLimitError}
    error = classify_error("instagram", 400)
    assert isinstance(error, RateLimitError)
    # other statuses still use the default
    assert isinstance(classify_error("instagram", 401), AuthError)


def test_error_for_result_success_returns_none() -> None:
    assert error_for_result("youtube", 200) is None
    assert error_for_result("youtube", 201) is None


def test_error_for_result_failure_returns_shared_taxonomy_error() -> None:
    error = error_for_result("youtube", 401, message="token expired")
    assert isinstance(error, AuthError)
    assert error.details["platform"] == "youtube"
    assert error.details["status"] == 401
    assert "token expired" in str(error)


def test_classify_error_permanent_errors_are_not_retryable() -> None:
    assert classify_error("youtube", 401).permanent is True
    assert classify_error("youtube", 400).permanent is True
    assert classify_error("youtube", 429).permanent is False
    assert classify_error("youtube", 500).permanent is False
