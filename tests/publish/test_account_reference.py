"""Reference accounts as style sources.

A reference account is a configured publish account whose top posts are read
through a read-capable API and merged into a style spec. Reading an account
requires the ``read`` scope; when the account does not hold it a clear error is
raised. API responses are cached so repeating a lookup does not refetch.
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from aivia.core.config.schedule import PublishConfig
from aivia.core.errors import ConfigError
from aivia.publish.account_reference import (
    CAPABILITY_READ,
    AccountReference,
    MissingScopeError,
    ReferenceAccount,
    ReferenceAccountSource,
    TopPost,
    account_reference,
)
from aivia.publish.accounts import AccountMap
from aivia.reference.schema import ReferenceAnalysis, StyleSpec


def _publish(**accounts: object) -> PublishConfig:
    return PublishConfig.from_mapping({"accounts": dict(accounts)})


class FakeSource(ReferenceAccountSource):
    """A read-capable API fake recording every top_posts call."""

    def __init__(
        self, posts_by_account: dict[str, list[TopPost]] | None = None
    ) -> None:
        super().__init__()
        self._posts_by_account: dict[str, list[TopPost]] = dict(posts_by_account or {})
        self.calls: list[tuple[str, int]] = []
        self.fail: Exception | None = None

    async def top_posts(
        self, account: ReferenceAccount, *, limit: int = 10
    ) -> list[TopPost]:
        self.calls.append((account.key, limit))
        if self.fail is not None:
            raise self.fail
        return list(self._posts_by_account.get(account.key, []))

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _posts(*texts: str) -> list[TopPost]:
    return [
        TopPost(post_id=f"post-{index}", text=text, score=90.0 - index * 5)
        for index, text in enumerate(texts, start=1)
    ]


def _fake_source(service: AccountReference) -> FakeSource:
    return cast(FakeSource, service.source)


def _account_reference(platform: str = "x", language: str = "en") -> AccountReference:
    publish = _publish(
        **{
            f"{platform}/{language}": {
                "credential_refs": ["TOKEN"],
                "options": {"scopes": ["read", "publish"]},
            },
            f"{platform}/no-read": {
                "credential_refs": ["TOKEN"],
                "options": {"scopes": ["publish"]},
            },
            f"{platform}/no-scopes": {
                "credential_refs": ["TOKEN"],
            },
        }
    )
    return AccountReference(AccountMap(publish), FakeSource())


# ---------------------------------------------------------------------------
# Happy path: top posts are merged into a style spec
# ---------------------------------------------------------------------------


def test_style_spec_merges_top_posts() -> None:
    service = _account_reference()
    source = _fake_source(service)
    source._posts_by_account["x/en"] = _posts("warm and minimal", "bold typography")

    spec = asyncio.run(service.style_spec("x", "en", limit=5))

    assert isinstance(spec, StyleSpec)
    assert spec.name == "x/en style"
    # source fetched once
    assert source.call_count == 1
    assert source.calls == [("x/en", 5)]
    # every post became a reference in the merged spec
    assert spec.references == ["x/en:post-1", "x/en:post-2"]
    # the style carries text from the posts
    assert spec.prompt_suffix in {"warm and minimal", "bold typography"}


def test_style_spec_uses_provided_name() -> None:
    service = _account_reference()
    _fake_source(service)._posts_by_account["x/en"] = _posts("one post")

    spec = asyncio.run(service.style_spec("x", "en", name="my-style"))

    assert spec.name == "my-style"


def test_empty_posts_return_an_empty_spec() -> None:
    service = _account_reference()
    _fake_source(service)._posts_by_account["x/en"] = []

    spec = asyncio.run(service.style_spec("x", "en"))

    assert isinstance(spec, StyleSpec)
    assert spec.name == "x/en style"
    assert spec.prompt_suffix == ""
    assert spec.references == []
    assert spec.shot_types == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_top_posts_are_cached_and_not_refetched() -> None:
    service = _account_reference()
    source = _fake_source(service)
    source._posts_by_account["x/en"] = _posts("first post")

    first = asyncio.run(service.style_spec("x", "en"))
    second = asyncio.run(service.style_spec("x", "en"))

    assert first == second
    assert source.call_count == 1


def test_different_limits_refetch() -> None:
    service = _account_reference()
    source = _fake_source(service)
    source._posts_by_account["x/en"] = _posts("a", "b")

    asyncio.run(service.style_spec("x", "en", limit=3))
    asyncio.run(service.style_spec("x", "en", limit=5))

    assert source.call_count == 2
    assert source.calls == [("x/en", 3), ("x/en", 5)]


def test_clear_drops_cached_responses() -> None:
    service = _account_reference()
    source = _fake_source(service)
    source._posts_by_account["x/en"] = _posts("cached post")

    asyncio.run(service.style_spec("x", "en"))
    assert source.call_count == 1
    assert len(service.responses) == 1

    service.clear()
    assert len(service.responses) == 0

    asyncio.run(service.style_spec("x", "en"))
    assert source.call_count == 2


def test_responses_expose_cached_pairs() -> None:
    service = _account_reference()
    source = _fake_source(service)
    source._posts_by_account["x/en"] = _posts("cached post")

    asyncio.run(service.style_spec("x", "en"))

    responses = service.responses
    assert len(responses) == 1
    key, posts = responses[0]
    assert key == '["x","en",10]'
    assert len(posts) == 1
    assert posts[0].text == "cached post"


# ---------------------------------------------------------------------------
# Scope errors
# ---------------------------------------------------------------------------


def test_missing_read_scope_raises_a_clear_error() -> None:
    service = _account_reference()

    with pytest.raises(MissingScopeError) as caught:
        asyncio.run(service.style_spec("x", "no-read"))

    error = caught.value
    assert isinstance(error, ConfigError)
    assert "not authorised to read posts" in str(error)
    assert error.code == "account_reference_scope"
    assert error.details["account"] == "x/no-read"
    assert error.details["required_scope"] == CAPABILITY_READ
    assert error.details["scopes"] == ["publish"]
    assert "x/no-read" in str(error)


def test_account_without_any_scopes_raises() -> None:
    service = _account_reference()

    with pytest.raises(MissingScopeError) as caught:
        asyncio.run(service.style_spec("x", "no-scopes"))

    assert caught.value.details["scopes"] == []


def test_unknown_account_raises_config_error() -> None:
    service = _account_reference()

    with pytest.raises(ConfigError) as caught:
        asyncio.run(service.style_spec("tiktok", "en"))

    assert "no publish account configured for tiktok/en" in str(caught.value)


# ---------------------------------------------------------------------------
# Alias and coercion
# ---------------------------------------------------------------------------


def test_account_reference_is_an_alias() -> None:
    assert account_reference is AccountReference


def test_reference_account_has_scope() -> None:
    account = ReferenceAccount(
        platform="x",
        language="en",
        credential_refs=("TOKEN",),
        scopes=("read", "publish"),
    )
    assert account.key == "x/en"
    assert account.has_scope("read")
    assert not account.has_scope("write")


def test_top_post_fields() -> None:
    post = TopPost(post_id="1", text="hello", url="https://example.com/1", score=50.0)
    assert post.post_id == "1"
    assert post.text == "hello"
    assert post.score == 50.0


# ---------------------------------------------------------------------------
# Merge determinism
# ---------------------------------------------------------------------------


def test_merged_spec_is_deterministic_across_runs() -> None:
    service = _account_reference()
    _fake_source(service)._posts_by_account["x/en"] = _posts("alpha", "beta")

    first = asyncio.run(service.style_spec("x", "en"))
    second = asyncio.run(service.style_spec("x", "en"))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_merge_keeps_post_references() -> None:
    service = _account_reference()
    _fake_source(service)._posts_by_account["x/en"] = _posts("one", "two", "three")

    spec = asyncio.run(service.style_spec("x", "en"))

    assert spec.references == ["x/en:post-1", "x/en:post-2", "x/en:post-3"]


# ---------------------------------------------------------------------------
# ReferenceAnalysis coercion (a StyleSpec-like value can be merged directly)
# ---------------------------------------------------------------------------


def test_reference_analysis_coerces_to_analysis() -> None:
    analysis = ReferenceAnalysis(
        reference_id="x/en:post-1",
        detected_text="a post",
        provider="reference-account",
    )
    merged = ReferenceAnalysis.model_validate(analysis.model_dump(mode="json"))
    assert isinstance(merged, ReferenceAnalysis)
    assert merged.detected_text == "a post"
