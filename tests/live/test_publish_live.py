"""Opt-in live publish smoke test to a private test account.

Runs only when ``--live`` is passed, ``AIVA_LIVE_BUDGET_USD`` is set, the
``AIVA_LIVE_PUBLISH`` flag is enabled and the target account is listed in the
``AIVA_LIVE_PUBLISH_ACCOUNTS`` allowlist. Posts a small image to a private
test account so nothing is ever public.

The account allowlist and the flag are read from the environment. Credentials
are resolved through :class:`~aivia.core.secrets.SecretsLoader` from the
reference named by ``AIVA_LIVE_PUBLISH_TOKEN_REF`` (default ``AIVA_X_ACCESS_TOKEN``).
"""

from __future__ import annotations

import asyncio
import io
import os
from decimal import Decimal

import pytest
from PIL import Image

from aivia.core.config.schedule import PublishConfig
from aivia.core.secrets import SecretsLoader
from aivia.providers.protocols.publish import (
    PostVisibility,
    PublishMedia,
    PublishRequest,
)
from aivia.providers.x_publish import XPublisher
from aivia.publish.accounts import AccountMap

_LIVE_PUBLISH_FLAG = "AIVA_LIVE_PUBLISH"
_ALLOWLIST_ENV = "AIVA_LIVE_PUBLISH_ACCOUNTS"
_TOKEN_REF_ENV = "AIVA_LIVE_PUBLISH_TOKEN_REF"
_DEFAULT_TOKEN_REF = "AIVA_X_ACCESS_TOKEN"
_TRUE = {"1", "true", "yes", "on"}


def _flag_enabled() -> bool:
    return os.environ.get(_LIVE_PUBLISH_FLAG, "").strip().lower() in _TRUE


def _allowlist() -> tuple[str, ...]:
    raw = os.environ.get(_ALLOWLIST_ENV, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _token_ref() -> str:
    return (
        os.environ.get(_TOKEN_REF_ENV, _DEFAULT_TOKEN_REF).strip() or _DEFAULT_TOKEN_REF
    )


@pytest.fixture
def live_publish_account() -> str:
    """The first allowlisted account key.

    Skips when live publishing is disabled by the flag or the allowlist is
    empty, so the offline suite never reaches the network.
    """
    if not _flag_enabled():
        pytest.skip(f"live publishing requires {_LIVE_PUBLISH_FLAG}=1")
    allowlist = _allowlist()
    if not allowlist:
        pytest.skip(f"no accounts in the {_ALLOWLIST_ENV} allowlist")
    return allowlist[0]


def _smoke_media() -> PublishMedia:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(12, 34, 56)).save(buffer, format="PNG")
    return PublishMedia(
        data=buffer.getvalue(),
        media_type="image/png",
        filename="smoke.png",
    )


@pytest.mark.live
def test_publish_live_smoke_posts_to_private_account(
    live_publish_account: str,
    live_budget_usd: Decimal,
) -> None:
    """Publish one small image to a private test account within the live budget."""
    assert live_budget_usd > 0, "live publish requires a positive budget"

    platform, separator, language = live_publish_account.partition("/")
    assert separator, f"account key must be platform/language: {live_publish_account!r}"

    publish = PublishConfig.from_mapping(
        {
            "accounts": {
                live_publish_account: {
                    "platform": platform,
                    "language": language,
                    "credential_ref": _token_ref(),
                    "visibility": "private",
                }
            }
        }
    )
    accounts = AccountMap(publish)
    credentials = accounts.resolve(platform, language)

    secrets = SecretsLoader()
    publisher = XPublisher(
        access_token_ref=credentials.credential_refs[0],
        secrets=secrets,
    )

    request = PublishRequest(
        platform=platform,
        account=live_publish_account,
        language=language,
        caption="aivia live publish smoke test",
        visibility=PostVisibility.PRIVATE,
        media=[_smoke_media()],
    )

    result = asyncio.run(publisher.publish(request))

    assert result.is_live is True
    assert result.dry_run is False
    assert result.remote_id
    assert result.url is not None
    assert result.platform == platform
    assert result.account == live_publish_account
    assert result.metadata.get("privacy") == "private"
