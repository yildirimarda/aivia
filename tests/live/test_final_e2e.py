"""Final live end-to-end test.

Produces one image, video, voice and song for every configured language under a
strict budget cap, asserts every produced asset is pinned, and runs disclosure
checks only when the publishing module is installed.

Runs only with ``--live``, ``AIVA_LIVE_BUDGET_USD`` and ``AIVA_LIVE_FINAL_E2E``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from decimal import Decimal

import pytest

from aivia.core.config import resolve_config
from aivia.core.platforms import load_platforms
from aivia.media.toolkit import MediaToolkit

_LIVE_FLAG = "AIVA_LIVE_FINAL_E2E"
_TRUE = {"1", "true", "yes", "on"}
_KINDS = ("image", "video", "speech", "music")


def _flag_enabled() -> bool:
    return os.environ.get(_LIVE_FLAG, "").strip().lower() in _TRUE


def _publishing_installed() -> bool:
    """Return whether the optional publishing module package can be located."""
    return importlib.util.find_spec("aivia.publish") is not None


@pytest.fixture
def live_final_e2e() -> None:
    """Skip unless the final live e2e flag is enabled."""
    if not _flag_enabled():
        pytest.skip(f"final live e2e requires {_LIVE_FLAG}=1")


async def _produce_all(
    toolkit: MediaToolkit, languages: list[str]
) -> tuple[float, int]:
    """Produce one image, video, voice and song per language.

    Returns the total spend and the number of media results produced.
    """
    total_cost = 0.0
    produced = 0
    for language in languages:
        for kind in _KINDS:
            if kind == "speech":
                result = await toolkit.speech(
                    f"Sound check in {language}.", language=language
                )
            else:
                result = await toolkit.generate(kind, f"A {kind} for {language}.")
            total_cost += float(result.cost_usd)
            produced += 1
            assert result.assets, f"{kind} for {language} produced no assets"
            for asset in result.assets:
                assert asset.pinned, f"{kind} asset for {language} is not pinned"
    return total_cost, produced


@pytest.mark.live
def test_final_e2e_one_media_per_language_under_budget(
    live_final_e2e: None,
    live_budget_usd: Decimal,
) -> None:
    """One image, video, voice and song per language within the live budget cap."""
    resolved = resolve_config()
    config = resolved.app_config
    languages_section = getattr(config, "languages", None)
    if languages_section is None:
        pytest.fail("no languages section is configured")
    languages = list(languages_section.languages.keys())
    assert languages, "no languages are configured"

    toolkit = MediaToolkit.from_config(config)
    try:
        total_cost, produced = asyncio.run(_produce_all(toolkit, languages))
        expected = len(languages) * len(_KINDS)
        assert produced == expected, (
            f"expected {expected} media results, produced {produced}"
        )
        assert total_cost <= float(live_budget_usd), (
            f"final e2e spend ${total_cost:.6f} exceeds the "
            f"${live_budget_usd} budget cap"
        )
        if _publishing_installed():
            _assert_disclosure_checks()
    finally:
        toolkit.close()


def _assert_disclosure_checks() -> None:
    from aivia.publish.docs import _disclosure

    for key, platform in load_platforms().items():
        if platform.disclosure.required:
            assert platform.disclosure.text, (
                f"platform {key!r} requires disclosure but has no disclosure text"
            )
            assert _disclosure(platform) == "yes", (
                f"platform {key!r} disclosure check failed"
            )
