"""Format to account compatibility.

A format whose output is published to a platform account must produce a media
type that platform supports. This module matrix-tests every packaged format
against every configured publish account and asserts the compatibility rule.
"""

from __future__ import annotations

from aivia.core.config.formats import FormatRecipeConfig, OutputHandler
from aivia.core.config.schedule import PublishConfig
from aivia.core.platforms import Platform, PlatformCatalog, load_platforms
from aivia.formats import load_packaged_formats
from aivia.publish.accounts import AccountMap


def _publish_config(**accounts: object) -> PublishConfig:
    return PublishConfig.from_mapping({"accounts": dict(accounts)})


def _format_compatible(recipe: FormatRecipeConfig, platform: Platform) -> bool:
    """A format is compatible with a platform when its output is supported."""
    return recipe.output_type in platform.media_types


def _configured_accounts() -> tuple[PublishConfig, AccountMap]:
    publish = _publish_config(
        **{
            "youtube/en": {"credential_refs": ["YOUTUBE_TOKEN"]},
            "instagram/en": {"credential_refs": ["INSTAGRAM_TOKEN"]},
            "tiktok/en": {"credential_refs": ["TIKTOK_TOKEN"]},
            "x/en": {"credential_refs": ["X_TOKEN"]},
            "telegram/en": {"credential_refs": ["TELEGRAM_TOKEN"]},
        }
    )
    return publish, AccountMap(publish)


def _platforms() -> PlatformCatalog:
    return load_platforms()


def test_packaged_formats_and_configured_accounts_load() -> None:
    formats = load_packaged_formats()
    assert formats.names()
    _publish, accounts = _configured_accounts()
    assert accounts.names()


def test_every_configured_account_resolves_to_a_known_platform() -> None:
    _publish, accounts = _configured_accounts()
    platforms = _platforms()
    for key in accounts.names():
        platform, _, _ = key.partition("/")
        platforms.require(platform)


def test_publish_formats_are_compatible_with_every_configured_account() -> None:
    formats = load_packaged_formats()
    _publish, accounts = _configured_accounts()
    platforms = _platforms()
    for format_name in formats.names():
        recipe = formats.get(format_name)
        if recipe.handler != OutputHandler.PUBLISH:
            continue
        for key in accounts.names():
            platform = platforms.require(key.partition("/")[0])
            assert recipe.output_type in platform.media_types, (
                f"format {format_name!r} outputs {recipe.output_type!r} which "
                f"{platform.id!r} does not support "
                f"(supported: {', '.join(platform.media_types) or 'none'})"
            )


def test_compatibility_predicate_matches_media_type_support() -> None:
    formats = load_packaged_formats()
    platforms = _platforms()
    image_post = formats.get("image_post")
    audio_only = formats.get("audio_only")
    song_video = formats.get("song_video")
    instagram = platforms.require("instagram")
    youtube = platforms.require("youtube")
    telegram = platforms.require("telegram")
    # image is supported by instagram and telegram, not by youtube (video only)
    assert _format_compatible(image_post, instagram)
    assert _format_compatible(image_post, telegram)
    assert not _format_compatible(image_post, youtube)
    # audio is supported by telegram only among these
    assert _format_compatible(audio_only, telegram)
    assert not _format_compatible(audio_only, youtube)
    # video is supported by all of them
    assert _format_compatible(song_video, instagram)
    assert _format_compatible(song_video, youtube)
    assert _format_compatible(song_video, telegram)


def test_every_format_is_compatible_with_at_least_one_configured_account() -> None:
    formats = load_packaged_formats()
    _publish, accounts = _configured_accounts()
    platforms = _platforms()
    for format_name in formats.names():
        recipe = formats.get(format_name)
        assert any(
            _format_compatible(recipe, platforms.require(key.partition("/")[0]))
            for key in accounts.names()
        ), f"format {format_name!r} cannot be published to any configured account"


def test_every_configured_account_has_at_least_one_compatible_format() -> None:
    formats = load_packaged_formats()
    _publish, accounts = _configured_accounts()
    platforms = _platforms()
    for key in accounts.names():
        platform = platforms.require(key.partition("/")[0])
        assert any(
            _format_compatible(formats.get(format_name), platform)
            for format_name in formats.names()
        ), f"account {key!r} has no compatible format on {platform.id!r}"
