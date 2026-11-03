"""The feeds topic mode: discover topics from feed sources only.

A feeds profile is driven by RSS or Atom feeds. Other configured sources are
ignored, even when the profile lists them, because a feed run must not silently
mix in a wiki, community or search source. Feed URLs may be supplied directly,
through a profile's source options, or through the feed provider's own
configuration.

The mode reuses the niche mode's concurrent discovery, deduplication and
tagging. It disables the relevance gate by default: a feed editor has already
chosen the items, so the mode reports them in feed order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aivia.core.config.profile import ContentProfileConfig, _coerce_source_entries
from aivia.core.errors import ConfigError
from aivia.providers.sources import RSSSourceProvider, resolve_source
from aivia.providers.sources.rss import normalize_feeds
from aivia.research.modes.base import register_mode
from aivia.research.modes.niche import NicheMode

__all__ = [
    "FEED_SOURCE_NAMES",
    "Feed",
    "FeedMode",
    "FeedModeProvider",
    "FeedSource",
    "FeedSourceMode",
    "FeedTopicMode",
    "Feeds",
    "FeedsMode",
    "FeedsModeProvider",
    "FeedsSourceMode",
    "FeedsTopicMode",
    "RSSFeeds",
    "RSSFeedsMode",
    "RSSMode",
    "RssFeeds",
    "RssMode",
]

FEED_SOURCE_NAMES = frozenset({"rss", "rss-source", "feed", "feeds", "atom"})
"""Source names that identify a feed adapter."""

_FEED_URL_KEYS = ("feeds", "feed", "feed_urls", "urls")


def _name_is_feed(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    text = name.strip().lower()
    if text in FEED_SOURCE_NAMES:
        return True
    return "rss" in text or "feed" in text or "atom" in text


def _is_feed_source(spec: Any) -> bool:
    """Whether *spec* names, wraps or is a feed source."""
    if isinstance(spec, str):
        return _name_is_feed(spec)
    if isinstance(spec, type):
        try:
            if issubclass(spec, RSSSourceProvider):
                return True
        except TypeError:
            pass
        return _name_is_feed(spec.__name__)
    if callable(spec) and not hasattr(spec, "discover"):
        return True
    provider = getattr(spec, "provider", None)
    if provider is not None and provider is not spec and _is_feed_source(provider):
        return True
    name = getattr(spec, "name", None)
    if _name_is_feed(name):
        return True
    if hasattr(spec, "feeds") or hasattr(spec, "feed_urls"):
        return True
    metadata = getattr(spec, "default_metadata", None)
    if metadata is None:
        metadata = getattr(spec, "metadata", None)
    if _name_is_feed(getattr(metadata, "name", None)):
        return True
    return _name_is_feed(type(spec).__name__)


def _profile_parts(profile: Any) -> tuple[list[Any] | None, dict[str, Any]]:
    if profile is None:
        return None, {}
    if isinstance(profile, ContentProfileConfig):
        profile_options = {
            name: entry.model_dump() for name, entry in profile.source_options.items()
        }
        return list(profile.sources), profile_options
    if isinstance(profile, Mapping):
        values = dict(profile)
        options: dict[str, Any] = {}
        for key in ("source_options", "source_configs", "source_settings"):
            value = values.get(key)
            if isinstance(value, Mapping):
                options.update(value)
        sources = values.get(
            "sources",
            values.get("source", values.get("topic_sources")),
        )
        if sources is None:
            return (list(options) if options else None), options
        try:
            names, embedded = _coerce_source_entries(sources)
        except (TypeError, ValueError) as error:
            raise ConfigError(str(error)) from error
        for name, entry in embedded.items():
            current = options.get(name)
            if isinstance(current, Mapping) and isinstance(entry, Mapping):
                options[name] = {**current, **entry}
            else:
                options[name] = entry
        return names, options
    if isinstance(profile, str):
        return [profile], {}
    if isinstance(profile, Sequence) and not isinstance(profile, (bytes, bytearray)):
        return list(profile), {}
    sources = getattr(profile, "sources", None)
    if sources is not None:
        options = getattr(profile, "source_options", {})
        return list(sources), dict(options)
    return None, {}


def _feed_urls_from_options(options: Mapping[str, Any]) -> tuple[str, ...]:
    urls: list[str] = []
    for name, entry in options.items():
        if not _name_is_feed(name):
            continue
        if isinstance(entry, Mapping):
            for key in _FEED_URL_KEYS:
                value = entry.get(key)
                if value is not None:
                    urls.extend(normalize_feeds(value))
        else:
            for key in _FEED_URL_KEYS:
                value = getattr(entry, key, None)
                if value is not None:
                    urls.extend(normalize_feeds(value))
    return tuple(dict.fromkeys(urls))


def _coerce_specs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _spec_label(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    name = getattr(spec, "name", None)
    return str(name) if name else type(spec).__name__


def _instantiate(spec: Any, feeds: tuple[str, ...]) -> Any:
    if isinstance(spec, str):
        spec = resolve_source(spec)
    provider = getattr(spec, "provider", None)
    if provider is not None and not hasattr(spec, "discover"):
        spec = provider
    if not isinstance(spec, type) and not hasattr(spec, "discover"):
        name = getattr(spec, "name", None)
        if isinstance(name, str) and name.strip():
            spec = resolve_source(name)
        elif callable(spec):
            return _instantiate(spec(), feeds)
    if not isinstance(spec, type):
        return spec
    try:
        is_rss = issubclass(spec, RSSSourceProvider)
    except TypeError:
        is_rss = False
    if is_rss and feeds:
        return spec(feeds=feeds)
    try:
        return spec()
    except TypeError:
        return spec


class FeedsMode(NicheMode):
    """Discover topics from feed sources, ignoring every other source.

    ``source`` or ``sources`` may name feed adapters, provider classes or
    instances. Non-feed entries are dropped and reported in metadata. When no
    source is supplied at all, the default RSS provider is used. ``feeds``
    overrides the feed URLs; otherwise they are read from the profile's feed
    source options.
    """

    name = "feeds"

    def __init__(
        self,
        source: Any = None,
        *,
        provider: Any = None,
        source_factory: Any = None,
        sources: Any = None,
        feeds: Any = None,
        feed: Any = None,
        feed_urls: Any = None,
        profile: Any = None,
        **options: Any,
    ) -> None:
        if provider is not None:
            if source is not None:
                raise ConfigError("use either source or provider, not both")
            source = provider
        if source_factory is not None:
            if source is not None:
                raise ConfigError("use either source or source_factory, not both")
            source = source_factory
        if sources is not None:
            if source is not None:
                raise ConfigError("use either source or sources, not both")
            source = sources

        supplied = [value for value in (feeds, feed, feed_urls) if value is not None]
        if len(supplied) > 1:
            raise ConfigError("use only one of feeds, feed or feed_urls")
        explicit_feeds = normalize_feeds(supplied[0] if supplied else None)

        profile_sources, profile_options = _profile_parts(profile)
        if source is not None:
            candidates = _coerce_specs(source)
            explicit_sources = True
        elif profile_sources is not None:
            candidates = list(profile_sources)
            explicit_sources = True
        else:
            candidates = []
            explicit_sources = False

        feed_specs: list[Any] = []
        ignored: list[str] = []
        for spec in candidates:
            if _is_feed_source(spec):
                feed_specs.append(spec)
            else:
                ignored.append(_spec_label(spec))
        for spec in profile_sources or ():
            if _is_feed_source(spec):
                continue
            label = _spec_label(spec)
            if label not in ignored:
                ignored.append(label)

        active_feeds = explicit_feeds or _feed_urls_from_options(profile_options)
        if not feed_specs and (not explicit_sources or active_feeds):
            feed_specs = [RSSSourceProvider]

        instances = [_instantiate(spec, active_feeds) for spec in feed_specs]

        self._feeds = active_feeds
        self._ignored_sources = ignored
        super().__init__(
            sources=instances,
            profile=profile,
            queries=[],
            min_relevance=None,
            **options,
        )
        self._source_specs = instances
        self._source_instances = instances
        if self._feeds:
            self._request_metadata = {
                **self._request_metadata,
                "feeds": list(self._feeds),
            }

    @property
    def feeds(self) -> tuple[str, ...]:
        """The feed URLs this mode uses, in order."""
        return self._feeds

    @property
    def feed_urls(self) -> tuple[str, ...]:
        """Alias for :attr:`feeds`."""
        return self._feeds

    @property
    def ignored_sources(self) -> list[str]:
        """The non-feed sources this mode dropped."""
        return list(self._ignored_sources)

    async def discover(self, request: Any) -> Any:
        """Discover feed candidates and expose the feed context in metadata."""
        response = await super().discover(request)
        metadata = dict(response.metadata)
        metadata["mode"] = self.name
        metadata["feeds"] = list(self._feeds)
        metadata["ignored_sources"] = list(self._ignored_sources)
        metadata["min_relevance"] = None
        metadata["skipped_relevance"] = 0
        metadata["skipped_irrelevant"] = 0
        metadata["skipped_off_topic"] = 0
        return response.model_copy(update={"metadata": metadata})


Feed = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedTopicMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedSource = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedSourceMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedModeProvider = FeedsMode
"""Alias for :class:`FeedsMode`."""

Feeds = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedsTopicMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedsModeProvider = FeedsMode
"""Alias for :class:`FeedsMode`."""

FeedsSourceMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

RSSFeeds = FeedsMode
"""Alias for :class:`FeedsMode`."""

RSSFeedsMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

RSSMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

RssFeeds = FeedsMode
"""Alias for :class:`FeedsMode`."""

RssMode = FeedsMode
"""Alias for :class:`FeedsMode`."""

register_mode("feeds", FeedsMode)
