"""Research engine: select sources, discover, score and verify topics.

The module is optional: the core never imports it, and it consumes the core
topic models and the provider source registry instead of duplicating either.
"""

from aivia.research.sources import (
    DEFAULT_SOURCE_WEIGHT,
    DEFAULT_WEIGHT,
    ProfileSource,
    ResolvedSource,
    ResolvedSources,
    SourceConfig,
    SourceResolution,
    SourceSelection,
    SourceSelector,
    SourceSpec,
    parse_source_specs,
    parse_sources,
    resolve_profile_sources,
    resolve_source_selection,
    resolve_sources,
    select_profile_sources,
    select_sources,
    source_selection,
)

__all__ = [
    "DEFAULT_SOURCE_WEIGHT",
    "DEFAULT_WEIGHT",
    "ProfileSource",
    "ResolvedSource",
    "ResolvedSources",
    "SourceConfig",
    "SourceResolution",
    "SourceSelection",
    "SourceSelector",
    "SourceSpec",
    "parse_source_specs",
    "parse_sources",
    "resolve_profile_sources",
    "resolve_source_selection",
    "resolve_sources",
    "select_profile_sources",
    "select_sources",
    "source_selection",
]
