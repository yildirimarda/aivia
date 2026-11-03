"""Research module configuration.

The model lives in :mod:`aivia.core.config.research` so the core can describe
the section without importing this module. This module re-exports it together
with the explicit registration helper the application calls once research is
installed.
"""

from aivia.core.config.research import (
    DEFAULT_DEDUPE_WINDOW,
    DEFAULT_DEDUPE_WINDOW_DAYS,
    DEFAULT_EMBEDDING_THRESHOLD,
    RESEARCH_SECTION_KEY,
    DedupeWindow,
    ResearchConfig,
    ResearchSettings,
    ResearchSettingsConfig,
    register_research_config,
    register_research_section,
    research_config,
)

__all__ = [
    "DEFAULT_DEDUPE_WINDOW",
    "DEFAULT_DEDUPE_WINDOW_DAYS",
    "DEFAULT_EMBEDDING_THRESHOLD",
    "RESEARCH_SECTION_KEY",
    "DedupeWindow",
    "ResearchConfig",
    "ResearchSettings",
    "ResearchSettingsConfig",
    "register_research_config",
    "register_research_section",
    "research_config",
]
