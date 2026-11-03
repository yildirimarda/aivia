"""The keyword topic mode: discover topics for explicit profile keywords.

A keyword profile does not need a brief or a derived query expansion. Its
explicit keywords are the search queries, and a candidate is kept when it
matches at least one of them. The mode reuses the niche mode's concurrent
discovery, deduplication and tagging, but replaces the share-of-terms gate with
an any-keyword gate so a profile with several keywords does not reject a topic
that matches only one of them.
"""

from __future__ import annotations

from typing import Any

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.research.modes.base import register_mode
from aivia.research.modes.niche import (
    DEFAULT_MIN_RELEVANCE,
    NicheMode,
    RelevanceGate,
    _mapping_value,
    _search_text,
    _term_matches,
    normalize_terms,
)

__all__ = [
    "Keyword",
    "KeywordGate",
    "KeywordMode",
    "KeywordModeProvider",
    "KeywordSource",
    "KeywordSourceMode",
    "KeywordTopicMode",
    "Keywords",
    "KeywordsMode",
    "KeywordsModeProvider",
    "KeywordsSourceMode",
    "KeywordsTopicMode",
    "WikipediaKeywordsMode",
]


class KeywordGate(RelevanceGate):
    """A relevance gate that accepts a candidate matching any keyword.

    The inherited gate scores a candidate by the share of required terms it
    matches. For explicit keywords that is too strict: a topic about one of
    three keywords is still on theme. This gate returns one when any required
    term matches and zero otherwise, while keeping the same exclude-term and
    injected-scorer behaviour.
    """

    def term_score(self, candidate: TopicCandidate) -> float:
        """Return one when *candidate* matches any required term."""
        text = _search_text(candidate)
        if any(_term_matches(text, term) for term in self.exclude_terms):
            return 0.0
        terms = self.required_terms
        if not terms:
            return 1.0
        return 1.0 if any(_term_matches(text, term) for term in terms) else 0.0


class KeywordsMode(NicheMode):
    """Discover topics for explicit keywords, one query per keyword.

    ``keywords`` may be a string or an iterable. When omitted, the profile's
    keywords are used, then the derivation's explicit keywords. ``queries``
    overrides the search queries without changing the keywords the gate uses.
    A custom ``relevance`` scorer or ``gate`` replaces the any-keyword gate.
    """

    name = "keywords"

    def __init__(
        self,
        source: Any = None,
        *,
        keywords: Any = None,
        keyword: Any = None,
        terms: Any = None,
        profile: Any = None,
        derivation: Any = None,
        brief_derivation: Any = None,
        queries: Any = None,
        search_queries: Any = None,
        include_terms: Any = None,
        include: Any = None,
        min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
        relevance_threshold: float | None = None,
        threshold: float | None = None,
        relevance: Any = None,
        relevance_scorer: Any = None,
        scorer: Any = None,
        gate: RelevanceGate | None = None,
        **options: Any,
    ) -> None:
        if keyword is not None:
            if keywords is not None:
                raise ConfigError("use either keywords or keyword, not both")
            keywords = keyword
        if terms is not None:
            if keywords is not None:
                raise ConfigError("use either keywords or terms, not both")
            keywords = terms
        if brief_derivation is not None:
            if derivation is not None:
                raise ConfigError("use either derivation or brief_derivation, not both")
            derivation = brief_derivation
        if search_queries is not None:
            if queries is not None:
                raise ConfigError("use either queries or search_queries, not both")
            queries = search_queries
        if include is not None:
            if include_terms is not None:
                raise ConfigError("use either include_terms or include, not both")
            include_terms = include

        if keywords is None:
            if profile is not None:
                keywords = _mapping_value(profile, "keywords", default=())
            if not keywords and derivation is not None:
                keywords = _mapping_value(derivation, "keywords", default=())
        normalized = normalize_terms(keywords, label="keyword")
        if queries is None:
            queries = normalized
        if include_terms is None:
            include_terms = normalized

        effective_min = min_relevance
        if relevance_threshold is not None:
            effective_min = relevance_threshold
        if threshold is not None:
            effective_min = threshold

        if (
            gate is None
            and relevance is None
            and relevance_scorer is None
            and scorer is None
        ):
            gate = KeywordGate(
                include_terms=include_terms,
                exclude_terms=options.get("exclude_terms", options.get("exclude")),
                queries=queries,
                keywords=normalized,
                min_relevance=effective_min,
            )

        self._keyword_terms = normalized
        super().__init__(
            source,
            profile=profile,
            derivation=derivation,
            queries=queries,
            keywords=[],
            include_terms=include_terms,
            min_relevance=effective_min,
            relevance=relevance,
            relevance_scorer=relevance_scorer,
            scorer=scorer,
            gate=gate,
            **options,
        )

    @property
    def keywords(self) -> list[str]:
        """The explicit keywords the gate matches, without expanding queries."""
        return list(self._keyword_terms)

    async def discover(self, request: Any) -> Any:
        """Discover keyword candidates and expose the keywords in metadata."""
        response = await super().discover(request)
        metadata = dict(response.metadata)
        metadata["mode"] = self.name
        metadata["keywords"] = list(self.keywords)
        metadata["include_terms"] = list(self.include_terms)
        metadata["skipped_keywords"] = metadata.get("skipped_relevance", 0)
        return response.model_copy(update={"metadata": metadata})


Keyword = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordMode = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordTopicMode = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordSourceMode = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

Keywords = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordsTopicMode = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordModeProvider = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordSource = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordsModeProvider = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

KeywordsSourceMode = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

WikipediaKeywordsMode = KeywordsMode
"""Alias for :class:`KeywordsMode`."""

register_mode("keywords", KeywordsMode)
