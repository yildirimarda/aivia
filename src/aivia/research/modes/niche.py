"""The niche topic mode: expand a brief into queries, discover concurrently.

A niche profile is steered by a brief. The brief is already turned into search
queries, subtopics and include/exclude terms by
:mod:`aivia.research.brief`; this mode takes that derivation, expands the
queries with its subtopics and keywords, asks the configured source for every
query at the same time and merges the answers in a deterministic order.

Discovery is concurrent, but the result is not completion-ordered: candidates
are appended query by query, so the same request and the same source always
produce the same list. Duplicate titles inside one run are dropped.

A candidate then has to pass a relevance gate. By default the gate is
deterministic and local: an exclude term rejects a candidate, and include terms
(or, when there are none, the expanded queries) set the score as the share of
terms the candidate matches. A caller may inject its own synchronous or
asynchronous scorer, which is how a later relevance stage can replace the
local judgement without changing this mode.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from aivia.core.errors import ConfigError
from aivia.core.models.topic import TopicCandidate
from aivia.providers.protocols.source import SourceRequest, SourceResponse
from aivia.providers.sources import WikipediaSourceProvider, resolve_source
from aivia.research.brief import BriefDerivation, BriefOutput
from aivia.research.modes.base import TopicModeBase, register_mode
from aivia.research.modes.otd import normalize_title

__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_QUERIES",
    "DEFAULT_MIN_RELEVANCE",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_QUERY_EXPANSION_LIMIT",
    "DEFAULT_QUERY_LIMIT",
    "DEFAULT_RELEVANCE",
    "DEFAULT_RELEVANCE_FLOOR",
    "DEFAULT_RELEVANCE_THRESHOLD",
    "MAX_CONCURRENCY",
    "Niche",
    "NicheMode",
    "NicheModeProvider",
    "NicheSourceMode",
    "NicheTopicMode",
    "QueryExpansion",
    "RelevanceGate",
    "WikipediaNicheMode",
    "build_query_expansion",
    "candidate_relevance",
    "expand_queries",
    "expand_query",
    "expand_search_queries",
    "filter_candidates",
    "filter_relevant",
    "gate_candidates",
    "is_relevant",
    "normalise_terms",
    "normalize_terms",
    "passes_relevance",
    "query_expansion",
    "relevance_gate",
    "relevance_score",
    "score_candidate",
    "score_relevance",
]

DEFAULT_MIN_RELEVANCE = 0.5
"""The share of include terms a candidate must match by default."""

DEFAULT_RELEVANCE = DEFAULT_MIN_RELEVANCE
"""Alias for :data:`DEFAULT_MIN_RELEVANCE`."""

DEFAULT_RELEVANCE_THRESHOLD = DEFAULT_MIN_RELEVANCE
"""Alias for :data:`DEFAULT_MIN_RELEVANCE`."""

DEFAULT_CONCURRENCY = 8
"""The number of source calls allowed to run at the same time."""

MAX_CONCURRENCY = 64
"""The largest concurrency a caller may configure."""

DEFAULT_MAX_QUERIES = 20
"""The largest number of expanded queries one run uses."""

DEFAULT_QUERY_LIMIT = DEFAULT_MAX_QUERIES
"""Alias for :data:`DEFAULT_MAX_QUERIES`."""

_TERM_SPLIT = re.compile(r"[\n,;]+")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_SCORE_ATTRIBUTES = ("relevance", "score", "value", "relevance_score")
_SCORER_METHODS = ("score", "relevance", "judge")


def _text(value: Any, *, label: str = "term") -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    normalized = unicodedata.normalize("NFKC", value).replace("_", " ")
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_terms(
    value: Any,
    *,
    label: str = "term",
) -> tuple[str, ...]:
    """Return *value* as ordered, de-duplicated, stripped terms."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw: Iterable[Any] = _TERM_SPLIT.split(value)
    elif isinstance(value, Iterable) and not isinstance(
        value, (bytes, bytearray, Mapping)
    ):
        raw = value
    else:
        raise ConfigError(f"{label}s must be a string or an iterable of strings")
    terms: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ConfigError(f"{label}s must contain only strings")
        text = _text(item, label=label)
        if not text:
            continue
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            terms.append(text)
    return tuple(terms)


normalise_terms = normalize_terms
"""Alias for :func:`normalize_terms`."""


def _mapping_value(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, Mapping):
            if name in source:
                return source[name]
        else:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return default


def _validate_threshold(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("min relevance must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError("min relevance must be finite")
    if not 0.0 <= number <= 1.0:
        raise ConfigError("min relevance must be between zero and one")
    return number


def _validate_concurrency(value: Any) -> int:
    if value is None:
        return DEFAULT_CONCURRENCY
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("concurrency must be an integer")
    if value < 1:
        raise ConfigError("concurrency must be at least one")
    if value > MAX_CONCURRENCY:
        raise ConfigError(f"concurrency must not exceed {MAX_CONCURRENCY}")
    return value


def _validate_max_queries(value: Any) -> int:
    if value is None:
        return DEFAULT_MAX_QUERIES
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("max queries must be an integer")
    if value < 1:
        raise ConfigError("max queries must be at least one")
    return value


def _validate_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError("per query limit must be an integer")
    if value < 1:
        raise ConfigError("per query limit must be at least one")
    return value


def _search_text(candidate: TopicCandidate) -> str:
    parts = [
        candidate.title,
        candidate.summary,
        candidate.category,
        *candidate.keywords,
    ]
    text = " ".join(part for part in parts if part)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", normalized)).strip()


def _term_matches(text: str, term: str) -> bool:
    normalized = _NON_WORD.sub(" ", unicodedata.normalize("NFKC", term).casefold())
    needle = _WHITESPACE.sub(" ", normalized).strip()
    return bool(needle) and needle in text


def _clamp_score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ConfigError("relevance score must be a number") from None
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ConfigError("relevance score must be finite")
        return min(1.0, max(0.0, number))
    for attribute in _SCORE_ATTRIBUTES:
        nested = getattr(value, attribute, None)
        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
            return _clamp_score(nested)
    if isinstance(value, Mapping):
        for key in _SCORE_ATTRIBUTES:
            if key in value:
                return _clamp_score(value[key])
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and value
    ):
        return _clamp_score(value[0])
    raise ConfigError("relevance score must be a number")


def _scorer_callable(scorer: Any) -> Callable[[TopicCandidate], Any]:
    for name in _SCORER_METHODS:
        method = getattr(scorer, name, None)
        if callable(method):
            return cast(Callable[[TopicCandidate], Any], method)
    if callable(scorer):
        return cast(Callable[[TopicCandidate], Any], scorer)
    raise ConfigError("relevance scorer must be callable or expose score()")


@dataclass(frozen=True, slots=True)
class QueryExpansion:
    """The queries a niche run uses and the terms its gate applies."""

    queries: tuple[str, ...] = ()
    subtopics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    niche: str = ""
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "queries",
            normalize_terms(self.queries, label="query"),
        )
        object.__setattr__(
            self,
            "subtopics",
            normalize_terms(self.subtopics, label="subtopic"),
        )
        object.__setattr__(
            self,
            "keywords",
            normalize_terms(self.keywords, label="keyword"),
        )
        object.__setattr__(self, "niche", _text(self.niche, label="niche"))
        object.__setattr__(
            self,
            "include_terms",
            normalize_terms(self.include_terms, label="include term"),
        )
        object.__setattr__(
            self,
            "exclude_terms",
            normalize_terms(self.exclude_terms, label="exclude term"),
        )

    @classmethod
    def from_sources(
        cls,
        derivation: Any = None,
        *,
        queries: Any = None,
        subtopics: Any = None,
        keywords: Any = None,
        niche: str | None = None,
        include_terms: Any = None,
        exclude_terms: Any = None,
    ) -> QueryExpansion:
        """Build an expansion from a derivation and explicit overrides."""
        if derivation is not None and not isinstance(
            derivation,
            (BriefDerivation, BriefOutput, Mapping),
        ):
            raise ConfigError(
                "derivation must be a BriefDerivation, BriefOutput or mapping"
            )
        base_queries = _mapping_value(
            derivation,
            "queries",
            "search_queries",
            "search_terms",
            default=(),
        )
        base_subtopics = _mapping_value(
            derivation,
            "subtopics",
            "sub_topics",
            "angles",
            default=(),
        )
        base_keywords = _mapping_value(derivation, "keywords", default=())
        base_niche = _mapping_value(derivation, "niche", default="")
        base_include = _mapping_value(
            derivation,
            "include_terms",
            "include",
            default=(),
        )
        base_exclude = _mapping_value(
            derivation,
            "exclude_terms",
            "exclude",
            default=(),
        )
        return cls(
            queries=base_queries if queries is None else queries,
            subtopics=base_subtopics if subtopics is None else subtopics,
            keywords=base_keywords if keywords is None else keywords,
            niche=str(base_niche or "") if niche is None else niche,
            include_terms=base_include if include_terms is None else include_terms,
            exclude_terms=base_exclude if exclude_terms is None else exclude_terms,
        )

    @property
    def expanded_queries(self) -> tuple[str, ...]:
        """Every query in stable order, without case-insensitive duplicates."""
        combined = [*self.queries, *self.subtopics, *self.keywords]
        if self.niche:
            combined.append(self.niche)
        terms: list[str] = []
        seen: set[str] = set()
        for term in combined:
            key = term.casefold()
            if key not in seen:
                seen.add(key)
                terms.append(term)
        return tuple(terms)

    @property
    def all_queries(self) -> tuple[str, ...]:
        """Alias for :attr:`expanded_queries`."""
        return self.expanded_queries

    @property
    def search_queries(self) -> list[str]:
        """The expanded queries as a list."""
        return list(self.expanded_queries)

    @property
    def required_terms(self) -> tuple[str, ...]:
        """The terms the local relevance gate expects a candidate to match."""
        if self.include_terms:
            return self.include_terms
        return self.expanded_queries

    def __iter__(self) -> Any:
        return iter(self.expanded_queries)

    def __len__(self) -> int:
        return len(self.expanded_queries)

    def __getitem__(self, index: int) -> str:
        return self.expanded_queries[index]


def expand_queries(
    derivation: Any = None,
    *,
    queries: Any = None,
    subtopics: Any = None,
    keywords: Any = None,
    niche: str | None = None,
    include_terms: Any = None,
    exclude_terms: Any = None,
    limit: int | None = None,
) -> list[str]:
    """Return the ordered, de-duplicated queries for a niche run."""
    expansion = QueryExpansion.from_sources(
        derivation,
        queries=queries,
        subtopics=subtopics,
        keywords=keywords,
        niche=niche,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
    )
    result = list(expansion.expanded_queries)
    if limit is not None:
        maximum = _validate_max_queries(limit)
        result = result[:maximum]
    return result


def query_expansion(
    derivation: Any = None,
    **values: Any,
) -> QueryExpansion:
    """Alias for :meth:`QueryExpansion.from_sources`."""
    return QueryExpansion.from_sources(derivation, **values)


build_query_expansion = query_expansion
"""Alias for :func:`query_expansion`."""


class RelevanceGate:
    """A deterministic term gate with an optional injected scorer."""

    def __init__(
        self,
        include_terms: Any = None,
        exclude_terms: Any = None,
        *,
        queries: Any = None,
        subtopics: Any = None,
        keywords: Any = None,
        niche: str = "",
        min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
        relevance: Any = None,
        scorer: Any = None,
    ) -> None:
        self.include_terms = normalize_terms(include_terms, label="include term")
        self.exclude_terms = normalize_terms(exclude_terms, label="exclude term")
        expansion = QueryExpansion(
            queries=normalize_terms(queries, label="query"),
            subtopics=normalize_terms(subtopics, label="subtopic"),
            keywords=normalize_terms(keywords, label="keyword"),
            niche=niche,
        )
        self.queries = expansion.queries
        self.subtopics = expansion.subtopics
        self.keywords = expansion.keywords
        self.niche = expansion.niche
        self.min_relevance = _validate_threshold(min_relevance)
        self.threshold = self.min_relevance
        self.scorer = scorer if scorer is not None else relevance

    @property
    def required_terms(self) -> tuple[str, ...]:
        """The terms a candidate should match to be on theme."""
        if self.include_terms:
            return self.include_terms
        expansion = QueryExpansion(
            queries=self.queries,
            subtopics=self.subtopics,
            keywords=self.keywords,
            niche=self.niche,
        )
        return expansion.expanded_queries

    def term_score(self, candidate: TopicCandidate) -> float:
        """Return the share of required terms *candidate* matches."""
        text = _search_text(candidate)
        if any(_term_matches(text, term) for term in self.exclude_terms):
            return 0.0
        terms = self.required_terms
        if not terms:
            return 1.0
        matched = sum(1 for term in terms if _term_matches(text, term))
        return matched / len(terms)

    def score(self, candidate: TopicCandidate) -> float:
        """Return the relevance score for *candidate*."""
        if self.scorer is None:
            return self.term_score(candidate)
        result = _scorer_callable(self.scorer)(candidate)
        if inspect.isawaitable(result):
            raise ConfigError(
                "an async relevance scorer requires the mode's discover() method"
            )
        return _clamp_score(result)

    async def score_async(self, candidate: TopicCandidate) -> float:
        """Return the relevance score, awaiting an async scorer when needed."""
        if self.scorer is None:
            return self.term_score(candidate)
        result = _scorer_callable(self.scorer)(candidate)
        if inspect.isawaitable(result):
            result = await result
        return _clamp_score(result)

    def passes_score(self, score: float) -> bool:
        """Whether *score* clears the configured threshold."""
        if self.min_relevance is None:
            return True
        return score >= self.min_relevance

    def is_relevant(self, candidate: TopicCandidate) -> bool:
        """Whether *candidate* passes the gate."""
        return self.passes_score(self.score(candidate))

    def passes(self, candidate: TopicCandidate) -> bool:
        """Alias for :meth:`is_relevant`."""
        return self.is_relevant(candidate)

    def accepts(self, candidate: TopicCandidate) -> bool:
        """Alias for :meth:`is_relevant`."""
        return self.is_relevant(candidate)

    def filter(self, candidates: Sequence[TopicCandidate]) -> list[TopicCandidate]:
        """Return the candidates that pass the gate."""
        return [candidate for candidate in candidates if self.is_relevant(candidate)]

    def __call__(self, candidate: TopicCandidate) -> bool:
        return self.is_relevant(candidate)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(min_relevance={self.min_relevance!r}, "
            f"include_terms={self.include_terms!r}, "
            f"exclude_terms={self.exclude_terms!r})"
        )


def candidate_relevance(
    candidate: TopicCandidate,
    include_terms: Any = None,
    exclude_terms: Any = None,
    *,
    queries: Any = None,
    subtopics: Any = None,
    keywords: Any = None,
    niche: str = "",
    min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
    relevance: Any = None,
    scorer: Any = None,
) -> float:
    """Score *candidate* against terms or an injected scorer."""
    gate = RelevanceGate(
        include_terms=include_terms,
        exclude_terms=exclude_terms,
        queries=queries,
        subtopics=subtopics,
        keywords=keywords,
        niche=niche,
        min_relevance=min_relevance,
        relevance=relevance,
        scorer=scorer,
    )
    return gate.score(candidate)


def score_candidate(candidate: TopicCandidate, **values: Any) -> float:
    """Alias for :func:`candidate_relevance`."""
    return candidate_relevance(candidate, **values)


def is_relevant(
    candidate: TopicCandidate,
    include_terms: Any = None,
    exclude_terms: Any = None,
    *,
    queries: Any = None,
    subtopics: Any = None,
    keywords: Any = None,
    niche: str = "",
    min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
    relevance: Any = None,
    scorer: Any = None,
) -> bool:
    """Whether *candidate* clears the local relevance gate."""
    gate = RelevanceGate(
        include_terms=include_terms,
        exclude_terms=exclude_terms,
        queries=queries,
        subtopics=subtopics,
        keywords=keywords,
        niche=niche,
        min_relevance=min_relevance,
        relevance=relevance,
        scorer=scorer,
    )
    return gate.is_relevant(candidate)


def filter_candidates(
    candidates: Sequence[TopicCandidate],
    include_terms: Any = None,
    exclude_terms: Any = None,
    **values: Any,
) -> list[TopicCandidate]:
    """Return the candidates that pass the local relevance gate."""
    gate = RelevanceGate(
        include_terms=include_terms,
        exclude_terms=exclude_terms,
        **values,
    )
    return gate.filter(candidates)


relevance_gate = RelevanceGate
"""Alias for :class:`RelevanceGate`."""


def _coerce_sources(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [resolve_source(value)]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


class NicheMode(TopicModeBase):
    """Discover on-theme topics for a niche brief, with a relevance gate.

    ``source`` may be a provider instance, a provider class, a zero-argument
    factory, a registered source name or a list of any of those. It is created
    lazily, so constructing the mode never touches the network.

    ``derivation`` is the :class:`~aivia.research.brief.BriefDerivation` for
    the profile brief. Explicit ``queries``, ``subtopics``, ``keywords``,
    ``include_terms`` and ``exclude_terms`` override the matching derivation
    fields. A ``relevance`` callable or scorer replaces the local term gate.
    """

    name = "niche"

    def __init__(
        self,
        source: Any = None,
        *,
        provider: Any = None,
        source_factory: Callable[[], Any] | None = None,
        sources: Any = None,
        derivation: Any = None,
        brief_derivation: Any = None,
        profile: Any = None,
        queries: Any = None,
        search_queries: Any = None,
        subtopics: Any = None,
        sub_topics: Any = None,
        keywords: Any = None,
        niche: str | None = None,
        include_terms: Any = None,
        include: Any = None,
        exclude_terms: Any = None,
        exclude: Any = None,
        min_relevance: float | None = DEFAULT_MIN_RELEVANCE,
        relevance_threshold: float | None = None,
        threshold: float | None = None,
        relevance: Any = None,
        relevance_scorer: Any = None,
        scorer: Any = None,
        gate: RelevanceGate | None = None,
        concurrency: int | None = None,
        max_concurrency: int | None = None,
        limit: int | None = None,
        per_query_limit: int | None = None,
        max_queries: int | None = None,
        query_limit: int | None = None,
        language: str | None = None,
        timeout: float | None = None,
        timeout_seconds: float | None = None,
        request_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        continue_on_error: bool = False,
        name: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(name)
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
        if brief_derivation is not None:
            if derivation is not None:
                raise ConfigError("use either derivation or brief_derivation, not both")
            derivation = brief_derivation
        if search_queries is not None:
            if queries is not None:
                raise ConfigError("use either queries or search_queries, not both")
            queries = search_queries
        if sub_topics is not None:
            if subtopics is not None:
                raise ConfigError("use either subtopics or sub_topics, not both")
            subtopics = sub_topics
        if include is not None:
            if include_terms is not None:
                raise ConfigError("use either include_terms or include, not both")
            include_terms = include
        if exclude is not None:
            if exclude_terms is not None:
                raise ConfigError("use either exclude_terms or exclude, not both")
            exclude_terms = exclude
        if max_concurrency is not None:
            if concurrency is not None:
                raise ConfigError("use either concurrency or max_concurrency, not both")
            concurrency = max_concurrency
        if per_query_limit is not None:
            if limit is not None:
                raise ConfigError("use either limit or per_query_limit, not both")
            limit = per_query_limit
        if query_limit is not None:
            if max_queries is not None:
                raise ConfigError("use either max_queries or query_limit, not both")
            max_queries = query_limit
        if timeout_seconds is not None:
            if timeout is not None:
                raise ConfigError("use either timeout or timeout_seconds, not both")
            timeout = timeout_seconds
        if relevance_threshold is not None:
            min_relevance = relevance_threshold
        if threshold is not None:
            min_relevance = threshold
        if relevance_scorer is not None:
            if relevance is not None:
                raise ConfigError("use either relevance or relevance_scorer, not both")
            relevance = relevance_scorer
        if scorer is not None:
            if relevance is not None:
                raise ConfigError("use either relevance or scorer, not both")
            relevance = scorer

        if profile is not None and derivation is None:
            if queries is None:
                profile_keywords = _mapping_value(profile, "keywords", default=())
                profile_niche = _mapping_value(profile, "niche", default="")
                profile_queries = [*profile_keywords]
                if profile_niche:
                    profile_queries.append(str(profile_niche))
                queries = profile_queries
            if niche is None:
                niche = str(_mapping_value(profile, "niche", default="") or "")
            if keywords is None:
                keywords = _mapping_value(profile, "keywords", default=())

        self._derivation = derivation
        self._expansion = QueryExpansion.from_sources(
            derivation,
            queries=queries,
            subtopics=subtopics,
            keywords=keywords,
            niche=niche,
            include_terms=include_terms,
            exclude_terms=exclude_terms,
        )
        self._max_queries = _validate_max_queries(max_queries)
        self._queries = self._expansion.expanded_queries[: self._max_queries]
        self._source_specs = _coerce_sources(source)
        self._source_instances: list[Any] | None = None
        self._concurrency = _validate_concurrency(concurrency)
        self._limit = _validate_limit(limit)
        self._language = (
            str(language).strip().lower().replace("_", "-") if language else None
        )
        number = timeout
        if number is not None:
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ConfigError("timeout must be a number")
            number = float(number)
            if not math.isfinite(number) or number <= 0:
                raise ConfigError("timeout must be a positive finite number")
        self._timeout = number
        self._request_metadata = dict(request_metadata or metadata or {})
        self._continue_on_error = bool(continue_on_error)
        self._options = dict(options)
        if gate is not None:
            self._gate = gate
        else:
            self._gate = RelevanceGate(
                include_terms=self._expansion.include_terms,
                exclude_terms=self._expansion.exclude_terms,
                queries=self._expansion.queries,
                subtopics=self._expansion.subtopics,
                keywords=self._expansion.keywords,
                niche=self._expansion.niche,
                min_relevance=min_relevance,
                relevance=relevance,
            )

    @property
    def derivation(self) -> Any:
        """The brief derivation this mode expands, when one was supplied."""
        return self._derivation

    @property
    def expansion(self) -> QueryExpansion:
        """The query expansion built from the derivation and overrides."""
        return self._expansion

    @property
    def queries(self) -> list[str]:
        """The expanded queries this mode sends to the source."""
        return list(self._queries)

    @property
    def expanded_queries(self) -> list[str]:
        """Alias for :attr:`queries`."""
        return list(self._queries)

    @property
    def search_queries(self) -> list[str]:
        """Alias for :attr:`queries`."""
        return list(self._queries)

    @property
    def base_queries(self) -> list[str]:
        """The derivation's own queries, before subtopics and keywords."""
        return list(self._expansion.queries)

    @property
    def subtopics(self) -> list[str]:
        """The subtopics used to expand the queries."""
        return list(self._expansion.subtopics)

    @property
    def keywords(self) -> list[str]:
        """The explicit keywords used to expand the queries."""
        return list(self._expansion.keywords)

    @property
    def include_terms(self) -> list[str]:
        """The include terms the gate uses."""
        return list(self._expansion.include_terms)

    @property
    def exclude_terms(self) -> list[str]:
        """The exclude terms the gate uses."""
        return list(self._expansion.exclude_terms)

    @property
    def niche(self) -> str:
        """The profile niche used to expand the queries."""
        return self._expansion.niche

    @property
    def min_relevance(self) -> float | None:
        """The relevance threshold, or ``None`` when the gate is disabled."""
        return self._gate.min_relevance

    @property
    def threshold(self) -> float | None:
        """Alias for :attr:`min_relevance`."""
        return self._gate.min_relevance

    @property
    def concurrency(self) -> int:
        """How many source calls may run at the same time."""
        return self._concurrency

    @property
    def gate(self) -> RelevanceGate:
        """The relevance gate applied to every candidate."""
        return self._gate

    @property
    def options(self) -> dict[str, Any]:
        """Additional options accepted for forward compatibility."""
        return dict(self._options)

    @property
    def sources(self) -> list[Any]:
        """The source instances, creating lazy defaults on first use."""
        if self._source_instances is None:
            specs = self._source_specs or [WikipediaSourceProvider]
            instances: list[Any] = []
            for spec in specs:
                if isinstance(spec, str):
                    spec = resolve_source(spec)
                if not hasattr(spec, "discover") and hasattr(spec, "provider"):
                    spec = spec.provider
                if isinstance(spec, type) or (
                    callable(spec) and not hasattr(spec, "discover")
                ):
                    spec = spec()
                if not hasattr(spec, "discover"):
                    raise ConfigError("niche source must define discover()")
                instances.append(spec)
            self._source_instances = instances
        return list(self._source_instances)

    @property
    def source(self) -> Any:
        """The only source instance, or the list when several are configured."""
        instances = self.sources
        if len(instances) == 1:
            return instances[0]
        return instances

    def _queries_for(self, request: SourceRequest) -> list[str]:
        queries = list(self._queries)
        if not queries:
            query = request.query.strip()
            queries = [query] if query else [""]
        return queries

    def _request_for(self, request: SourceRequest, query: str) -> SourceRequest:
        update: dict[str, Any] = {"query": query, "mode": self.name}
        if self._limit is not None:
            update["limit"] = self._limit
        if self._language is not None:
            update["language"] = self._language
        if self._timeout is not None:
            update["timeout_seconds"] = self._timeout
        if self._request_metadata:
            update["metadata"] = {**request.metadata, **self._request_metadata}
        return request.model_copy(update=update)

    async def _discover_one(
        self,
        source: Any,
        request: SourceRequest,
        query: str,
        semaphore: asyncio.Semaphore,
    ) -> SourceResponse:
        async with semaphore:
            response = await source.discover(self._request_for(request, query))
        if not isinstance(response, SourceResponse):
            raise ConfigError("niche source must return a SourceResponse")
        return response

    @staticmethod
    def _provider_label(providers: Sequence[str]) -> str:
        labels: list[str] = []
        for provider in providers:
            if provider and provider not in labels:
                labels.append(provider)
        return ", ".join(labels)

    async def discover(self, request: SourceRequest) -> SourceResponse:
        """Discover and gate niche candidates for *request*."""
        if not isinstance(request, SourceRequest):
            raise TypeError("request must be a SourceRequest")

        queries = self._queries_for(request)
        sources = self.sources
        semaphore = asyncio.Semaphore(self._concurrency)
        tasks = [
            self._discover_one(source, request, query, semaphore)
            for source in sources
            for query in queries
        ]
        results: list[Any]
        if self._continue_on_error:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = list(await asyncio.gather(*tasks))

        responses: list[SourceResponse] = []
        errors: list[dict[str, Any]] = []
        labels = [
            (index // len(queries), queries[index % len(queries)])
            for index in range(len(tasks))
        ]
        for (source_index, query), result in zip(labels, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(
                    {
                        "query": query,
                        "source": source_index,
                        "error": str(result),
                        "type": type(result).__name__,
                    }
                )
                continue
            responses.append(result)

        if not responses and errors:
            empty_metadata = {
                "mode": self.name,
                "source_provider": "",
                "queries": list(queries),
                "expanded_queries": list(queries),
                "query_count": len(queries),
                "source_count": len(sources),
                "concurrency": self._concurrency,
                "min_relevance": self._gate.min_relevance,
                "skipped_relevance": 0,
                "skipped_irrelevant": 0,
                "skipped_off_topic": 0,
                "skipped_duplicates": 0,
                "errors": errors,
                "failed_queries": len(errors),
            }
            return SourceResponse(
                provider=self.name,
                query=request.query or (queries[0] if queries else ""),
                candidates=[],
                metadata=empty_metadata,
            )

        metadata: dict[str, Any] = {}
        providers: list[str] = []
        request_id: str | None = None
        for response in responses:
            for key, value in response.metadata.items():
                metadata.setdefault(key, value)
            providers.append(response.provider)
            if request_id is None:
                request_id = response.request_id

        candidates: list[TopicCandidate] = []
        seen: set[str] = set()
        skipped_relevance = 0
        skipped_duplicates = 0
        for response in responses:
            for candidate in response.candidates:
                key = normalize_title(candidate.title)
                if key in seen:
                    skipped_duplicates += 1
                    continue
                score = await self._gate.score_async(candidate)
                if not self._gate.passes_score(score):
                    skipped_relevance += 1
                    continue
                seen.add(key)
                candidates.append(
                    candidate.model_copy(
                        update={"mode": self.name, "rank": len(candidates)}
                    )
                )

        metadata.update(
            {
                "mode": self.name,
                "source_provider": self._provider_label(providers),
                "queries": list(queries),
                "expanded_queries": list(queries),
                "query_count": len(queries),
                "source_count": len(sources),
                "concurrency": self._concurrency,
                "min_relevance": self._gate.min_relevance,
                "skipped_relevance": skipped_relevance,
                "skipped_irrelevant": skipped_relevance,
                "skipped_off_topic": skipped_relevance,
                "skipped_duplicates": skipped_duplicates,
            }
        )
        if errors:
            metadata["errors"] = errors
            metadata["failed_queries"] = len(errors)

        return SourceResponse(
            provider=self.name,
            query=request.query or (queries[0] if queries else ""),
            candidates=candidates,
            request_id=request_id,
            metadata=metadata,
        )


Niche = NicheMode
"""Alias for :class:`NicheMode`."""

NicheTopicMode = NicheMode
"""Alias for :class:`NicheMode`."""

WikipediaNicheMode = NicheMode
"""Alias for :class:`NicheMode`."""

NicheModeProvider = NicheMode
"""Alias for :class:`NicheMode`."""

NicheSourceMode = NicheMode
"""Alias for :class:`NicheMode`."""

DEFAULT_MAX_CONCURRENCY = DEFAULT_CONCURRENCY
"""Alias for :data:`DEFAULT_CONCURRENCY`."""

DEFAULT_RELEVANCE_FLOOR = DEFAULT_MIN_RELEVANCE
"""Alias for :data:`DEFAULT_MIN_RELEVANCE`."""

DEFAULT_MIN_SCORE = DEFAULT_MIN_RELEVANCE
"""Alias for :data:`DEFAULT_MIN_RELEVANCE`."""

DEFAULT_QUERY_EXPANSION_LIMIT = DEFAULT_MAX_QUERIES
"""Alias for :data:`DEFAULT_MAX_QUERIES`."""

expand_query = expand_queries
"""Alias for :func:`expand_queries`."""

expand_search_queries = expand_queries
"""Alias for :func:`expand_queries`."""

score_relevance = candidate_relevance
"""Alias for :func:`candidate_relevance`."""

relevance_score = candidate_relevance
"""Alias for :func:`candidate_relevance`."""

passes_relevance = is_relevant
"""Alias for :func:`is_relevant`."""

filter_relevant = filter_candidates
"""Alias for :func:`filter_candidates`."""

gate_candidates = filter_candidates
"""Alias for :func:`filter_candidates`."""

register_mode("niche", NicheMode)
