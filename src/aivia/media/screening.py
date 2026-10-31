"""Screen prompts and content before a paid provider call.

A paid service must never talk to a provider until the text it is about to send
has passed the configured content filter. Screening returns an immutable
:class:`ScreeningToken` that binds the approved text and its kind to a stable
digest. Paid services call :func:`require_screened` with that token and refuse
to run when it is missing, malformed or belongs to different text, so a caller
cannot skip the filter by constructing a request directly.

Terms come from the ``safety`` configuration section (``blocked_terms`` and
``block_categories``) and may be extended per call. Matching is literal,
case-insensitive and Unicode-normalised; a term only matches whole words or
phrases, so ``class`` does not trigger a configured ``ass`` term. The module is
deterministic and never performs a provider call, a network request or a model
judgement.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel
from aivia.core.config.sections import SafetyConfig
from aivia.core.errors import ContentPolicyError, ValidationError

__all__ = [
    "DEFAULT_CONTENT_KIND",
    "DEFAULT_PROMPT_KIND",
    "SCREENING_TOKEN_PREFIX",
    "ContentScreener",
    "PromptScreener",
    "Screener",
    "ScreeningConfig",
    "ScreeningResult",
    "ScreeningService",
    "ScreeningToken",
    "blocked_terms_in",
    "build_screening_token",
    "check_content",
    "check_prompt",
    "configured_terms",
    "find_blocked_terms",
    "is_screened",
    "match_blocked_terms",
    "parse_screening_token",
    "require_screened",
    "require_token",
    "screen",
    "screen_content",
    "screen_media",
    "screen_prompt",
    "screen_text",
    "screening_terms",
    "screening_token",
    "terms_from_config",
    "validate_token",
]

SCREENING_TOKEN_PREFIX = "aivia-screened"
"""The fixed prefix of every issued screening token."""

DEFAULT_PROMPT_KIND = "prompt"
"""The kind recorded for prompts sent to image, video, speech or music models."""

DEFAULT_CONTENT_KIND = "content"
"""The kind recorded for user or fetched text that is not a generation prompt."""

_PROMPT_KINDS = frozenset(
    {
        "prompt",
        "text",
        "script",
        "caption",
        "title",
        "description",
        "lyrics",
        "query",
    }
)
_TERM_ALIASES = (
    "block_terms",
    "blocked",
    "terms",
    "banned_terms",
    "deny_terms",
    "blocklist",
)
_CONFIG_FIELDS = (
    "enabled",
    "screen_prompts",
    "screen_media",
    "blocked_terms",
    "block_categories",
    "allow_regenerate",
)


def _text(value: Any, *, label: str = "text") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"{label} must be valid UTF-8") from exc
    raise ValidationError(f"{label} must be a string")


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _clean_terms(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        candidates: Sequence[Any] = (values,)
    elif isinstance(values, Sequence):
        candidates = values
    else:
        candidates = (values,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value is None:
            continue
        term = _text(value, label="screening term").strip()
        if not term:
            continue
        key = _normalise(term)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(term)
    return tuple(cleaned)


def _term_pattern(term: str) -> re.Pattern[str]:
    normalised = _normalise(term).strip()
    parts = re.split(r"\s+", normalised)
    body = r"\s+".join(re.escape(part) for part in parts)
    prefix = r"(?<!\w)" if normalised and _is_word_character(normalised[0]) else ""
    suffix = r"(?!\w)" if normalised and _is_word_character(normalised[-1]) else ""
    return re.compile(prefix + body + suffix)


def find_blocked_terms(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    """Return the configured terms that occur in *text*, in term order.

    Matching is case-insensitive, Unicode-normalised and literal. A term that
    starts or ends with a word character only matches at a word boundary, so a
    short term cannot accidentally block a longer unrelated word.
    """
    normalised = _normalise(_text(text))
    matches: list[str] = []
    for term in _clean_terms(terms):
        if _term_pattern(term).search(normalised):
            matches.append(term)
    return tuple(matches)


def _kind(value: Any) -> str:
    text = str(value or DEFAULT_PROMPT_KIND).strip().lower()
    safe = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("._-")
    return safe or DEFAULT_PROMPT_KIND


def _is_prompt_kind(kind: str) -> bool:
    return kind in _PROMPT_KINDS or kind.startswith("prompt")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ScreeningToken(str):
    """An immutable proof that *text* passed screening.

    The value is a self-describing string so it can be stored in a sidecar or
    passed through a queue. Attribute access exposes the bound kind, digest and
    the policy names that were checked, which paid services can record without
    re-parsing the value.
    """

    __slots__ = (
        "_blocked_terms",
        "_categories",
        "_kind",
        "_prompt_hash",
        "_text",
    )

    _blocked_terms: tuple[str, ...]
    _categories: tuple[str, ...]
    _kind: str
    _prompt_hash: str
    _text: str

    def __new__(
        cls,
        text: str = "",
        *,
        kind: str = DEFAULT_PROMPT_KIND,
        blocked_terms: Sequence[str] = (),
        categories: Sequence[str] = (),
        prompt_hash: str | None = None,
    ) -> ScreeningToken:
        content = _text(text)
        token_kind = _kind(kind)
        digest = prompt_hash or _digest(content)
        value = f"{SCREENING_TOKEN_PREFIX}:{token_kind}:{digest}"
        token = super().__new__(cls, value)
        token._text = content
        token._kind = token_kind
        token._prompt_hash = digest
        token._blocked_terms = tuple(blocked_terms)
        token._categories = tuple(categories)
        return token

    @property
    def token(self) -> str:
        """The token as a plain string."""
        return str(self)

    @property
    def value(self) -> str:
        """Alias for :attr:`token`."""
        return str(self)

    @property
    def kind(self) -> str:
        """The text kind the token was issued for."""
        return self._kind

    @property
    def prompt_hash(self) -> str:
        """The SHA-256 digest of the screened text."""
        return self._prompt_hash

    @property
    def text_hash(self) -> str:
        """Alias for :attr:`prompt_hash`."""
        return self._prompt_hash

    @property
    def digest(self) -> str:
        """Alias for :attr:`prompt_hash`."""
        return self._prompt_hash

    @property
    def text(self) -> str:
        """The screened text the token was issued for."""
        return self._text

    @property
    def prompt(self) -> str:
        """Alias for :attr:`text`."""
        return self._text

    @property
    def content(self) -> str:
        """Alias for :attr:`text`."""
        return self._text

    @property
    def blocked_terms(self) -> tuple[str, ...]:
        """Terms that matched, which is always empty for a valid token."""
        return self._blocked_terms

    @property
    def terms(self) -> tuple[str, ...]:
        """Alias for :attr:`blocked_terms`."""
        return self._blocked_terms

    @property
    def categories(self) -> tuple[str, ...]:
        """Policy categories configured when the token was issued."""
        return self._categories

    @property
    def allowed(self) -> bool:
        """Whether screening allowed the text."""
        return True

    @property
    def screened(self) -> bool:
        """Alias for :attr:`allowed`."""
        return True

    @property
    def passed(self) -> bool:
        """Alias for :attr:`allowed`."""
        return True

    @property
    def blocked(self) -> bool:
        """Whether screening rejected the text."""
        return False

    def as_mapping(self) -> dict[str, Any]:
        """Return the token and its bound metadata as a plain mapping."""
        return {
            "token": str(self),
            "kind": self._kind,
            "prompt_hash": self._prompt_hash,
            "text": self._text,
            "blocked_terms": list(self._blocked_terms),
            "categories": list(self._categories),
        }


_ScreeningTokenMaybe = ScreeningToken | None


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """The non-raising outcome of a screening check."""

    allowed: bool
    text: str
    kind: str
    token: _ScreeningTokenMaybe = None
    blocked_terms: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    reason: str = ""

    @property
    def blocked(self) -> bool:
        """Whether the text was rejected."""
        return not self.allowed

    @property
    def ok(self) -> bool:
        """Alias for :attr:`allowed`."""
        return self.allowed

    @property
    def passed(self) -> bool:
        """Alias for :attr:`allowed`."""
        return self.allowed

    @property
    def terms(self) -> tuple[str, ...]:
        """Alias for :attr:`blocked_terms`."""
        return self.blocked_terms

    def as_mapping(self) -> dict[str, Any]:
        """Return the outcome as a plain mapping."""
        return {
            "allowed": self.allowed,
            "text": self.text,
            "kind": self.kind,
            "token": str(self.token) if self.token is not None else None,
            "blocked_terms": list(self.blocked_terms),
            "categories": list(self.categories),
            "reason": self.reason,
        }


class ScreeningConfig(ConfigModel):
    """Standalone screening settings for callers without an app config.

    The canonical source is :class:`~aivia.core.config.sections.SafetyConfig`;
    this model mirrors it and accepts common aliases so the screener can also
    be used on its own.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = True
    screen_prompts: bool = True
    screen_media: bool = True
    blocked_terms: list[str] = Field(default_factory=list)
    block_categories: list[str] = Field(default_factory=list)
    allow_regenerate: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for alias in _TERM_ALIASES:
            if alias in values:
                value = values.pop(alias)
                values.setdefault("blocked_terms", value)
        return values

    @field_validator("blocked_terms", "block_categories", mode="before")
    @classmethod
    def _coerce_terms(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("blocked_terms", "block_categories")
    @classmethod
    def _validate_terms(cls, value: list[str]) -> list[str]:
        terms = [term.strip() for term in value]
        if any(not term for term in terms):
            raise ValueError("screening terms must not be empty")
        return terms


@dataclass(frozen=True, slots=True)
class _Settings:
    enabled: bool
    screen_prompts: bool
    screen_media: bool
    blocked_terms: tuple[str, ...]
    categories: tuple[str, ...]
    allow_regenerate: bool


def _config_values(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        nested = config.get("safety")
        if nested is not None:
            if isinstance(nested, Mapping):
                return nested
            if isinstance(nested, (SafetyConfig, ScreeningConfig)):
                return {name: getattr(nested, name) for name in _CONFIG_FIELDS}
        return {name: config[name] for name in _CONFIG_FIELDS if name in config}
    safety = getattr(config, "safety", None)
    if safety is not None and safety is not config:
        return _config_values(safety)
    return {
        name: getattr(config, name) for name in _CONFIG_FIELDS if hasattr(config, name)
    }


def _settings(config: Any) -> _Settings:
    if config is None:
        values: Mapping[str, Any] = {}
    elif isinstance(config, (SafetyConfig, ScreeningConfig)):
        values = {name: getattr(config, name) for name in _CONFIG_FIELDS}
    else:
        values = _config_values(config)
    parsed = ScreeningConfig.model_validate(dict(values))
    categories = _clean_terms(parsed.block_categories)
    return _Settings(
        enabled=parsed.enabled,
        screen_prompts=parsed.screen_prompts,
        screen_media=parsed.screen_media,
        blocked_terms=_clean_terms(parsed.blocked_terms),
        categories=categories,
        allow_regenerate=parsed.allow_regenerate,
    )


def configured_terms(config: Any = None) -> tuple[str, ...]:
    """Return every literal term configured for *config*."""
    settings = _settings(config)
    return _clean_terms((*settings.blocked_terms, *settings.categories))


def terms_from_config(config: Any = None) -> tuple[str, ...]:
    """Alias for :func:`configured_terms`."""
    return configured_terms(config)


def screening_terms(config: Any = None) -> tuple[str, ...]:
    """Alias for :func:`configured_terms`."""
    return configured_terms(config)


def _matches_for(
    text: str,
    settings: _Settings,
    extra_terms: Sequence[str],
    extra_categories: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    categories = _clean_terms((*settings.categories, *extra_categories))
    terms = _clean_terms((*settings.blocked_terms, *extra_terms, *categories))
    return find_blocked_terms(text, terms), categories


def _raise_blocked(
    text: str,
    kind: str,
    matches: Sequence[str],
    categories: Sequence[str],
) -> None:
    listed = ", ".join(matches)
    raise ContentPolicyError(
        f"{kind} blocked by screening: matched configured term(s): {listed}",
        details={
            "source": "screening",
            "kind": kind,
            "blocked_terms": list(matches),
            "categories": list(categories),
            "text_hash": _digest(text),
        },
    )


def screen_text(
    text: str,
    config: Any = None,
    *,
    kind: str = DEFAULT_PROMPT_KIND,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> ScreeningToken:
    """Screen *text* and return the token a paid service requires.

    A configured term raises :class:`~aivia.core.errors.ContentPolicyError`
    before any provider is touched. When safety or the matching screen switch
    is disabled, the text is still wrapped in a token so the paid path stays
    uniform.
    """
    content = _text(text)
    token_kind = _kind(kind)
    settings = _settings(config)
    extra_terms = _clean_terms(terms)
    extra_categories = _clean_terms(categories)
    prompt_kind = _is_prompt_kind(token_kind)
    enabled = settings.enabled and (
        settings.screen_prompts if prompt_kind else settings.screen_media
    )
    matches: tuple[str, ...] = ()
    if enabled:
        matches, matched_categories = _matches_for(
            content,
            settings,
            extra_terms,
            extra_categories,
        )
        if matches:
            _raise_blocked(content, token_kind, matches, matched_categories)
    else:
        matched_categories = _clean_terms((*settings.categories, *extra_categories))
    return ScreeningToken(
        content,
        kind=token_kind,
        categories=matched_categories,
    )


def screen_prompt(
    prompt: str,
    config: Any = None,
    *,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
    kind: str = DEFAULT_PROMPT_KIND,
) -> ScreeningToken:
    """Screen a generation prompt and return its screening token."""
    return screen_text(
        prompt,
        config,
        kind=kind,
        terms=terms,
        categories=categories,
    )


def screen_content(
    content: str,
    config: Any = None,
    *,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
    kind: str = DEFAULT_CONTENT_KIND,
) -> ScreeningToken:
    """Screen non-prompt content such as fetched text or an image caption."""
    return screen_text(
        content,
        config,
        kind=kind,
        terms=terms,
        categories=categories,
    )


def screen_media(
    content: str,
    config: Any = None,
    *,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
    kind: str = DEFAULT_CONTENT_KIND,
) -> ScreeningToken:
    """Alias for :func:`screen_content`."""
    return screen_content(
        content,
        config,
        terms=terms,
        categories=categories,
        kind=kind,
    )


def screen(
    text: str,
    config: Any = None,
    *,
    kind: str = DEFAULT_PROMPT_KIND,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> ScreeningToken:
    """Alias for :func:`screen_text`."""
    return screen_text(
        text,
        config,
        kind=kind,
        terms=terms,
        categories=categories,
    )


def screening_token(
    text: str,
    *,
    kind: str = DEFAULT_PROMPT_KIND,
    blocked_terms: Sequence[str] = (),
    categories: Sequence[str] = (),
) -> ScreeningToken:
    """Build a token for already screened *text* without running the filter."""
    return ScreeningToken(
        text,
        kind=kind,
        blocked_terms=blocked_terms,
        categories=categories,
    )


def build_screening_token(
    text: str,
    *,
    kind: str = DEFAULT_PROMPT_KIND,
    blocked_terms: Sequence[str] = (),
    categories: Sequence[str] = (),
) -> ScreeningToken:
    """Alias for :func:`screening_token`."""
    return screening_token(
        text,
        kind=kind,
        blocked_terms=blocked_terms,
        categories=categories,
    )


def parse_screening_token(token: Any) -> ScreeningToken | None:
    """Parse a token value, returning ``None`` when it is not well formed."""
    if isinstance(token, ScreeningToken):
        return token
    if not isinstance(token, str):
        return None
    parts = token.split(":")
    if len(parts) != 3 or parts[0] != SCREENING_TOKEN_PREFIX:
        return None
    kind, digest = parts[1], parts[2]
    if not kind or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return None
    return ScreeningToken("", kind=kind, prompt_hash=digest)


def require_screened(
    token: Any,
    text: str,
    *,
    kind: str | None = None,
    config: Any = None,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> ScreeningToken:
    """Return the token when it proves *text* was screened, otherwise raise.

    Paid services call this immediately before building a provider request. A
    missing, malformed or mismatched token is a
    :class:`~aivia.core.errors.ContentPolicyError`, as is text that matches a
    configured term when a configuration is supplied.
    """
    content = _text(text)
    parsed = parse_screening_token(token)
    if parsed is None:
        raise ContentPolicyError(
            "a screening token is required before a paid call",
            details={
                "source": "screening",
                "reason": "missing_token" if token is None else "invalid_token",
            },
        )
    if kind is not None and parsed.kind != _kind(kind):
        raise ContentPolicyError(
            "the screening token was issued for a different text kind",
            details={
                "source": "screening",
                "reason": "kind_mismatch",
                "expected_kind": _kind(kind),
                "token_kind": parsed.kind,
            },
        )
    expected = _digest(content)
    if parsed.prompt_hash != expected:
        raise ContentPolicyError(
            "the screening token does not match the text about to be sent",
            details={
                "source": "screening",
                "reason": "text_mismatch",
                "expected_hash": expected,
                "token_hash": parsed.prompt_hash,
            },
        )
    screen_text(
        content,
        config,
        kind=parsed.kind,
        terms=terms,
        categories=categories,
    )
    return ScreeningToken(
        content,
        kind=parsed.kind,
        blocked_terms=parsed.blocked_terms,
        categories=parsed.categories,
        prompt_hash=parsed.prompt_hash,
    )


def validate_token(
    token: Any,
    text: str,
    *,
    kind: str | None = None,
    config: Any = None,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> bool:
    """Return whether *token* proves *text* was screened."""
    try:
        require_screened(
            token,
            text,
            kind=kind,
            config=config,
            terms=terms,
            categories=categories,
        )
    except ContentPolicyError:
        return False
    return True


def is_screened(
    token: Any,
    text: str,
    *,
    kind: str | None = None,
    config: Any = None,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> bool:
    """Alias for :func:`validate_token`."""
    return validate_token(
        token,
        text,
        kind=kind,
        config=config,
        terms=terms,
        categories=categories,
    )


def require_token(
    token: Any,
    text: str,
    *,
    kind: str | None = None,
    config: Any = None,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> ScreeningToken:
    """Alias for :func:`require_screened`."""
    return require_screened(
        token,
        text,
        kind=kind,
        config=config,
        terms=terms,
        categories=categories,
    )


def _check(
    text: str,
    config: Any,
    kind: str,
    terms: Sequence[str] | str | None,
    categories: Sequence[str] | str | None,
) -> ScreeningResult:
    content = _text(text)
    token_kind = _kind(kind)
    try:
        token = screen_text(
            content,
            config,
            kind=token_kind,
            terms=terms,
            categories=categories,
        )
    except ContentPolicyError as error:
        return ScreeningResult(
            allowed=False,
            text=content,
            kind=token_kind,
            blocked_terms=tuple(error.details.get("blocked_terms", ())),
            categories=tuple(error.details.get("categories", ())),
            reason=error.message,
        )
    return ScreeningResult(
        allowed=True,
        text=content,
        kind=token_kind,
        token=token,
        categories=token.categories,
    )


def check_prompt(
    prompt: str,
    config: Any = None,
    *,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
) -> ScreeningResult:
    """Screen a prompt without raising and return a structured outcome."""
    return _check(
        prompt,
        config,
        DEFAULT_PROMPT_KIND,
        terms,
        categories,
    )


def check_content(
    content: str,
    config: Any = None,
    *,
    terms: Sequence[str] | str | None = None,
    categories: Sequence[str] | str | None = None,
    kind: str = DEFAULT_CONTENT_KIND,
) -> ScreeningResult:
    """Screen non-prompt content without raising."""
    return _check(content, config, kind, terms, categories)


def blocked_terms_in(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    """Alias for :func:`find_blocked_terms`."""
    return find_blocked_terms(text, terms)


def match_blocked_terms(text: str, terms: Sequence[str]) -> tuple[str, ...]:
    """Alias for :func:`find_blocked_terms`."""
    return find_blocked_terms(text, terms)


class Screener:
    """A reusable screener bound to one configuration and optional terms."""

    def __init__(
        self,
        config: Any = None,
        *,
        terms: Sequence[str] | str | None = None,
        categories: Sequence[str] | str | None = None,
    ) -> None:
        self.config = config
        self.terms = _clean_terms(terms)
        self.categories = _clean_terms(categories)

    @property
    def blocked_terms(self) -> tuple[str, ...]:
        """Every literal term this screener will block."""
        return _clean_terms(
            (*configured_terms(self.config), *self.terms, *self.categories)
        )

    def screen_prompt(
        self, prompt: str, *, kind: str = DEFAULT_PROMPT_KIND
    ) -> ScreeningToken:
        """Screen a prompt and return its token."""
        return screen_prompt(
            prompt,
            self.config,
            terms=self.terms,
            categories=self.categories,
            kind=kind,
        )

    def screen_content(
        self,
        content: str,
        *,
        kind: str = DEFAULT_CONTENT_KIND,
    ) -> ScreeningToken:
        """Screen non-prompt content and return its token."""
        return screen_content(
            content,
            self.config,
            terms=self.terms,
            categories=self.categories,
            kind=kind,
        )

    def screen_text(
        self,
        text: str,
        *,
        kind: str = DEFAULT_PROMPT_KIND,
    ) -> ScreeningToken:
        """Screen text and return its token."""
        return screen_text(
            text,
            self.config,
            terms=self.terms,
            categories=self.categories,
            kind=kind,
        )

    def require(
        self,
        token: Any,
        text: str,
        *,
        kind: str | None = None,
    ) -> ScreeningToken:
        """Require a valid token for *text*."""
        return require_screened(
            token,
            text,
            kind=kind,
            config=self.config,
            terms=self.terms,
            categories=self.categories,
        )

    def validate(
        self,
        token: Any,
        text: str,
        *,
        kind: str | None = None,
    ) -> bool:
        """Return whether *token* proves *text* was screened."""
        return validate_token(
            token,
            text,
            kind=kind,
            config=self.config,
            terms=self.terms,
            categories=self.categories,
        )

    def check_prompt(self, prompt: str) -> ScreeningResult:
        """Screen a prompt without raising."""
        return check_prompt(
            prompt,
            self.config,
            terms=self.terms,
            categories=self.categories,
        )

    def check_content(
        self,
        content: str,
        *,
        kind: str = DEFAULT_CONTENT_KIND,
    ) -> ScreeningResult:
        """Screen non-prompt content without raising."""
        return check_content(
            content,
            self.config,
            terms=self.terms,
            categories=self.categories,
            kind=kind,
        )


ContentScreener = Screener
PromptScreener = Screener
ScreeningService = Screener
