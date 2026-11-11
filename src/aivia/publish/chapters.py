"""Generate description chapters from a transcript and validate them.

A transcript (word level timestamps from speech to text) is split into timed
chapters, each with a start time and a title derived from the words spoken in
that window. The chapters are validated against the platform chapter rules
(chapter count, minimum chapter duration, chapter title length and the
resulting description length) before they are embedded in a description ready
for publishing.

The generator is deterministic: the same transcript, platform and target
chapter length always produce the same chapters. It never calls a provider or
the network.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from aivia.core.errors import ConfigError, ContentPolicyError
from aivia.core.platforms import (
    ChapterRules,
    Platform,
    PlatformCatalog,
    get_platform,
)
from aivia.providers.protocols.audio import WordTimestamp
from aivia.providers.protocols.stt import STTResponse

__all__ = [
    "Chapter",
    "ChapterGenerator",
    "ChapterSet",
    "build_chapters",
    "generate_chapters",
]

_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+", re.UNICODE)
_ELLIPSIS = "…"


def _clean(value: Any) -> str:
    return _WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _sentence(text: str) -> str:
    text = _clean(text)
    if text and text[-1] not in ".!?":
        return f"{text}."
    return text


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    if not limit:
        return ""
    return text[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS


def _join(parts: Sequence[str]) -> str:
    return " ".join(part for part in (_clean(value) for value in parts) if part)


def _format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``M:SS`` or ``H:MM:SS`` for a description chapter."""
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class Chapter(BaseModel):
    """One timed chapter: when it starts and the title shown in a description."""

    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(ge=0.0)
    title: str = ""

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, value: Any) -> str:
        return _clean(value)

    @property
    def label(self) -> str:
        """The ``timestamp title`` line embedded in a description."""
        return f"{_format_timestamp(self.start_seconds)} {self.title}".strip()


class ChapterSet(BaseModel):
    """The chapters generated for one transcript and platform.

    ``chapters`` already respects the platform chapter count and title limits.
    ``description`` is the chapter text embedded in a base description and fitted
    to the platform description limit. ``truncated`` says at least one field was
    cut and ``warnings`` names what was cut.
    """

    model_config = ConfigDict(
        extra="allow", populate_by_name=True, validate_assignment=True
    )

    platform: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    description: str = ""
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chapters", mode="before")
    @classmethod
    def _coerce_chapters(cls, value: Any) -> list[Chapter]:
        if isinstance(value, Chapter):
            return [value]
        return list(value or ())

    @property
    def within_limits(self) -> bool:
        """Whether every chapter stayed inside its platform limits."""
        return not self.truncated

    def to_dict(self) -> dict[str, Any]:
        """Return the chapter set as a JSON-compatible mapping."""
        return self.model_dump(mode="json")


def _collect_words(transcript: Any) -> list[WordTimestamp]:
    """Return the timed words carried by *transcript*."""
    if isinstance(transcript, STTResponse):
        return list(transcript.all_words)
    if isinstance(transcript, Sequence) and not isinstance(
        transcript, (str, bytes, bytearray)
    ):
        words: list[WordTimestamp] = []
        for item in transcript:
            if isinstance(item, WordTimestamp):
                words.append(item)
            elif isinstance(item, Mapping):
                words.append(WordTimestamp.model_validate(item))
            else:
                raise ConfigError(
                    "transcript words must be WordTimestamp values or mappings",
                    details={"field": "transcript"},
                )
        return words
    if isinstance(transcript, Mapping):
        if "words" in transcript:
            return _collect_words(transcript["words"])
        raise ConfigError(
            "transcript mapping has no words",
            details={"field": "transcript"},
        )
    raise ConfigError(
        "transcript must be a speech to text response or a list of timed words",
        details={"field": "transcript"},
    )


def _split_into_windows(
    words: Sequence[WordTimestamp], chapter_span: float
) -> list[list[WordTimestamp]]:
    """Split *words* into fixed duration windows starting at the first word."""
    if chapter_span <= 0:
        return [list(words)]
    windows: list[list[WordTimestamp]] = []
    index = 0
    total = len(words)
    while index < total:
        window_start = words[index].start_seconds
        limit = window_start + chapter_span
        end = index
        while end < total and words[end].start_seconds < limit:
            end += 1
        windows.append(list(words[index:end]))
        index = end
    return windows


def _chapter_title(words: Sequence[WordTimestamp], max_length: int | None) -> str:
    text = _clean(" ".join(word.word for word in words))
    parts = _SENTENCE_BREAK.split(text)
    title = _clean(parts[0]) if parts else ""
    if not title:
        title = text
    return _truncate(title, max_length)


def _chapter_duration(
    chapter_start: float,
    next_start: float | None,
    transcript_duration: float | None,
    last_word_end: float,
) -> float:
    if next_start is not None:
        return next_start - chapter_start
    if transcript_duration is not None:
        return transcript_duration - chapter_start
    return last_word_end - chapter_start


def _candidate_chapters(
    words: Sequence[WordTimestamp],
    chapter_span: float,
    transcript_duration: float | None,
) -> list[tuple[float, float, Sequence[WordTimestamp]]]:
    """Split *words* into windows and return (start, span, window) triples."""
    last_word_end = max(word.end_seconds for word in words)
    windows = _split_into_windows(words, chapter_span)
    candidates: list[tuple[float, float, Sequence[WordTimestamp]]] = []
    for index, window in enumerate(windows):
        if not window:
            continue
        start = window[0].start_seconds
        next_start: float | None = None
        for later in windows[index + 1 :]:
            if later:
                next_start = later[0].start_seconds
                break
        chapter_span = _chapter_duration(
            start, next_start, transcript_duration, last_word_end
        )
        if chapter_span <= 0:
            continue
        candidates.append((start, chapter_span, window))
    return candidates


class ChapterGenerator:
    """Build description chapters for a transcript from platform chapter rules.

    A platform may be passed as a :class:`Platform`, as its name when a catalog
    is supplied, or as a name resolved from the packaged platform data.
    """

    def __init__(
        self,
        platform: Platform | str | None = None,
        *,
        platforms: PlatformCatalog | Mapping[str, Any] | None = None,
        target_chapter_seconds: float = 60.0,
        config_dir: Any = None,
    ) -> None:
        self.platform = _resolve_platform(
            platform,
            platforms=platforms,
            config_dir=config_dir,
        )
        self.target_chapter_seconds = target_chapter_seconds

    @property
    def chapter_rules(self) -> ChapterRules:
        """The chapter rules of the selected platform."""
        return self.platform.chapter_rules

    def build(
        self,
        transcript: Any,
        *,
        base_description: str = "",
    ) -> ChapterSet:
        """Build the chapter set for *transcript*, applying platform limits.

        Chapters are generated from the timed words, capped by the platform
        chapter count and title limits, then embedded in *base_description*
        and fitted to the platform description limit.
        """
        rules = self.chapter_rules
        caption_limits = self.platform.caption_limits
        description_limit = caption_limits.max_description_length
        if description_limit is None:
            description_limit = caption_limits.max_length

        warnings: list[str] = []
        truncated = False

        if not rules.enabled:
            return ChapterSet(
                platform=self.platform.id,
                description=_fit_description(base_description, description_limit),
                truncated=truncated,
                warnings=warnings,
                limits=self._limits(),
            )

        words = _collect_words(transcript)
        chapters: list[Chapter] = []
        if words:
            transcript_duration = _transcript_duration(transcript)
            candidates = _candidate_chapters(
                words, self.target_chapter_seconds, transcript_duration
            )
            # Drop chapters shorter than the platform minimum duration, but keep
            # at least one chapter when there are words to describe.
            kept = [
                (start, window)
                for (start, chapter_span, window) in candidates
                if not (
                    rules.min_chapter_duration_seconds is not None
                    and chapter_span < rules.min_chapter_duration_seconds
                )
            ]
            if not kept and candidates:
                kept = [(candidates[0][0], candidates[0][2])]
            chapters = [
                Chapter(
                    start_seconds=start,
                    title=_chapter_title(window, rules.max_chapter_title_length),
                )
                for (start, window) in kept
            ]

        if rules.max_chapters is not None and len(chapters) > rules.max_chapters:
            chapters = chapters[: rules.max_chapters]
            warnings.append("chapters capped to max_chapters")
            truncated = True

        chapters_text = _format_chapters(chapters, rules.separator)
        full = (
            _join([base_description, chapters_text])
            if base_description
            else chapters_text
        )
        description, cut = _fit_description_to(full, description_limit)
        if cut:
            warnings.append("description truncated")
            truncated = True

        return ChapterSet(
            platform=self.platform.id,
            chapters=chapters,
            description=description,
            truncated=truncated,
            warnings=warnings,
            limits=self._limits(),
        )

    def build_strict(
        self,
        transcript: Any,
        *,
        base_description: str = "",
    ) -> ChapterSet:
        """Build chapters and refuse any platform rule violation.

        Unlike :meth:`build`, nothing is silently capped or truncated: exceeding
        the chapter count, a chapter shorter than the minimum duration, a title
        over the limit or a description over the limit raises
        :class:`~aivia.core.errors.ContentPolicyError`.
        """
        rules = self.chapter_rules
        result = self.build(transcript, base_description=base_description)
        words = _collect_words(transcript) if rules.enabled else []
        if words:
            transcript_duration = _transcript_duration(transcript)
            candidates = _candidate_chapters(
                words, self.target_chapter_seconds, transcript_duration
            )
        else:
            candidates = []

        if rules.max_chapters is not None and len(candidates) > rules.max_chapters:
            raise ContentPolicyError(
                "too many chapters for platform",
                details={
                    "chapter_count": len(candidates),
                    "max_chapters": rules.max_chapters,
                    "platform": self.platform.id,
                },
            )
        last_word_end = max((word.end_seconds for word in words), default=0.0)
        transcript_duration = _transcript_duration(transcript)
        for position, (start, _chapter_span, window) in enumerate(candidates):
            next_start: float | None = None
            for later in candidates[position + 1 :]:
                if later:
                    next_start = later[0]
                    break
            chapter_duration = _chapter_duration(
                start, next_start, transcript_duration, last_word_end
            )
            title = _chapter_title(window, rules.max_chapter_title_length)
            chapter = Chapter(start_seconds=start, title=title)
            if (
                rules.min_chapter_duration_seconds is not None
                and chapter_duration < rules.min_chapter_duration_seconds
            ):
                raise ContentPolicyError(
                    "chapter shorter than the platform minimum duration",
                    details={
                        "chapter": chapter.label,
                        "duration_seconds": chapter_duration,
                        "min_chapter_duration_seconds": (
                            rules.min_chapter_duration_seconds
                        ),
                        "platform": self.platform.id,
                    },
                )
            if (
                rules.max_chapter_title_length is not None
                and len(chapter.title) > rules.max_chapter_title_length
            ):
                raise ContentPolicyError(
                    "chapter title exceeds the platform limit",
                    details={
                        "chapter": chapter.label,
                        "title_length": len(chapter.title),
                        "max_chapter_title_length": rules.max_chapter_title_length,
                        "platform": self.platform.id,
                    },
                )
        description_limit = self.platform.caption_limits.max_description_length
        if description_limit is None:
            description_limit = self.platform.caption_limits.max_length
        chapters_text = _format_chapters(
            [
                Chapter(
                    start_seconds=s,
                    title=_chapter_title(w, rules.max_chapter_title_length),
                )
                for (s, _, w) in candidates
            ],
            rules.separator,
        )
        full = (
            _join([base_description, chapters_text])
            if base_description
            else chapters_text
        )
        if description_limit is not None and len(full) > description_limit:
            raise ContentPolicyError(
                "description with chapters exceeds the platform limit",
                details={
                    "description_length": len(full),
                    "max_description_length": description_limit,
                    "platform": self.platform.id,
                },
            )
        return result

    def _limits(self) -> dict[str, Any]:
        rules = self.chapter_rules
        limits = self.platform.caption_limits
        return {
            "max_chapters": rules.max_chapters,
            "min_chapter_duration_seconds": rules.min_chapter_duration_seconds,
            "max_chapter_title_length": rules.max_chapter_title_length,
            "max_description_length": limits.max_description_length,
            "max_length": limits.max_length,
        }

    generate = build
    apply = build
    __call__ = build


def _fit_description(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    if not limit:
        return ""
    return text[: limit - len(_ELLIPSIS)].rstrip() + _ELLIPSIS


def _fit_description_to(text: str, limit: int | None) -> tuple[str, bool]:
    if limit is None or len(text) <= limit:
        return text, False
    return _fit_description(text, limit), True


def _format_chapters(chapters: Sequence[Chapter], separator: str) -> str:
    return separator.join(chapter.label for chapter in chapters)


def _transcript_duration(transcript: Any) -> float | None:
    if isinstance(transcript, STTResponse):
        return transcript.duration_seconds
    return None


def _resolve_platform(
    value: Platform | str | None,
    *,
    platforms: PlatformCatalog | Mapping[str, Any] | None,
    config_dir: Any,
) -> Platform:
    if isinstance(value, Platform):
        return value
    if value is None:
        raise ConfigError(
            "chapters need a platform",
            details={"platform": None},
        )
    catalog: PlatformCatalog | None
    if platforms is None:
        catalog = None
    elif isinstance(platforms, PlatformCatalog):
        catalog = platforms
    elif isinstance(platforms, Mapping):
        catalog = PlatformCatalog.from_mapping(platforms)
    else:
        raise ConfigError(
            "platforms must be a catalog or a mapping",
            details={"platforms": type(platforms).__name__},
        )
    if catalog is None and not isinstance(value, str):
        raise ConfigError(
            "platform must be a name or a Platform",
            details={"platform": type(value).__name__},
        )
    return get_platform(value, catalog=catalog, config_dir=config_dir)


def build_chapters(
    transcript: Any,
    platform: Platform | str | None = None,
    **kwargs: Any,
) -> ChapterSet:
    """Return the chapter set for *transcript* on *platform*."""
    return ChapterGenerator(platform, **kwargs).build(transcript)


generate_chapters = build_chapters
