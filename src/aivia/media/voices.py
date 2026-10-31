"""Resolve a voice for each language, list configured voices and pass hints.

A content profile, a script and a character may all name a voice differently.
The voice stage needs one deterministic answer for a requested language even
when that language has no dedicated entry, so a catalog keeps an explicit
fallback chain:

* an explicit voice on the request always wins;
* an exact language entry is preferred;
* a configured fallback for that language is followed next;
* the base language (``de-AT`` -> ``de``) is tried before the catalog default;
* a default voice is the last configured fallback.

The catalog never calls a provider. It only turns configuration into a voice
id, so the same selection is reproducible offline. Pronunciation hints are
carried on the request and passed on only to providers that declare the
``pronunciation`` option; other adapters must not silently receive a field they
cannot consume.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from aivia.core.errors import ConfigError, InvalidRequestError
from aivia.providers.protocols.tts import CAPABILITY_TTS, TTSRequest
from aivia.providers.tts.common import phoneme_ssml

__all__ = [
    "DEFAULT_ALPHABET",
    "LanguageVoice",
    "LanguageVoices",
    "Pronunciation",
    "PronunciationHint",
    "Voice",
    "VoiceCatalog",
    "VoiceEntry",
    "VoiceList",
    "VoiceMapping",
    "VoiceResolver",
    "VoiceSelection",
    "apply_pronunciation",
    "available_voices",
    "build_voice_catalog",
    "list_available_voices",
    "list_voices",
    "normalize_pronunciation",
    "parse_pronunciation",
    "phoneme_ssml",
    "prepare_speech_request",
    "pronunciation_hints",
    "pronunciation_supported",
    "resolve_language_voice",
    "resolve_voice",
    "select_voice",
    "supports_pronunciation",
    "voice_catalog",
    "voice_for_language",
    "voices_for_language",
    "with_pronunciation",
]

DEFAULT_ALPHABET = "ipa"
"""The phonetic alphabet used when a hint does not name one."""

_PRONUNCIATION_TOKENS = frozenset(
    {"pronunciation", "pronunciations", "phoneme", "phonemes", "ssml"}
)
_RESERVED_KEYS = frozenset(
    {
        "default",
        "fallback",
        "fallbacks",
        "language",
        "languages",
        "provider",
        "pronunciation",
        "voices",
    }
)


def _language_key(value: str) -> str:
    """Normalize a language tag to a lower-case, hyphen-separated key."""
    return value.strip().replace("_", "-").lower()


def _base_language(value: str) -> str:
    return value.split("-", 1)[0]


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


class Voice(BaseModel):
    """One configured voice and the language it speaks.

    ``voice`` is the provider voice id or a logical alias an adapter resolves
    later. ``language`` is optional so a catalog entry may be a global default.
    The model is frozen because a selected voice is copied into requests and
    cache keys.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    voice: str = Field(
        validation_alias=AliasChoices(
            "voice",
            "id",
            "voice_id",
            "name",
            "value",
        )
    )
    language: str = Field(
        default="",
        validation_alias=AliasChoices("language", "lang", "locale"),
    )
    provider: str = ""
    fallback: str = ""
    pronunciation: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"voice": data}
        return data

    @field_validator("voice", mode="before")
    @classmethod
    def _validate_voice(cls, value: Any) -> Any:
        return _require_text(value, field="voice")

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, value: Any) -> Any:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise TypeError("language must be a string")
        return _language_key(value)

    @field_validator("provider", "fallback", mode="before")
    @classmethod
    def _validate_optional_text(cls, value: Any) -> Any:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise TypeError("voice fields must be strings")
        return value.strip()

    @field_validator("pronunciation", mode="before")
    @classmethod
    def _validate_pronunciation(cls, value: Any) -> Any:
        return normalize_pronunciation(value)

    @property
    def id(self) -> str:
        """Alias for :attr:`voice`."""
        return self.voice

    @property
    def voice_id(self) -> str:
        """Alias for :attr:`voice`."""
        return self.voice

    @property
    def name(self) -> str:
        """Alias for :attr:`voice`."""
        return self.voice

    @property
    def alias(self) -> str:
        """Alias for :attr:`voice`."""
        return self.voice

    @property
    def label(self) -> str:
        """A stable ``language:voice`` label for display and logging."""
        return f"{self.language}:{self.voice}" if self.language else self.voice

    def __str__(self) -> str:
        return self.voice

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.voice == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash((self.voice, self.language))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Return a field value, like a mapping."""
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the voice as a plain mapping."""
        return self.model_dump()


LanguageVoice = Voice
VoiceEntry = Voice
VoiceSelection = Voice
VoiceList = list[Voice]


class VoiceCatalog:
    """An immutable-by-convention language to voice catalog with fallbacks.

    ``voices`` accepts a language mapping, a list of :class:`Voice` entries or
    a configuration mapping with ``languages``, ``voices``, ``default``,
    ``fallbacks`` and ``pronunciation`` keys. A plain string entry is treated
    as a provider voice id.
    """

    def __init__(
        self,
        voices: object | None = None,
        *,
        default: object | None = None,
        fallback: object | None = None,
        fallbacks: Mapping[str, str] | None = None,
        pronunciation: Mapping[str, str] | None = None,
        language: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._voices: dict[str, Voice] = {}
        self._default: Voice | None = None
        self._fallback_voice: Voice | None = None
        self._fallbacks: dict[str, str] = {}
        self._pronunciation = normalize_pronunciation(pronunciation)
        self._default_language = (
            _language_key(language) if isinstance(language, str) and language else ""
        )
        self._provider = provider.strip() if isinstance(provider, str) else ""
        if voices is not None:
            self._load(voices)
        if default is not None:
            if isinstance(default, str) and _language_key(default) in self._voices:
                self._default_language = _language_key(default)
            else:
                self._default = self._coerce_voice(default, language="")
        if fallback is not None:
            self._fallback_voice = self._coerce_voice(fallback, language="")
        if fallbacks is not None:
            self._merge_fallbacks(fallbacks)

    @classmethod
    def from_value(
        cls,
        value: VoiceCatalog | Mapping[str, Any] | Iterable[Any] | str | None,
        *,
        default: object | None = None,
        fallback: object | None = None,
        fallbacks: Mapping[str, str] | None = None,
        pronunciation: Mapping[str, str] | None = None,
        language: str | None = None,
        provider: str | None = None,
    ) -> VoiceCatalog:
        """Return *value* as a catalog, copying a mapping once."""
        if isinstance(value, VoiceCatalog):
            return value
        return cls(
            value,
            default=default,
            fallback=fallback,
            fallbacks=fallbacks,
            pronunciation=pronunciation,
            language=language,
            provider=provider,
        )

    @classmethod
    def from_config(
        cls,
        value: object | None = None,
        **kwargs: Any,
    ) -> VoiceCatalog:
        """Alias for :meth:`from_value` for configuration callers."""
        return cls.from_value(value, **kwargs)  # type: ignore[arg-type]

    def _load(self, value: object) -> None:
        if isinstance(value, VoiceCatalog):
            self._merge(value)
            return
        if isinstance(value, Voice):
            self._add_voice(value)
            return
        if isinstance(value, str):
            self._add_voice(self._coerce_voice(value, language=self._default_language))
            return
        if isinstance(value, Mapping):
            self._load_mapping(value)
            return
        if isinstance(value, Iterable):
            for entry in value:
                if isinstance(entry, Voice):
                    self._add_voice(entry)
                elif isinstance(entry, Mapping):
                    self._add_voice(self._coerce_voice(entry, language=""))
                elif isinstance(entry, str):
                    self._add_voice(
                        self._coerce_voice(entry, language=self._default_language)
                    )
                else:
                    raise TypeError("voice entries must be strings or mappings")
            return
        raise TypeError("voices must be a mapping, an iterable or a catalog")

    def _load_mapping(self, value: Mapping[str, Any]) -> None:
        data = dict(value)
        keys = {key.casefold() for key in data if isinstance(key, str)}
        if keys & _RESERVED_KEYS:
            default = data.pop("default", None)
            fallback = data.pop("fallback", None)
            fallbacks = data.pop("fallbacks", None)
            language = data.pop("language", None)
            provider = data.pop("provider", None)
            pronunciation = data.pop("pronunciation", None)
            nested = data.pop("languages", None)
            voices = data.pop("voices", None)
            if isinstance(language, str) and language:
                self._default_language = _language_key(language)
            if isinstance(provider, str) and provider:
                self._provider = provider.strip()
            self._pronunciation = normalize_pronunciation(pronunciation)
            if nested is not None:
                self._load(nested)
            if voices is not None:
                self._load(voices)
            for key, entry in data.items():
                self._add_entry(str(key), entry)
            if default is not None:
                if isinstance(default, str) and _language_key(default) in self._voices:
                    self._default_language = _language_key(default)
                else:
                    self._default = self._coerce_voice(default, language="")
            if fallback is not None:
                self._fallback_voice = self._coerce_voice(fallback, language="")
            if fallbacks is not None:
                if not isinstance(fallbacks, Mapping):
                    raise TypeError("fallbacks must be a mapping")
                self._merge_fallbacks(fallbacks)
            return
        for key, entry in data.items():
            self._add_entry(str(key), entry)

    def _merge(self, other: VoiceCatalog) -> None:
        for key, voice in other._voices.items():
            self._voices.setdefault(key, voice)
        if other._default is not None:
            self._default = other._default
        if other._fallback_voice is not None:
            self._fallback_voice = other._fallback_voice
        self._fallbacks.update(other._fallbacks)
        self._pronunciation = {**other._pronunciation, **self._pronunciation}
        if other._default_language:
            self._default_language = other._default_language
        if other._provider:
            self._provider = other._provider

    def _merge_fallbacks(self, fallbacks: Mapping[str, str]) -> None:
        for raw_key, raw_value in fallbacks.items():
            key = _language_key(_require_text(raw_key, field="fallback language"))
            value = _require_text(raw_value, field="fallback voice")
            self._fallbacks[key] = value

    def _add_entry(self, language: str, entry: object) -> None:
        voice = self._coerce_voice(entry, language=language)
        self._add_voice(voice)

    def _coerce_voice(self, entry: object, *, language: str) -> Voice:
        if isinstance(entry, Voice):
            if entry.language or not language:
                return entry
            return entry.model_copy(update={"language": _language_key(language)})
        if isinstance(entry, str):
            return Voice(
                voice=entry,
                language=_language_key(language) if language else "",
                provider=self._provider,
                pronunciation=self._pronunciation,
            )
        if isinstance(entry, Mapping):
            data = dict(entry)
            if language and not any(
                key in data for key in ("language", "lang", "locale")
            ):
                data["language"] = language
            data.setdefault("provider", self._provider)
            data.setdefault("pronunciation", self._pronunciation)
            return Voice.model_validate(data)
        raise TypeError("a voice entry must be a string or a mapping")

    def _add_voice(self, voice: Voice) -> None:
        key = _language_key(voice.language)
        if not key:
            if self._default is None:
                self._default = voice
            else:
                self._voices[""] = voice
            return
        if key in self._voices:
            raise ConfigError(
                f"duplicate voice for language {voice.language!r}",
                details={"language": voice.language},
            )
        self._voices[key] = voice

    def _candidate_keys(self, language: str) -> Iterator[str]:
        key = _language_key(language)
        seen: set[str] = set()
        while key and key not in seen:
            seen.add(key)
            yield key
            target = self._fallbacks.get(key)
            if target is not None:
                target_key = _language_key(target)
                if target_key in self._voices:
                    key = target_key
                    continue
            base = _base_language(key)
            if base and base != key:
                key = base
                continue
            break

    def select(
        self,
        language: str | None = None,
        *,
        fallback: str | None = None,
    ) -> Voice:
        """Return the :class:`Voice` chosen for *language*.

        A language may fall back to a configured target, its base language and
        finally the catalog default. A cycle in the fallback chain is rejected
        instead of looping forever.
        """
        requested = (
            _language_key(language)
            if isinstance(language, str) and language.strip()
            else self._default_language
        )
        visited: set[str] = set()
        for key in self._candidate_keys(requested):
            if key in visited:
                raise ConfigError(
                    "voice fallback cycle",
                    details={"language": requested},
                )
            visited.add(key)
            voice = self._voices.get(key)
            if voice is not None:
                return voice
            target = self._fallbacks.get(key)
            if target is not None:
                target_key = _language_key(target)
                if target_key not in self._voices:
                    return Voice(
                        voice=target,
                        language=key,
                        provider=self._provider,
                        pronunciation=self._pronunciation,
                    )
        if self._default is not None:
            return self._default
        if fallback is not None:
            return self._coerce_voice(fallback, language=requested)
        if self._fallback_voice is not None:
            return self._fallback_voice
        detail = requested or "default"
        raise InvalidRequestError(
            f"no voice is configured for language {detail!r}",
            details={"language": detail},
        )

    def resolve(
        self,
        language: str | None = None,
        voice: str | None = None,
        *,
        fallback: str | None = None,
    ) -> str:
        """Return the provider voice id for *language*.

        An explicit non-empty *voice* wins so a caller may bypass the catalog.
        """
        if voice is not None:
            if not isinstance(voice, str):
                raise TypeError("voice must be a string")
            explicit = voice.strip()
            if explicit:
                return explicit
        return self.select(language, fallback=fallback).voice

    def resolve_voice(
        self,
        language: str | None = None,
        voice: str | None = None,
        *,
        fallback: str | None = None,
    ) -> str:
        """Alias for :meth:`resolve`."""
        return self.resolve(language, voice, fallback=fallback)

    def list(self, language: str | None = None) -> list[Voice]:
        """Return configured voices ordered by language and voice id."""
        entries: list[Voice] = []
        if language is None:
            entries.extend(self._voices.values())
            if self._default is not None:
                entries.append(self._default)
            if self._fallback_voice is not None:
                entries.append(self._fallback_voice)
        else:
            requested = _language_key(language)
            for key in (requested, _base_language(requested)):
                voice = self._voices.get(key)
                if voice is not None and voice not in entries:
                    entries.append(voice)
            if not entries and self._default is not None:
                entries.append(self._default)
        unique: dict[tuple[str, str], Voice] = {}
        for voice in entries:
            unique[(voice.language, voice.voice)] = voice
        return sorted(unique.values(), key=lambda item: (item.language, item.voice))

    def list_voices(self, language: str | None = None) -> builtins.list[Voice]:
        """Alias for :meth:`list`."""
        return self.list(language)

    def voices_for_language(self, language: str) -> builtins.list[Voice]:
        """Alias for :meth:`list` with a language argument."""
        return self.list(language)

    @property
    def voices(self) -> dict[str, Voice]:
        """A copy of the language to voice mapping."""
        return dict(self._voices)

    @property
    def languages(self) -> dict[str, Voice]:
        """Alias for :attr:`voices`."""
        return self.voices

    @property
    def entries(self) -> tuple[Voice, ...]:
        """Every configured voice, including defaults, as a tuple."""
        return tuple(self.list())

    @property
    def default(self) -> Voice | None:
        """The voice used when no language matches."""
        return self._default

    @property
    def default_language(self) -> str:
        """The language tried when a request does not name one."""
        return self._default_language

    @property
    def fallbacks(self) -> dict[str, str]:
        """A copy of the configured language fallback targets."""
        return dict(self._fallbacks)

    @property
    def pronunciation(self) -> dict[str, str]:
        """A copy of the catalog-wide pronunciation hints."""
        return dict(self._pronunciation)

    def __contains__(self, language: object) -> bool:
        return isinstance(language, str) and _language_key(language) in self._voices

    def __len__(self) -> int:
        return len(self._voices)

    def __iter__(self) -> Iterator[Voice]:
        return iter(self.list())


VoiceResolver = VoiceCatalog
LanguageVoices = VoiceCatalog
VoiceMapping = VoiceCatalog


def voice_catalog(value: object | None = None, **kwargs: Any) -> VoiceCatalog:
    """Build a :class:`VoiceCatalog` from *value*."""
    return VoiceCatalog.from_value(value, **kwargs)  # type: ignore[arg-type]


build_voice_catalog = voice_catalog


def resolve_voice(
    catalog: VoiceCatalog | Mapping[str, Any] | str | None,
    language: str | Mapping[str, Any] | VoiceCatalog | None = None,
    voice: str | None = None,
    *,
    fallback: str | None = None,
) -> str:
    """Resolve a voice for *language* from *catalog*.

    The first two arguments may be swapped so ``resolve_voice("de", catalog)``
    and ``resolve_voice(catalog, "de")`` both work.
    """
    if isinstance(catalog, str) and not isinstance(language, str):
        catalog, language = language, catalog
    selected = VoiceCatalog.from_value(catalog)
    return selected.resolve(
        language if isinstance(language, str) else None, voice, fallback=fallback
    )


resolve_language_voice = resolve_voice
voice_for_language = resolve_voice
select_voice = resolve_voice


def list_voices(
    catalog: VoiceCatalog | Mapping[str, Any] | str | None = None,
    language: str | None = None,
) -> list[Voice]:
    """List voices from *catalog*, optionally filtered to *language*."""
    if isinstance(catalog, str) and language is None:
        language, catalog = catalog, None
    return VoiceCatalog.from_value(catalog).list(language)


available_voices = list_voices
list_available_voices = list_voices
voices_for_language = list_voices


class PronunciationHint(BaseModel):
    """One word and the phonetic spelling a provider should say instead."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    word: str = Field(validation_alias=AliasChoices("word", "text", "term", "grapheme"))
    phonetic: str = Field(
        validation_alias=AliasChoices(
            "phonetic",
            "phoneme",
            "pronunciation",
            "ipa",
            "value",
        )
    )
    alphabet: str = DEFAULT_ALPHABET
    language: str = ""

    @field_validator("word", "phonetic", mode="before")
    @classmethod
    def _validate_text(cls, value: Any) -> Any:
        return _require_text(value, field="pronunciation")

    @field_validator("alphabet", mode="before")
    @classmethod
    def _validate_alphabet(cls, value: Any) -> Any:
        if value is None:
            return DEFAULT_ALPHABET
        return _require_text(value, field="alphabet")

    @field_validator("language", mode="before")
    @classmethod
    def _validate_language(cls, value: Any) -> Any:
        if value is None:
            return ""
        if not isinstance(value, str):
            raise TypeError("language must be a string")
        return _language_key(value)

    @property
    def term(self) -> str:
        """Alias for :attr:`word`."""
        return self.word

    @property
    def phoneme(self) -> str:
        """Alias for :attr:`phonetic`."""
        return self.phonetic


Pronunciation = PronunciationHint


def normalize_pronunciation(value: object | None) -> dict[str, str]:
    """Return *value* as a validated word to phonetic mapping.

    A mapping, an iterable of :class:`PronunciationHint` or hint mappings is
    accepted. ``None`` becomes an empty mapping.
    """
    if value is None:
        return {}
    if isinstance(value, PronunciationHint):
        return {value.word: value.phonetic}
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _require_text(raw_key, field="pronunciation word")
            if not isinstance(raw_value, str):
                raise ValueError("pronunciation values must be strings")
            phonetic = raw_value.strip()
            if not phonetic:
                raise ValueError("pronunciation values must be non-empty")
            result[key] = phonetic
        return result
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result = {}
        for entry in value:
            hint = (
                entry
                if isinstance(entry, PronunciationHint)
                else PronunciationHint.model_validate(entry)
            )
            result[hint.word] = hint.phonetic
        return result
    raise TypeError("pronunciation must be a mapping or an iterable of hints")


parse_pronunciation = normalize_pronunciation
pronunciation_hints = normalize_pronunciation


def _metadata_supports(metadata: object) -> bool:
    options = getattr(metadata, "supported_options", None)
    if not isinstance(options, Mapping):
        return False
    tokens: object = options.get(CAPABILITY_TTS)
    if tokens is None:
        return False
    if isinstance(tokens, str):
        values = (tokens,)
    elif isinstance(tokens, Iterable):
        values = tuple(tokens)
    else:
        return False
    return any(
        isinstance(token, str) and token.strip().casefold() in _PRONUNCIATION_TOKENS
        for token in values
    )


def supports_pronunciation(provider: object | None) -> bool:
    """Return whether *provider* declares pronunciation hint support.

    Support is read from ``ProviderMeta.supported_options`` for the ``tts``
    capability, so no vendor name is hard-coded outside an adapter.
    """
    if provider is None:
        return False
    if isinstance(provider, bool):
        return provider
    flag = getattr(provider, "supports_pronunciation", None)
    if isinstance(flag, bool):
        return flag
    metadata = getattr(provider, "metadata", provider)
    return _metadata_supports(metadata)


pronunciation_supported = supports_pronunciation


def apply_pronunciation(
    request: TTSRequest,
    provider: object | None = None,
    *,
    supported: bool | None = None,
    strict: bool = False,
) -> TTSRequest:
    """Return *request* with hints only when the provider can consume them.

    With ``strict=True`` an unsupported hint set raises instead of being
    dropped, which is how a caller opts out of a silent downgrade.
    """
    if not isinstance(request, TTSRequest):
        raise TypeError("request must be a TTSRequest")
    hints = normalize_pronunciation(request.pronunciation)
    if not hints:
        return request
    can_use = supported if supported is not None else supports_pronunciation(provider)
    if can_use:
        return request.model_copy(update={"pronunciation": hints})
    if strict:
        raise InvalidRequestError(
            "provider does not support pronunciation hints",
            details={"provider": getattr(provider, "name", "")},
        )
    return request.model_copy(update={"pronunciation": {}})


with_pronunciation = apply_pronunciation


def prepare_speech_request(
    request: TTSRequest,
    catalog: VoiceCatalog | Mapping[str, Any] | None = None,
    *,
    language: str | None = None,
    voice: str | None = None,
    provider: object | None = None,
    fallback: str | None = None,
    strict: bool = False,
) -> TTSRequest:
    """Resolve the request voice and pass hints when the provider supports it.

    An explicit *voice* argument or a non-empty ``request.voice`` overrides the
    catalog; pass ``voice=""`` to force the language lookup.
    """
    if not isinstance(request, TTSRequest):
        raise TypeError("request must be a TTSRequest")
    updates: dict[str, Any] = {}
    if catalog is not None:
        selected = VoiceCatalog.from_value(catalog)
        updates["voice"] = selected.resolve(
            language if language is not None else request.language,
            voice if voice is not None else request.voice,
            fallback=fallback,
        )
        if language is not None:
            updates["language"] = language
    prepared = request.model_copy(update=updates) if updates else request
    return apply_pronunciation(prepared, provider, strict=strict)
