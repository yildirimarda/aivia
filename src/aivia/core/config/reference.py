"""Reference configuration: similarity, gates, regeneration and downloads.

A reference is a user-supplied example whose structure and style are learned
and then reproduced with new content. ``similarity_level`` selects how much of
the extracted specification is enforced (``low``, ``medium`` or ``high``).
``visual_threshold`` and ``text_threshold`` are the maximum allowed similarity
to the reference: a candidate above either gate is regenerated, up to
``max_regenerations``, and then refused rather than shipped as a copy.

``download_policy`` controls remote references. ``mode`` is ``allow`` (fetch
without asking), ``ask`` (confirm first) or ``deny`` (local files only).
``allowed_hosts`` is an optional allowlist, and an empty list leaves host
validation to the shared SSRF guard. Size, timeout, redirect and cache limits
bound every fetch so a slow or oversized response cannot stall a run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

from aivia.core.config.base import ConfigModel

__all__ = [
    "DEFAULT_NGRAM_SIMILARITY_THRESHOLD",
    "DEFAULT_TEXT_SIMILARITY_THRESHOLD",
    "DEFAULT_TITLE_SIMILARITY_THRESHOLD",
    "DEFAULT_VIDEO_FRAME_CAP",
    "DEFAULT_VIDEO_SCENE_MATCH_TOLERANCE_SECONDS",
    "DEFAULT_VISUAL_SIMILARITY_THRESHOLD",
    "DownloadPolicy",
    "ReferenceConfig",
    "ReferenceDownloadConfig",
    "ReferenceDownloadPolicy",
    "SimilarityLevel",
]

DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.85
"""Default maximum text similarity to a reference before regeneration."""

DEFAULT_VISUAL_SIMILARITY_THRESHOLD = 0.9
"""Default maximum visual similarity to a reference before regeneration."""

DEFAULT_TITLE_SIMILARITY_THRESHOLD = 0.85
"""Default title similarity at or above which two titles are the same story."""

DEFAULT_NGRAM_SIMILARITY_THRESHOLD = 0.6
"""Default word-trigram overlap at or above which wording is considered reused."""

DEFAULT_VIDEO_FRAME_CAP = 12
"""The most frames one video reference may send to a vision model."""

DEFAULT_VIDEO_SCENE_MATCH_TOLERANCE_SECONDS = 0.75
"""How far a vision scene start may drift from the measured cut, in seconds."""


class SimilarityLevel(StrEnum):
    """How closely generated content should follow the reference."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_SIMILARITY_ALIASES = {
    "low": SimilarityLevel.LOW,
    "loose": SimilarityLevel.LOW,
    "minimal": SimilarityLevel.LOW,
    "medium": SimilarityLevel.MEDIUM,
    "balanced": SimilarityLevel.MEDIUM,
    "default": SimilarityLevel.MEDIUM,
    "high": SimilarityLevel.HIGH,
    "strict": SimilarityLevel.HIGH,
    "close": SimilarityLevel.HIGH,
}


class DownloadPolicy(StrEnum):
    """Whether remote references may be downloaded."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_DOWNLOAD_ALIASES = {
    "allow": DownloadPolicy.ALLOW,
    "allowed": DownloadPolicy.ALLOW,
    "always": DownloadPolicy.ALLOW,
    "download": DownloadPolicy.ALLOW,
    "on": DownloadPolicy.ALLOW,
    "permit": DownloadPolicy.ALLOW,
    "yes": DownloadPolicy.ALLOW,
    "ask": DownloadPolicy.ASK,
    "confirm": DownloadPolicy.ASK,
    "on_demand": DownloadPolicy.ASK,
    "prompt": DownloadPolicy.ASK,
    "deny": DownloadPolicy.DENY,
    "block": DownloadPolicy.DENY,
    "never": DownloadPolicy.DENY,
    "no": DownloadPolicy.DENY,
    "off": DownloadPolicy.DENY,
}

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "similarity_level": ("similarity", "level", "match_level"),
    "visual_threshold": (
        "visual_similarity_threshold",
        "visual_limit",
        "max_visual_similarity",
    ),
    "text_threshold": (
        "text_similarity_threshold",
        "text_limit",
        "max_text_similarity",
    ),
    "max_regenerations": (
        "regeneration_limit",
        "regenerate_limit",
        "max_retries",
        "regenerations",
        "retries",
    ),
    "download_policy": (
        "download",
        "downloads",
        "remote_downloads",
        "fetch_policy",
    ),
}


class ReferenceDownloadConfig(ConfigModel):
    """Limits and policy for fetching remote reference URLs.

    ``mode`` decides whether a remote reference is fetched, confirmed or
    rejected. ``max_bytes`` caps the response body, ``timeout_seconds`` caps a
    single request, and ``max_redirects`` bounds redirect chains when
    ``follow_redirects`` is enabled. A cached download is reused unless
    ``cache`` is disabled.
    """

    mode: DownloadPolicy = DownloadPolicy.ALLOW
    max_bytes: int = Field(default=52_428_800, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)
    allowed_hosts: list[str] = Field(default_factory=list)
    follow_redirects: bool = True
    max_redirects: int = Field(default=5, ge=0)
    cache: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, data: Any) -> Any:
        if isinstance(data, bool):
            return {"mode": data}
        if isinstance(data, str):
            return {"mode": data}
        return data

    @field_validator("mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        if isinstance(value, DownloadPolicy):
            return value
        if isinstance(value, bool):
            return DownloadPolicy.ALLOW if value else DownloadPolicy.DENY
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _DOWNLOAD_ALIASES:
                return _DOWNLOAD_ALIASES[key]
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timeout_seconds must be finite")
        return value

    @field_validator("allowed_hosts")
    @classmethod
    def _normalise_hosts(cls, value: list[str]) -> list[str]:
        hosts: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError("allowed host names must be non-empty")
            host = entry.strip().lower()
            if host in seen:
                raise ValueError(f"duplicate allowed host: {host}")
            seen.add(host)
            hosts.append(host)
        return hosts

    @property
    def enabled(self) -> bool:
        """Whether remote references may be fetched at all."""
        return self.mode is not DownloadPolicy.DENY

    @property
    def allow(self) -> bool:
        """Whether remote references are fetched without confirmation."""
        return self.mode is DownloadPolicy.ALLOW

    @property
    def ask(self) -> bool:
        """Whether fetching a remote reference requires confirmation."""
        return self.mode is DownloadPolicy.ASK


ReferenceDownloadPolicy = ReferenceDownloadConfig


class ReferenceConfig(ConfigModel):
    """Similarity level, similarity gates, regeneration cap and downloads."""

    enabled: bool = True
    similarity_level: SimilarityLevel = SimilarityLevel.MEDIUM
    visual_threshold: float = Field(
        default=DEFAULT_VISUAL_SIMILARITY_THRESHOLD, ge=0, le=1
    )
    text_threshold: float = Field(default=DEFAULT_TEXT_SIMILARITY_THRESHOLD, ge=0, le=1)
    max_regenerations: int = Field(default=2, ge=0)
    download_policy: ReferenceDownloadConfig = Field(
        default_factory=ReferenceDownloadConfig
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        for canonical, aliases in _FIELD_ALIASES.items():
            for alias in aliases:
                if alias in values:
                    value = values.pop(alias)
                    values.setdefault(canonical, value)
        if values.get("download_policy") is None:
            values["download_policy"] = {}
        return values

    @field_validator("similarity_level", mode="before")
    @classmethod
    def _normalise_similarity(cls, value: Any) -> Any:
        if isinstance(value, SimilarityLevel):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            if key in _SIMILARITY_ALIASES:
                return _SIMILARITY_ALIASES[key]
        return value

    @field_validator("download_policy", mode="before")
    @classmethod
    def _normalise_download_policy(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, (bool, str)):
            return {"mode": value}
        return value

    @field_validator("visual_threshold", "text_threshold")
    @classmethod
    def _finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("similarity threshold must be finite")
        return value

    @property
    def similarity(self) -> SimilarityLevel:
        """Alias for :attr:`similarity_level`."""
        return self.similarity_level

    @property
    def level(self) -> SimilarityLevel:
        """Alias for :attr:`similarity_level`."""
        return self.similarity_level

    @property
    def visual_limit(self) -> float:
        """Alias for :attr:`visual_threshold`."""
        return self.visual_threshold

    @property
    def text_limit(self) -> float:
        """Alias for :attr:`text_threshold`."""
        return self.text_threshold

    @property
    def regeneration_limit(self) -> int:
        """Alias for :attr:`max_regenerations`."""
        return self.max_regenerations

    @property
    def download(self) -> ReferenceDownloadConfig:
        """Alias for :attr:`download_policy`."""
        return self.download_policy

    @property
    def downloads_enabled(self) -> bool:
        """Whether remote references may be fetched at all."""
        return self.download_policy.enabled
