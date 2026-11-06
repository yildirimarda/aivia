"""Build a music prompt from a content profile and explicit style filters.

This module keeps the profile driven prompt helpers importable under the name
of the feature. The implementation lives in :mod:`aivia.audio.music_select`
alongside the music stage that sends the prompt to a provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aivia.audio.music_select import (
    build_music_prompt,
    build_music_prompt_from_profile,
    build_profile_music_prompt,
    music_prompt,
    music_prompt_for_profile,
    music_prompt_from_profile,
    profile_music_prompt,
)

__all__ = [
    "MusicPromptBuilder",
    "MusicPromptFactory",
    "MusicPromptGenerator",
    "ProfileMusicPrompt",
    "build_music_prompt",
    "build_music_prompt_for_profile",
    "build_music_prompt_from_profile",
    "build_profile_music_prompt",
    "generate_music_prompt",
    "music_prompt",
    "music_prompt_for_profile",
    "music_prompt_from_profile",
    "profile_music_prompt",
]


class MusicPromptBuilder:
    """Build a deterministic music prompt for one profile.

    Explicit values passed to :meth:`build` override the profile settings, and
    an explicit ``prompt`` wins over everything. The builder never calls a
    model, reads the network or chooses a provider.
    """

    def __init__(
        self,
        profile: Any = None,
        *,
        instrumental: bool | None = None,
        **defaults: Any,
    ) -> None:
        self.profile = profile
        self.instrumental = instrumental
        self.defaults = dict(defaults)

    def build(self, **values: Any) -> str:
        """Return the prompt for this profile with *values* taking precedence."""
        options: dict[str, Any] = dict(self.defaults)
        options.update(values)
        if self.instrumental is not None and options.get("instrumental") is None:
            options["instrumental"] = self.instrumental
        return build_music_prompt(self.profile, **options)

    __call__ = build
    prompt = build
    generate = build

    def to_dict(self, **values: Any) -> Mapping[str, Any]:
        """Return the prompt with the values used to build it."""
        prompt = self.build(**values)
        return {"prompt": prompt, **values}


MusicPromptGenerator = MusicPromptBuilder
MusicPromptFactory = MusicPromptBuilder
ProfileMusicPrompt = MusicPromptBuilder
generate_music_prompt = build_music_prompt
build_music_prompt_for_profile = build_profile_music_prompt
