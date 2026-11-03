"""``aivia profile``: inspect or create a content profile."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config import ContentProfileConfig
from aivia.core.config.masking import mask_secrets
from aivia.core.config.migrations import CURRENT_SCHEMA_VERSION
from aivia.core.config.resolve import resolve_config
from aivia.core.errors import ConfigError
from aivia.providers.protocols.llm import CAPABILITY_LLM
from aivia.providers.provider_factory import build_providers
from aivia.research.brief import BriefDerivation
from aivia.research.derived import BriefDerivationStore, ensure_derivation

__all__ = [
    "SAMPLE_TOPIC_COUNT",
    "ProfileAnswers",
    "build_profile",
    "configure",
    "derive_profile_settings",
    "generate_sample_topics",
    "run_create",
    "run_profile",
    "run_show",
    "save_profile",
]

SAMPLE_TOPIC_COUNT = 10
"""The number of sample topics shown before a new profile is saved."""

_QUESTION_NICHE = "What niche or topic area is this profile for? "
_QUESTION_BRIEF = "Describe the profile's brief in one sentence: "
_QUESTION_GUIDANCE = "Keywords and tone (comma separated; put the tone after '|'): "

_SAMPLE_TEMPLATES: tuple[str, ...] = (
    "The surprising history of {subject}",
    "How {subject} actually works",
    "Five myths about {subject}, debunked",
    "What most people get wrong about {subject}",
    "A beginner's guide to {subject}",
    "The future of {subject}",
    "Why {subject} matters more than you think",
    "{subject} explained in five minutes",
    "The hidden side of {subject}",
    "Questions worth asking about {subject}",
)

_YES_ANSWERS = frozenset({"y", "yes"})


@dataclass(frozen=True, slots=True)
class ProfileAnswers:
    """The answers a profile creation chat collects from the user."""

    niche: str = ""
    brief: str = ""
    keywords: tuple[str, ...] = ()
    tone: str = ""
    format: str = ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _split_keywords(value: str) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\n,;]+", value):
        keyword = " ".join(part.split())
        key = keyword.casefold()
        if keyword and key not in seen:
            seen.add(key)
            keywords.append(keyword)
    return keywords


def _read_answer(stream: TextIO, target: TextIO, prompt: str) -> str:
    target.write(prompt)
    target.flush()
    return stream.readline().strip()


def _guidance(value: str) -> tuple[list[str], str]:
    keywords, separator, tone = value.partition("|")
    return _split_keywords(keywords), tone.strip() if separator else ""


def _answers_from_args(
    args: argparse.Namespace,
    *,
    stdin: TextIO,
    stdout: TextIO,
) -> ProfileAnswers:
    niche = _text(getattr(args, "niche", None))
    brief = _text(getattr(args, "brief", None))
    tone = _text(getattr(args, "tone", None))
    format_name = _text(getattr(args, "format", None))
    keywords = _split_keywords(
        ",".join(str(value) for value in getattr(args, "keyword_options", None) or [])
    )

    if not niche:
        niche = _read_answer(stdin, stdout, _QUESTION_NICHE)
    if not brief:
        brief = _read_answer(stdin, stdout, _QUESTION_BRIEF)
    if not keywords and not tone:
        guidance_keywords, guidance_tone = _guidance(
            _read_answer(stdin, stdout, _QUESTION_GUIDANCE)
        )
        keywords = guidance_keywords
        tone = guidance_tone

    return ProfileAnswers(
        niche=niche,
        brief=brief,
        keywords=tuple(keywords),
        tone=tone,
        format=format_name,
    )


def build_profile(answers: ProfileAnswers) -> ContentProfileConfig:
    """Validate chat answers into a content profile."""
    if not isinstance(answers, ProfileAnswers):
        raise TypeError("answers must be a ProfileAnswers")
    niche = _text(answers.niche)
    brief = _text(answers.brief)
    if not niche and not brief:
        raise ConfigError(
            "a new profile needs a niche or a brief",
            details={"niche": niche, "brief": brief},
        )
    values: dict[str, Any] = {
        "niche": niche,
        "brief": brief,
        "keywords": list(answers.keywords),
    }
    tone = _text(answers.tone)
    if tone:
        values["tone"] = tone
    format_name = _text(answers.format)
    if format_name:
        values["format"] = format_name
    return ContentProfileConfig.from_mapping(values)


def _subject(profile: ContentProfileConfig) -> str:
    subject = profile.niche or profile.brief or "your niche"
    words = " ".join(subject.split()).split()
    if len(words) > 8:
        subject = " ".join(words[:8])
    return subject.rstrip(".,;:!?")


def generate_sample_topics(
    profile: ContentProfileConfig,
    count: int = SAMPLE_TOPIC_COUNT,
) -> list[str]:
    """Return deterministic, distinct sample topics for *profile*.

    The samples are deliberately generated without a model or the network so
    the approval step costs nothing and works offline.
    """
    if not isinstance(profile, ContentProfileConfig):
        raise TypeError("profile must be a ContentProfileConfig")
    if count <= 0:
        return []
    subject = _subject(profile)
    subjects = [subject, *profile.keywords]
    topics: list[str] = []
    seen: set[str] = set()
    index = 0
    while len(topics) < count:
        template = _SAMPLE_TEMPLATES[index % len(_SAMPLE_TEMPLATES)]
        active = subjects[index % len(subjects)]
        topic = template.format(subject=active)
        key = topic.casefold()
        if key in seen:
            topic = f"{topic} ({index + 1})"
            key = topic.casefold()
        seen.add(key)
        topics.append(topic)
        index += 1
    return topics


def _print_samples(
    name: str,
    profile: ContentProfileConfig,
    topics: list[str],
) -> None:
    print(f"Profile: {name}")
    if profile.niche:
        print(f"Niche: {profile.niche}")
    if profile.brief:
        print(f"Brief: {profile.brief}")
    if profile.keywords:
        print(f"Keywords: {', '.join(profile.keywords)}")
    if profile.tone:
        print(f"Tone: {profile.tone}")
    if profile.format:
        print(f"Format: {profile.format}")
    print(f"Sample topics ({len(topics)}):")
    for index, topic in enumerate(topics, start=1):
        print(f"{index:2d}. {topic}")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {path}") from exc
    if not text.strip():
        return {}
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in config {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"config root must be a mapping: {path}")
    return dict(payload)


def _profiles_in(values: Mapping[str, Any], path: Path) -> dict[str, Any]:
    profiles = values.get("profiles", {})
    if profiles is None:
        return {}
    if not isinstance(profiles, Mapping):
        raise ConfigError(f"config profiles must be a mapping: {path}")
    return dict(profiles)


def save_profile(
    path: str | Path,
    name: str,
    profile: ContentProfileConfig,
    *,
    force: bool = False,
) -> Path:
    """Merge *profile* into the config at *path* and write it atomically."""
    target = Path(path).expanduser()
    profile_name = _text(name)
    if not profile_name:
        raise ConfigError("profile name must be a non-empty string")
    if not isinstance(profile, ContentProfileConfig):
        raise TypeError("profile must be a ContentProfileConfig")
    values = _load_config(target)
    profiles = _profiles_in(values, target)
    if profile_name in profiles and not force:
        raise ConfigError(
            f"profile already exists: {profile_name} (use --force to replace it)",
            details={"profile": profile_name},
        )
    profiles[profile_name] = profile.model_dump(
        mode="json", exclude_defaults=True, exclude_none=True
    )
    values["profiles"] = profiles
    if "schema_version" not in values:
        values = {"schema_version": CURRENT_SCHEMA_VERSION, **values}

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                values,
                handle,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def _requested_name(args: argparse.Namespace) -> str:
    positional = _text(getattr(args, "name", None))
    option = _text(getattr(args, "option_name", None))
    if positional and option and positional != option:
        raise ConfigError(
            f"conflicting profile names: {positional} and {option}",
            details={"profiles": [positional, option]},
        )
    return positional or option


def _derived_name(profile: ContentProfileConfig) -> str:
    source = profile.niche or profile.brief or "custom"
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    return slug or "custom"


def _available_name(name: str, profiles: Mapping[str, Any]) -> str:
    if name not in profiles:
        return name
    index = 2
    while f"{name}-{index}" in profiles:
        index += 1
    return f"{name}-{index}"


def _approve(name: str, *, stdin: TextIO, stdout: TextIO) -> bool:
    stdout.write(f"Save profile {name!r}? [y/N] ")
    stdout.flush()
    return stdin.readline().strip().lower() in _YES_ANSWERS


def _available_formats(config: Any) -> set[str] | None:
    section = getattr(config, "formats", None)
    formats = getattr(section, "formats", None)
    if not isinstance(formats, Mapping):
        return None
    return {str(name) for name in formats}


def _validate_profile_format(
    name: str,
    profile: ContentProfileConfig,
    config: Any,
) -> None:
    format_name = profile.format
    formats = _available_formats(config)
    if not format_name or formats is None or format_name in formats:
        return
    available = ", ".join(sorted(formats)) or "none"
    raise ConfigError(
        f"profile {name} references unknown format: {format_name} "
        f"(available: {available})",
        details={
            "profile": name,
            "format": format_name,
            "available": sorted(formats),
        },
    )


def run_create(args: argparse.Namespace) -> int:
    """Chat through a new profile, show sample topics and save on approval."""
    try:
        target = paths.resolve_paths(getattr(args, "config", None)).config_file
        values = _load_config(target)
        profiles = _profiles_in(values, target)
        requested = _requested_name(args)
        force = bool(getattr(args, "force", False))
        if requested and requested in profiles and not force:
            raise ConfigError(
                f"profile already exists: {requested} (use --force to replace it)",
                details={"profile": requested},
            )

        resolved = resolve_config(
            config_file=getattr(args, "config", None),
            environment=getattr(args, "environment", None),
            profile=getattr(args, "profile", None),
            preset=getattr(args, "preset", None),
            set_values=getattr(args, "set_values", None),
        )
        config = resolved.to_app_config()

        answers = _answers_from_args(args, stdin=sys.stdin, stdout=sys.stdout)
        profile = build_profile(answers)
        name = requested or _available_name(_derived_name(profile), profiles)
        _validate_profile_format(name, profile, config)
        topics = generate_sample_topics(profile, SAMPLE_TOPIC_COUNT)
        _print_samples(name, profile, topics)

        if bool(getattr(args, "dry_run", False)):
            print("Dry run: profile not saved.")
            return 0
        if not bool(getattr(args, "yes", False)) and not _approve(
            name,
            stdin=sys.stdin,
            stdout=sys.stdout,
        ):
            print("error: profile not saved; approval declined", file=sys.stderr)
            return 1

        save_profile(target, name, profile, force=force)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved profile: {name}")
    print(f"Config: {target}")
    return 0


def _configure_show(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="profile to show; defaults to the selected or default profile",
    )
    parser.add_argument(
        "--derived",
        action="store_true",
        help="include the brief-derived research settings",
    )
    parser.add_argument(
        "--refresh",
        "--force",
        dest="refresh",
        action="store_true",
        help="regenerate derived settings even when the brief is unchanged",
    )
    parser.set_defaults(_profile_handler=run_show)


def _configure_create(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "name",
        nargs="?",
        metavar="NAME",
        help="profile name; defaults to a slug derived from the answers",
    )
    parser.add_argument(
        "--name",
        dest="option_name",
        metavar="NAME",
        help="profile name as an option instead of a positional argument",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="ask the profile questions interactively (the default)",
    )
    parser.add_argument("--niche", default="", metavar="TEXT", help="topic area")
    parser.add_argument("--brief", default="", metavar="TEXT", help="steering brief")
    parser.add_argument(
        "--keywords",
        "--keyword",
        dest="keyword_options",
        action="append",
        default=None,
        metavar="LIST",
        help="comma separated keywords; repeatable",
    )
    parser.add_argument("--tone", default="", metavar="TEXT", help="content tone")
    parser.add_argument(
        "--format",
        default="",
        metavar="NAME",
        help="format recipe the profile should use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing profile with the same name",
    )
    parser.set_defaults(_profile_handler=run_create)


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the profile command and its ``show`` and ``create`` subcommands."""
    subparsers = parser.add_subparsers(dest="profile_command", metavar="COMMAND")
    show = subparsers.add_parser(
        "show",
        help="Show a content profile, optionally with its derived settings",
        description="Show a content profile, optionally with its derived settings",
    )
    _configure_show(show)
    create = subparsers.add_parser(
        "create",
        help="Create a profile through a short chat and approve sample topics",
        description="Create a profile through a short chat and approve sample topics",
    )
    _configure_create(create)


def run_profile(args: argparse.Namespace) -> int:
    """Dispatch to the selected profile subcommand."""
    handler = getattr(args, "_profile_handler", None)
    if handler is None:
        print("error: missing profile command", file=sys.stderr)
        print("usage: aivia profile {show,create}", file=sys.stderr)
        return 2
    return int(handler(args))


def _profiles(config: Any) -> dict[str, ContentProfileConfig]:
    section = getattr(config, "profiles", None)
    profiles = getattr(section, "profiles", None)
    return dict(profiles) if isinstance(profiles, dict) else {}


def _default_profile(config: Any, profiles: dict[str, ContentProfileConfig]) -> str:
    publish = getattr(config, "publish", None)
    configured = getattr(publish, "default_profile", "")
    if isinstance(configured, str) and configured.strip() in profiles:
        return configured.strip()
    if "default" in profiles:
        return "default"
    if profiles:
        return next(iter(profiles))
    return ""


def _selected_name(args: argparse.Namespace, config: Any) -> str:
    profiles = _profiles(config)
    positional = getattr(args, "name", None)
    option = getattr(args, "profile", None)
    name = positional.strip() if isinstance(positional, str) else ""
    selected = option.strip() if isinstance(option, str) else ""
    if name and selected and name != selected:
        raise ConfigError(
            f"conflicting profiles: {name} and {selected}",
            details={"profiles": [name, selected]},
        )
    resolved = name or selected or _default_profile(config, profiles)
    if not resolved:
        raise ConfigError("no profiles are configured")
    if resolved not in profiles:
        available = ", ".join(sorted(profiles)) or "none"
        raise ConfigError(
            f"unknown profile: {resolved} (available: {available})",
            details={"profile": resolved, "available": list(profiles)},
        )
    return resolved


def _llm_provider(config: Any) -> Any | None:
    section = getattr(config, "providers", None)
    if section is None or not getattr(section, "providers", None):
        return None
    providers = build_providers(section)
    llm = getattr(config, "llm", None)
    chain = list(getattr(llm, "chain", []) or [])
    names = [*chain, *(name for name in providers if name not in chain)]
    for name in names:
        provider = providers.get(name)
        if provider is not None and CAPABILITY_LLM in provider.capabilities:
            return provider
    return None


async def derive_profile_settings(
    name: str,
    profile: ContentProfileConfig,
    *,
    config: Any = None,
    store: BriefDerivationStore | None = None,
    force: bool = False,
    no_cache: bool = False,
) -> BriefDerivation:
    """Return the profile's derived settings, regenerating only on a brief change."""
    return await ensure_derivation(
        name,
        profile,
        store=store,
        provider=_llm_provider(config),
        force=force,
        no_cache=no_cache,
    )


def _payload(name: str, profile: ContentProfileConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {"profile": name}
    payload.update(profile.model_dump(mode="json"))
    return payload


def run_show(args: argparse.Namespace) -> int:
    """Print one content profile as YAML, optionally with derived settings."""
    try:
        resolved = resolve_config(
            config_file=args.config,
            environment=args.environment,
            profile=args.profile,
            preset=args.preset,
            set_values=args.set_values,
        )
        config = resolved.to_app_config()
        name = _selected_name(args, config)
        profile = _profiles(config)[name]
        payload = _payload(name, profile)
        if args.derived:
            store = BriefDerivationStore(
                data_dir=paths.resolve_paths(args.config).data_dir
            )
            derivation = asyncio.run(
                derive_profile_settings(
                    name,
                    profile,
                    config=config,
                    store=store,
                    force=bool(getattr(args, "refresh", False)),
                    no_cache=bool(getattr(args, "no_cache", False)),
                )
            )
            payload["derived"] = derivation.model_dump(mode="json")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        yaml.safe_dump(
            mask_secrets(payload),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        end="",
    )
    return 0


register_command(
    "profile",
    run_profile,
    help="Inspect content profiles and their derived settings",
    configure=configure,
)
