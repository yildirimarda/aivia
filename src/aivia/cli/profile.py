"""``aivia profile``: inspect a content profile and its derived brief settings."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

import yaml

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.core.config import ContentProfileConfig
from aivia.core.config.masking import mask_secrets
from aivia.core.config.resolve import resolve_config
from aivia.core.errors import ConfigError
from aivia.providers.protocols.llm import CAPABILITY_LLM
from aivia.providers.provider_factory import build_providers
from aivia.research.brief import BriefDerivation
from aivia.research.derived import BriefDerivationStore, ensure_derivation

__all__ = [
    "configure",
    "derive_profile_settings",
    "run_profile",
    "run_show",
]


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


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the profile command and its ``show`` subcommand."""
    subparsers = parser.add_subparsers(dest="profile_command", metavar="COMMAND")
    show = subparsers.add_parser(
        "show",
        help="Show a content profile, optionally with its derived settings",
        description="Show a content profile, optionally with its derived settings",
    )
    _configure_show(show)


def run_profile(args: argparse.Namespace) -> int:
    """Dispatch to the selected profile subcommand."""
    handler = getattr(args, "_profile_handler", None)
    if handler is None:
        print("error: missing profile command", file=sys.stderr)
        print("usage: aivia profile {show}", file=sys.stderr)
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
