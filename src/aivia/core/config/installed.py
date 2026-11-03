"""Register the config sections owned by installed optional modules.

Optional modules are packages under ``aivia`` (for example ``aivia.research``).
The core never imports them; it only checks whether the package is installed
before registering the section model the core already describes. This keeps a
core-only installation valid while making an installed module's settings
available to ``config validate`` and the generated schema.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from aivia.core.config.registry import SectionRegistry, registry
from aivia.core.config.research import register_research_section

__all__ = [
    "INSTALLED_MODULE_REGISTRARS",
    "installed_module_names",
    "register_installed_sections",
]

INSTALLED_MODULE_REGISTRARS: dict[str, Callable[[SectionRegistry], SectionRegistry]] = {
    "research": register_research_section,
}


def _is_installed(name: str) -> bool:
    """Return whether the optional module package is installed.

    ``find_spec`` resolves the package without importing it, so the core does
    not depend on the module's code.
    """
    try:
        return importlib.util.find_spec(f"aivia.{name}") is not None
    except (ImportError, ValueError):
        return False


def installed_module_names() -> tuple[str, ...]:
    """Return the sorted names of installed optional modules with sections."""
    return tuple(
        name for name in sorted(INSTALLED_MODULE_REGISTRARS) if _is_installed(name)
    )


def register_installed_sections(
    sections: SectionRegistry | None = None,
) -> SectionRegistry:
    """Register every installed module's section on *sections* and return it.

    Registration is idempotent, so calling this before several commands or
    twice in one process is safe.
    """
    target = registry if sections is None else sections
    for name in installed_module_names():
        INSTALLED_MODULE_REGISTRARS[name](target)
    return target
