"""The Alembic environment must import every tables module.

Autogenerate compares the live database with ``target_metadata``. A tables
module that ``migrations/env.py`` never imports stays off that metadata, so a
new model would silently escape autogenerate. The modules are discovered from
the package instead of listed by hand, so this guard keeps working as tables
are added.
"""

from __future__ import annotations

import ast
import pkgutil

import aivia.core.db.tables as tables_package
from aivia.core.db import migrations_dir

_TABLES_PACKAGE = tables_package.__name__


def _module_names() -> set[str]:
    return {
        info.name
        for info in pkgutil.iter_modules(tables_package.__path__)
        if not info.name.startswith("_")
    }


def _env_source() -> str:
    return (migrations_dir() / "env.py").read_text(encoding="utf-8")


def _imported_tables_modules(source: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == _TABLES_PACKAGE:
                imported.update(alias.name for alias in node.names)
            elif module.startswith(f"{_TABLES_PACKAGE}."):
                imported.add(module[len(_TABLES_PACKAGE) + 1 :].split(".", 1)[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _TABLES_PACKAGE:
                    imported.update(_module_names())
                elif alias.name.startswith(f"{_TABLES_PACKAGE}."):
                    imported.add(
                        alias.name[len(_TABLES_PACKAGE) + 1 :].split(".", 1)[0]
                    )
    return imported


def test_the_migrations_environment_imports_every_tables_module() -> None:
    imported = _imported_tables_modules(_env_source())

    missing = _module_names() - imported

    assert missing == set(), (
        "migrations/env.py does not import these tables modules, so "
        f"autogenerate never sees their models: {sorted(missing)}"
    )


def test_the_check_detects_a_tables_module_the_environment_does_not_import() -> None:
    names = _module_names()
    assert len(names) > 1, "expected more than one tables module"
    skipped = sorted(names)[0]
    imported = ", ".join(sorted(names - {skipped}))

    missing = names - _imported_tables_modules(
        f"from {_TABLES_PACKAGE} import {imported}\n"
    )

    assert missing == {skipped}
