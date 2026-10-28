"""Every tables module must register its tables on the shared metadata.

The modules under :mod:`aivia.core.db.tables` are discovered from the package
instead of being listed by hand, so a newly added tables module is checked
automatically. A table that ends up on a private :class:`MetaData` (or a module
the package does not import) would otherwise silently disappear from
migrations.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
from types import ModuleType

from sqlalchemy import Column, Integer, MetaData, Table

import aivia.core.db.tables as tables_package
from aivia.core.db.schema import metadata


def _module_names() -> list[str]:
    return [
        info.name
        for info in pkgutil.iter_modules(tables_package.__path__)
        if not info.name.startswith("_")
    ]


def _modules() -> list[ModuleType]:
    return [
        importlib.import_module(f"{tables_package.__name__}.{name}")
        for name in _module_names()
    ]


def _tables_in(module: ModuleType) -> dict[str, Table]:
    return {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, Table) and not name.startswith("_")
    }


def _unregistered_tables(module: ModuleType) -> list[str]:
    return [
        name
        for name, table in _tables_in(module).items()
        if table.metadata is not metadata
        or metadata.tables.get(table.name) is not table
    ]


def _imported_submodules(source: str) -> set[str]:
    package = tables_package.__name__
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(f"{package}."):
                imported.add(module.rsplit(".", 1)[-1])
            elif node.level and module:
                imported.add(module.split(".", 1)[0])
            elif node.level:
                imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{package}."):
                    imported.add(alias.name.rsplit(".", 1)[-1])
    return imported


def test_the_package_imports_every_tables_module() -> None:
    source = Path(tables_package.__file__).read_text(encoding="utf-8")

    missing = set(_module_names()) - _imported_submodules(source)

    assert missing == set(), (
        "tables modules are not imported by aivia.core.db.tables, so their "
        f"tables never reach the shared metadata: {sorted(missing)}"
    )


def test_every_tables_module_registers_its_tables_on_the_shared_metadata() -> None:
    assert _modules(), "expected at least one tables module"

    unregistered = {
        module.__name__: names
        for module in _modules()
        if (names := _unregistered_tables(module))
    }

    assert unregistered == {}, (
        f"tables are not attached to aivia.core.db.schema.metadata: {unregistered}"
    )


def test_the_shared_metadata_holds_exactly_the_tables_module_tables() -> None:
    defined = {
        name: table
        for module in _modules()
        for name, table in _tables_in(module).items()
    }

    assert set(metadata.tables) == set(defined)
    for name, table in defined.items():
        assert metadata.tables[name] is table


def test_the_check_detects_a_table_on_a_private_metadata() -> None:
    private = MetaData()
    stray = Table("stray", private, Column("id", Integer, primary_key=True))
    module = ModuleType("fake_tables")
    module.__dict__["stray"] = stray

    assert _unregistered_tables(module) == ["stray"]
