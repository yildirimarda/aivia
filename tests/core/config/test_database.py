"""The database config section and engines built from it."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from aivia.core.config import (
    CORE_SECTION_KEYS,
    AppConfig,
    DatabaseConfig,
    SectionRegistry,
    register_core_sections,
)
from aivia.core.db import (
    DEFAULT_BUSY_TIMEOUT_MS,
    DEFAULT_DATABASE_FILENAME,
    create_database_engine,
    database_path,
    upgrade_database,
)
from aivia.core.errors import ConfigError


def _pragma(engine: Engine, name: str) -> object:
    with engine.connect() as connection:
        return connection.execute(text(f"PRAGMA {name}")).scalar()


def test_database_defaults_match_the_engine_defaults() -> None:
    config = DatabaseConfig.from_mapping({})

    assert config.filename == DEFAULT_DATABASE_FILENAME
    assert config.busy_timeout_ms == DEFAULT_BUSY_TIMEOUT_MS


def test_database_section_is_registered_and_parsed_from_the_root() -> None:
    assert "database" in CORE_SECTION_KEYS
    assert "database" in register_core_sections(SectionRegistry())

    config = AppConfig.from_mapping(
        {"database": {"filename": "custom.db", "busy_timeout_ms": 1_234}}
    )

    assert config.database is not None  # type: ignore[attr-defined]
    assert config.database.filename == "custom.db"  # type: ignore[attr-defined]
    assert config.database.busy_timeout_ms == 1_234  # type: ignore[attr-defined]


def test_database_rejects_a_filename_that_is_a_path() -> None:
    for filename in ("", " ", ".", "..", "sub/aivia.db", "/tmp/aivia.db"):
        with pytest.raises(ConfigError) as caught:
            DatabaseConfig.from_mapping({"filename": filename})

        assert "filename" in str(caught.value)


def test_database_rejects_a_negative_busy_timeout() -> None:
    with pytest.raises(ConfigError) as caught:
        DatabaseConfig.from_mapping({"busy_timeout_ms": -1})

    assert "busy_timeout_ms" in str(caught.value)


def test_engine_uses_the_configured_filename_in_the_data_directory(
    tmp_path: Path,
) -> None:
    config = DatabaseConfig.from_mapping({"filename": "custom.db"})
    data_dir = tmp_path / "data"

    engine = create_database_engine(config=config, data_dir=data_dir)
    try:
        with engine.connect():
            pass
        expected = database_path(data_dir, filename="custom.db")
        assert expected.is_file()
        assert not database_path(data_dir).exists()
    finally:
        engine.dispose()


def test_engine_uses_the_configured_filename_in_the_user_data_directory(
    tmp_path: Path,
) -> None:
    environ = {"XDG_CONFIG_HOME": str(tmp_path)}
    config = DatabaseConfig.from_mapping({"filename": "user.db"})

    engine = create_database_engine(config=config, environ=environ)
    try:
        with engine.connect():
            pass
        assert database_path(environ=environ, filename="user.db").is_file()
    finally:
        engine.dispose()


def test_engine_uses_the_configured_busy_timeout(tmp_path: Path) -> None:
    config = DatabaseConfig.from_mapping({"busy_timeout_ms": 1_234})

    engine = create_database_engine(config=config, path=tmp_path / "aivia.db")
    try:
        assert _pragma(engine, "busy_timeout") == 1_234
    finally:
        engine.dispose()


def test_an_explicit_path_wins_over_the_configured_filename(tmp_path: Path) -> None:
    config = DatabaseConfig.from_mapping({"filename": "ignored.db"})
    explicit = tmp_path / "explicit.db"

    engine = create_database_engine(explicit, config=config)
    try:
        with engine.connect():
            pass
        assert explicit.is_file()
        assert not (tmp_path / "ignored.db").exists()
    finally:
        engine.dispose()


def test_config_cannot_be_combined_with_explicit_engine_arguments(
    tmp_path: Path,
) -> None:
    config = DatabaseConfig.from_mapping({})

    with pytest.raises(ValueError, match="config"):
        create_database_engine(
            config=config, path=tmp_path / "aivia.db", busy_timeout_ms=1_000
        )
    with pytest.raises(ValueError, match="config"):
        create_database_engine(
            config=config, path=tmp_path / "aivia.db", filename="other.db"
        )


def test_upgrading_from_config_creates_the_configured_file(tmp_path: Path) -> None:
    config = DatabaseConfig.from_mapping(
        {"filename": "migrated.db", "busy_timeout_ms": 1_234}
    )
    data_dir = tmp_path / "data"

    upgrade_database(config=config, data_dir=data_dir)

    expected = database_path(data_dir, filename="migrated.db")
    assert expected.is_file()
    engine = create_database_engine(config=config, data_dir=data_dir)
    try:
        assert _pragma(engine, "busy_timeout") == 1_234
    finally:
        engine.dispose()
