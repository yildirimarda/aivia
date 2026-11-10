"""``aivia order import``: batch order import from CSV or JSON.

Each row is validated and either stored or isolated with a per-row reason, so
good rows are never blocked by bad ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.db import (
    OrderRepository,
    create_database_engine,
    upgrade_database,
)


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(tmp_path, "schema_version: 1\n")


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "aivia.db"
    upgrade_database(path)
    return path


def run_command(
    capsys: pytest.CaptureFixture[str],
    *args: str,
) -> tuple[int, str, str]:
    code = main(["order", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def common_args(config: Path, database: Path) -> list[str]:
    return ["--config", str(config), "--database", str(database)]


def stored_order_ids(database: Path) -> list[str]:
    engine = create_database_engine(database)
    try:
        return [order["id"] for order in OrderRepository(engine).list()]
    finally:
        engine.dispose()


def test_csv_import_creates_valid_rows_and_isolates_bad_rows(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    batch = tmp_path / "orders.csv"
    batch.write_text(
        "occasion,name,message,amount_usd\n"
        "birthday,Ada,Happy birthday,25\n"
        "anniversary,Ben,Happy anniversary,bad-amount\n"
        "birthday,Cel,Congrats,10\n",
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "import",
        "--file",
        str(batch),
        *common_args(config, database),
    )

    assert code == 1
    assert "Succeeded: 2" in out
    assert "Failed: 1" in out
    assert "Row 2: bad amount_usd: 'bad-amount'" in err
    assert len(stored_order_ids(database)) == 2


def test_json_import_creates_all_rows(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    batch = tmp_path / "orders.json"
    batch.write_text(
        json.dumps(
            [
                {"occasion": "birthday", "name": "Ada", "amount_usd": 25},
                {"occasion": "anniversary", "name": "Ben", "amount_usd": 10},
            ]
        ),
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "import",
        "--file",
        str(batch),
        *common_args(config, database),
    )

    assert code == 0
    assert err == ""
    assert "Succeeded: 2" in out
    assert "Failed: 0" in out
    assert len(stored_order_ids(database)) == 2


def test_json_import_with_items_key(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    batch = tmp_path / "orders.json"
    batch.write_text(
        json.dumps(
            {
                "items": [
                    {"name": "Ada", "amount_usd": 5},
                    {"name": "Ben", "bogus": 1},
                    {"name": "Cel", "amount_usd": "not-money"},
                ]
            }
        ),
        encoding="utf-8",
    )

    code, out, err = run_command(
        capsys,
        "import",
        "--file",
        str(batch),
        *common_args(config, database),
    )

    assert code == 1
    assert "Succeeded: 1" in out
    assert "Failed: 2" in out
    assert "Row 2: unknown field: bogus" in err
    assert "Row 3: bad amount_usd: 'not-money'" in err
    assert len(stored_order_ids(database)) == 1


def test_unknown_file_extension_is_rejected(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    batch = tmp_path / "orders.txt"
    batch.write_text("name\nAda\n", encoding="utf-8")

    code, _out, err = run_command(
        capsys,
        "import",
        "--file",
        str(batch),
        *common_args(config, database),
    )

    assert code == 1
    assert "expected .csv or .json" in err
    assert stored_order_ids(database) == []


def test_format_flag_overrides_extension(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    batch = tmp_path / "orders.dat"
    batch.write_text('{"name": "Ada", "amount_usd": 3}', encoding="utf-8")

    code, out, err = run_command(
        capsys,
        "import",
        "--file",
        str(batch),
        "--format",
        "json",
        *common_args(config, database),
    )

    assert code == 0
    assert err == ""
    assert "Succeeded: 1" in out
    assert len(stored_order_ids(database)) == 1


def test_missing_batch_file_is_rejected(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    code, _out, err = run_command(
        capsys,
        "import",
        "--file",
        str(tmp_path / "does-not-exist.csv"),
        *common_args(config, database),
    )

    assert code == 1
    assert "batch file not found" in err
    assert stored_order_ids(database) == []


def test_empty_batch_succeeds_with_nothing(
    config: Path,
    database: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    batch = tmp_path / "orders.csv"
    batch.write_text("name\n", encoding="utf-8")

    code, out, err = run_command(
        capsys,
        "import",
        "--file",
        str(batch),
        *common_args(config, database),
    )

    assert code == 0
    assert err == ""
    assert "Succeeded: 0" in out
    assert "Failed: 0" in out
    assert stored_order_ids(database) == []
