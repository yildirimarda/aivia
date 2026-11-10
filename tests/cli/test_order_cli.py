"""``aivia order``: create, show and delete personalized orders."""

from __future__ import annotations

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


def order_id_from(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("Order: "))
    return line.removeprefix("Order: ").strip()


def common_args(config: Path, database: Path) -> list[str]:
    return ["--config", str(config), "--database", str(database)]


def stored_order_ids(database: Path) -> list[str]:
    engine = create_database_engine(database)
    try:
        return [order["id"] for order in OrderRepository(engine).list()]
    finally:
        engine.dispose()


def test_order_create_prints_a_new_order(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(
        capsys,
        "create",
        "--occasion",
        "birthday",
        "--name",
        "Ada",
        "--message",
        "Happy birthday",
        "--amount",
        "10",
        *common_args(config, database),
    )

    assert code == 0
    assert err == ""
    assert "Order: " in out
    assert "State: draft" in out
    assert "Name: Ada" in out
    assert "Amount: $10.000000" in out

    ids = stored_order_ids(database)
    assert len(ids) == 1
    assert order_id_from(out) == ids[0]


def test_a_repeated_idempotency_key_returns_the_existing_order(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_code, first_out, first_err = run_command(
        capsys,
        "create",
        "--occasion",
        "birthday",
        "--name",
        "Ada",
        "--idempotency-key",
        "  order-42  ",
        *common_args(config, database),
    )
    assert first_code == 0
    assert first_err == ""
    order_id = order_id_from(first_out)

    second_code, second_out, second_err = run_command(
        capsys,
        "create",
        "--occasion",
        "anniversary",
        "--name",
        "Someone else",
        "--message",
        "different payload",
        "--idempotency-key",
        "order-42",
        *common_args(config, database),
    )
    assert second_code == 0
    assert second_err == ""
    assert order_id_from(second_out) == order_id

    assert stored_order_ids(database) == [order_id]


def test_order_create_without_a_key_creates_a_new_order_each_time(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_code, first_out, _ = run_command(
        capsys,
        "create",
        "--name",
        "Ada",
        *common_args(config, database),
    )
    second_code, second_out, _ = run_command(
        capsys,
        "create",
        "--name",
        "Ada",
        *common_args(config, database),
    )

    assert first_code == 0
    assert second_code == 0
    first_id = order_id_from(first_out)
    second_id = order_id_from(second_out)
    assert first_id != second_id
    assert stored_order_ids(database) == [first_id, second_id]


def test_order_show_prints_an_order(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_code, create_out, _ = run_command(
        capsys,
        "create",
        "--name",
        "Ada",
        *common_args(config, database),
    )
    assert create_code == 0
    order_id = order_id_from(create_out)

    code, out, err = run_command(
        capsys, "show", order_id, *common_args(config, database)
    )

    assert code == 0
    assert err == ""
    assert f"Order: {order_id}" in out
    assert "State: draft" in out
    assert "Name: Ada" in out


def test_order_show_rejects_an_unknown_order(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(
        capsys, "show", "does-not-exist", *common_args(config, database)
    )

    assert code == 1
    assert out == ""
    assert "order not found: does-not-exist" in err


def test_order_show_requires_an_order_id(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(capsys, "show", *common_args(config, database))

    assert code == 2
    assert out == ""
    assert "an order id is required" in err


def test_order_delete_removes_an_order(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    create_code, create_out, _ = run_command(
        capsys,
        "create",
        "--name",
        "Ada",
        *common_args(config, database),
    )
    assert create_code == 0
    order_id = order_id_from(create_out)

    code, out, err = run_command(
        capsys, "delete", order_id, *common_args(config, database)
    )

    assert code == 0
    assert err == ""
    assert f"Deleted: {order_id}" in out
    assert stored_order_ids(database) == []

    show_code, _, show_err = run_command(
        capsys, "show", order_id, *common_args(config, database)
    )
    assert show_code == 1
    assert "order not found" in show_err


def test_order_delete_rejects_an_unknown_order(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(
        capsys, "delete", "does-not-exist", *common_args(config, database)
    )

    assert code == 1
    assert out == ""
    assert "order not found: does-not-exist" in err


def test_order_delete_requires_an_order_id(
    config: Path, database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run_command(capsys, "delete", *common_args(config, database))

    assert code == 2
    assert out == ""
    assert "an order id is required" in err
