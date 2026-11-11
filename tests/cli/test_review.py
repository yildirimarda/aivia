"""``aivia review``: list, approve and reject reviews without a notifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aivia.cli import main


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    return path


def reviews_dir(config: Path) -> Path:
    return config.parent / "data"


def write_reviews(directory: Path, *reviews: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "reviews.json").write_text(
        json.dumps({"reviews": list(reviews)}), encoding="utf-8"
    )


def pending_review(
    review_id: str = "r-1", profile: str = "default"
) -> dict[str, object]:
    return {
        "id": review_id,
        "profile": profile,
        "state": "pending",
        "created_at": "2026-01-02T10:00:00+00:00",
    }


def approved_review(
    review_id: str = "r-1", profile: str = "default"
) -> dict[str, object]:
    return {
        "id": review_id,
        "profile": profile,
        "state": "approved",
        "created_at": "2026-01-02T10:00:00+00:00",
        "answered_at": "2026-01-02T11:00:00+00:00",
        "answered_by": "alice",
        "answer_reason": "looks good",
    }


def run_command(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["review", *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def common_args(config: Path) -> list[str]:
    return ["--config", str(config)]


def test_review_list_reports_no_reviews_when_the_store_is_empty(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)

    code, out, err = run_command(capsys, "list", *common_args(config))

    assert code == 0
    assert err == ""
    assert "No reviews." in out


def test_review_list_shows_each_review_with_its_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(
        reviews_dir(config),
        pending_review("r-1"),
        approved_review("r-2"),
    )

    code, out, err = run_command(capsys, "list", *common_args(config))

    assert code == 0
    assert err == ""
    assert "r-1" in out
    assert "r-2" in out
    assert "pending" in out
    assert "approved" in out


def test_review_approve_moves_a_pending_review_to_approved(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(reviews_dir(config), pending_review("r-1"))

    code, out, err = run_command(
        capsys,
        "approve",
        "r-1",
        "--actor",
        "alice",
        "--reason",
        "looks good",
        *common_args(config),
    )

    assert code == 0, err
    assert err == ""
    assert "Approved: r-1" in out

    directory = reviews_dir(config)
    stored = json.loads((directory / "reviews.json").read_text(encoding="utf-8"))
    assert stored["reviews"][0]["state"] == "approved"
    assert stored["reviews"][0]["answered_by"] == "alice"
    assert stored["reviews"][0]["answer_reason"] == "looks good"


def test_review_reject_moves_a_pending_review_to_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(reviews_dir(config), pending_review("r-1"))

    code, out, err = run_command(
        capsys,
        "reject",
        "r-1",
        "--actor",
        "bob",
        "--reason",
        "no",
        *common_args(config),
    )

    assert code == 0, err
    assert err == ""
    assert "Rejected: r-1" in out

    directory = reviews_dir(config)
    stored = json.loads((directory / "reviews.json").read_text(encoding="utf-8"))
    assert stored["reviews"][0]["state"] == "rejected"


def test_review_approve_persists_across_invocations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(reviews_dir(config), pending_review("r-1"))

    code, _, _ = run_command(capsys, "approve", "r-1", *common_args(config))
    assert code == 0

    code, out, _ = run_command(capsys, "list", *common_args(config))
    assert code == 0
    assert "r-1" in out
    assert "approved" in out
    assert "pending" not in out


def test_review_approve_rejects_an_unknown_review(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(reviews_dir(config), pending_review("r-1"))

    code, out, err = run_command(capsys, "approve", "r-missing", *common_args(config))

    assert code == 1
    assert out == ""
    assert "review not found: r-missing" in err


def test_review_reject_rejects_an_unknown_review(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(reviews_dir(config), pending_review("r-1"))

    code, out, err = run_command(capsys, "reject", "r-missing", *common_args(config))

    assert code == 1
    assert out == ""
    assert "review not found: r-missing" in err


def test_review_approve_rejects_an_already_terminal_review(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)
    write_reviews(reviews_dir(config), approved_review("r-1"))

    code, out, err = run_command(capsys, "approve", "r-1", *common_args(config))

    assert code == 1
    assert out == ""
    assert "approved" in err.lower()


def test_review_approve_requires_a_review_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)

    code, out, err = run_command(capsys, "approve", *common_args(config))

    assert code == 2
    assert out == ""
    assert "a review id is required" in err


def test_review_reject_requires_a_review_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)

    code, out, err = run_command(capsys, "reject", *common_args(config))

    assert code == 2
    assert out == ""
    assert "a review id is required" in err


def test_review_without_a_subcommand_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(tmp_path)

    code, _, err = run_command(capsys, *common_args(config))

    assert code == 2
    assert err != ""
