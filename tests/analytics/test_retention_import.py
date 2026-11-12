"""Importing a retention export reports the hook leak, steepest drops and closing slide.

A retention export records audience retention (the percent of viewers still
watching) at timestamps throughout a video. Importing one is deterministic and
offline: it never calls a provider and never touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.analytics import (
    ClosingSlide,
    HookLeak,
    RetentionExport,
    RetentionPoint,
    SteepestDrop,
    closing_slide,
    hook_leak,
    import_retention_export,
    steepest_drops,
)


def _write_csv(tmp_path: Path, name: str, rows: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("timestamp,retention\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_json(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _points(*pairs: tuple[float, float]) -> tuple[RetentionPoint, ...]:
    return tuple(RetentionPoint(timestamp=t, retention=r) for t, r in pairs)


def test_import_retention_export_from_csv(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "retention.csv",
        [
            "0.0,100.0",
            "30.0,85.0",
            "60.0,70.0",
            "90.0,60.0",
            "120.0,50.0",
        ],
    )
    export = import_retention_export(path)

    assert export.video_id is None
    assert export.duration is None
    assert [p.timestamp for p in export.sorted_points] == [0.0, 30.0, 60.0, 90.0, 120.0]
    assert [p.retention for p in export.sorted_points] == [
        100.0,
        85.0,
        70.0,
        60.0,
        50.0,
    ]


def test_import_retention_export_unsorted_csv_is_sorted(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "retention.csv",
        [
            "60.0,70.0",
            "0.0,100.0",
            "120.0,50.0",
            "30.0,85.0",
        ],
    )
    export = import_retention_export(path)

    assert [p.timestamp for p in export.sorted_points] == [0.0, 30.0, 60.0, 120.0]


def test_import_retention_export_from_json_object(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path,
        "retention.json",
        (
            '{"video_id": "vid-1", "duration": 120.0, "points": ['
            '{"timestamp": 0.0, "retention": 100.0}, '
            '{"timestamp": 60.0, "retention": 50.0}]}'
        ),
    )
    export = import_retention_export(path)

    assert export.video_id == "vid-1"
    assert export.duration == 120.0
    assert [p.timestamp for p in export.sorted_points] == [0.0, 60.0]
    assert [p.retention for p in export.sorted_points] == [100.0, 50.0]


def test_import_retention_export_from_json_array(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path,
        "retention.json",
        (
            '[{"timestamp": 0.0, "retention": 100.0}, '
            '{"timestamp": 30.0, "retention": 80.0}]'
        ),
    )
    export = import_retention_export(path)

    assert export.video_id is None
    assert export.duration is None
    assert [p.retention for p in export.sorted_points] == [100.0, 80.0]


def test_import_retention_export_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="retention export not found"):
        import_retention_export(Path("does/not/exist.csv"))


def test_import_retention_export_unsupported_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "retention.txt"
    path.write_text("0.0,100.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported retention export format"):
        import_retention_export(path)


def test_hook_leak_reports_drop_during_the_hook() -> None:
    export = RetentionExport(points=_points((0.0, 100.0), (30.0, 70.0), (60.0, 50.0)))

    leak = hook_leak(export, hook_seconds=30.0)

    assert isinstance(leak, HookLeak)
    assert leak.hook_seconds == 30.0
    assert leak.start_timestamp == 0.0
    assert leak.end_timestamp == 30.0
    assert leak.start_retention == 100.0
    assert leak.end_retention == 70.0
    assert leak.leak_points == 30.0
    assert leak.leak_percent == 30.0


def test_hook_leak_percent_is_zero_when_start_retention_is_zero() -> None:
    export = RetentionExport(points=_points((0.0, 0.0), (30.0, 0.0)))

    leak = hook_leak(export, hook_seconds=30.0)

    assert leak.leak_points == 0.0
    assert leak.leak_percent == 0.0


def test_hook_leak_defaults_to_thirty_seconds() -> None:
    export = RetentionExport(points=_points((0.0, 100.0), (30.0, 60.0), (60.0, 40.0)))

    leak = hook_leak(export)

    assert leak.hook_seconds == 30.0
    assert leak.end_timestamp == 30.0
    assert leak.end_retention == 60.0


def test_hook_leak_uses_point_at_or_before_the_hook_end() -> None:
    export = RetentionExport(points=_points((0.0, 100.0), (20.0, 90.0), (40.0, 50.0)))

    leak = hook_leak(export, hook_seconds=30.0)

    # end_timestamp is 30.0; the greatest timestamp not exceeding it is 20.0.
    assert leak.end_timestamp == 30.0
    assert leak.end_retention == 90.0
    assert leak.leak_points == 10.0


def test_hook_leak_requires_non_positive_hook_seconds_to_be_rejected() -> None:
    export = RetentionExport(points=_points((0.0, 100.0)))

    with pytest.raises(ValueError, match="hook_seconds must be greater than zero"):
        hook_leak(export, hook_seconds=0)


def test_hook_leak_rejects_empty_export() -> None:
    export = RetentionExport(points=())

    with pytest.raises(ValueError, match="empty retention export"):
        hook_leak(export)


def test_steepest_drops_ranks_by_drop_per_second() -> None:
    export = RetentionExport(
        points=_points(
            (0.0, 100.0),
            (10.0, 90.0),  # drop 10 over 10s -> 1.0/s
            (20.0, 50.0),  # drop 40 over 10s -> 4.0/s
            (30.0, 45.0),  # drop 5 over 10s -> 0.5/s
        )
    )

    drops = steepest_drops(export, count=2)

    assert len(drops) == 2
    assert all(isinstance(drop, SteepestDrop) for drop in drops)
    assert drops[0].from_timestamp == 10.0
    assert drops[0].to_timestamp == 20.0
    assert drops[0].drop_points == 40.0
    assert drops[0].drop_per_second == 4.0
    assert drops[1].from_timestamp == 0.0
    assert drops[1].to_timestamp == 10.0
    assert drops[1].drop_per_second == 1.0


def test_steepest_drops_ignores_non_drops() -> None:
    export = RetentionExport(
        points=_points(
            (0.0, 100.0),
            (10.0, 100.0),  # retention flat: not a drop
            (20.0, 90.0),  # drop 20 over 10s -> 2.0/s
        )
    )

    drops = steepest_drops(export)

    assert len(drops) == 1
    assert drops[0].from_timestamp == 10.0
    assert drops[0].to_timestamp == 20.0


def test_steepest_drops_defaults_to_five() -> None:
    export = RetentionExport(
        points=_points(
            (0.0, 100.0),
            (10.0, 90.0),
            (20.0, 80.0),
            (30.0, 70.0),
            (40.0, 60.0),
            (50.0, 50.0),
        )
    )

    drops = steepest_drops(export)

    assert len(drops) == 5


def test_steepest_drops_requires_positive_count() -> None:
    export = RetentionExport(points=_points((0.0, 100.0), (10.0, 90.0)))

    with pytest.raises(ValueError, match="count must be greater than zero"):
        steepest_drops(export, count=0)


def test_steepest_drops_empty_or_single_point_is_empty() -> None:
    assert steepest_drops(RetentionExport(points=_points((0.0, 100.0)))) == []
    assert steepest_drops(RetentionExport(points=())) == []


def test_closing_slide_reports_the_close_of_the_video() -> None:
    export = RetentionExport(
        points=_points((0.0, 100.0), (60.0, 70.0), (90.0, 60.0), (120.0, 50.0))
    )

    slide = closing_slide(export, slide_seconds=30.0)

    assert isinstance(slide, ClosingSlide)
    assert slide.slide_seconds == 30.0
    assert slide.end_timestamp == 120.0
    assert slide.start_timestamp == 90.0
    assert slide.start_retention == 60.0
    assert slide.end_retention == 50.0


def test_closing_slide_defaults_to_thirty_seconds() -> None:
    export = RetentionExport(points=_points((0.0, 100.0), (60.0, 70.0), (120.0, 50.0)))

    slide = closing_slide(export)

    assert slide.slide_seconds == 30.0
    assert slide.start_timestamp == 90.0


def test_closing_slide_covers_the_whole_video_when_shorter_than_the_window() -> None:
    export = RetentionExport(points=_points((0.0, 100.0), (10.0, 80.0)))

    slide = closing_slide(export, slide_seconds=60.0)

    assert slide.start_timestamp == 0.0
    assert slide.end_timestamp == 10.0


def test_closing_slide_requires_positive_slide_seconds() -> None:
    export = RetentionExport(points=_points((0.0, 100.0)))

    with pytest.raises(ValueError, match="slide_seconds must be greater than zero"):
        closing_slide(export, slide_seconds=0)


def test_closing_slide_rejects_empty_export() -> None:
    export = RetentionExport(points=())

    with pytest.raises(ValueError, match="empty retention export"):
        closing_slide(export)


def test_retention_export_is_a_model() -> None:
    export = RetentionExport(
        video_id="vid-1",
        duration=120.0,
        points=_points((0.0, 100.0), (60.0, 50.0)),
    )

    assert export.video_id == "vid-1"
    assert export.duration == 120.0
