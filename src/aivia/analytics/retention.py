"""Import a retention export and report where viewers drop off.

A retention export records audience retention (the percentage of viewers still
watching) at timestamps throughout a video. Importing one lets us report the
*hook leak* (how much retention is lost during the hook), the *steepest drops*
(the consecutive segments where retention falls fastest) and the *closing slide*
(the retention at the close of the video), each with timestamps.

The module is deterministic and offline: it never calls a provider and never
touches the network.
"""
# config-coverage: allow-file

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ClosingSlide",
    "HookLeak",
    "RetentionExport",
    "RetentionPoint",
    "SteepestDrop",
    "closing_slide",
    "hook_leak",
    "import_retention_export",
    "steepest_drops",
]

#: Column names accepted for the timestamp column of a CSV retention export.
_TIMESTAMP_COLUMNS = (
    "timestamp",
    "time",
    "elapsed_time",
    "elapsed",
)

#: Column names accepted for the retention column of a CSV retention export.
_RETENTION_COLUMNS = (
    "retention",
    "retention_percent",
    "average_percentage_viewed",
    "avg_%_viewed",
    "average_%_viewed",
)


class RetentionPoint(BaseModel):
    """A single retention measurement: timestamp in seconds and percent retained."""

    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(ge=0)
    retention: float = Field(ge=0, le=100)


class RetentionExport(BaseModel):
    """A retention export: video id, duration and timestamped retention points.

    Points are kept in the order they appear in the export; :attr:`sorted_points`
    returns them ordered by timestamp.
    """

    model_config = ConfigDict(frozen=True)

    video_id: str | None = None
    duration: float | None = Field(default=None, ge=0)
    points: tuple[RetentionPoint, ...] = ()

    @property
    def sorted_points(self) -> tuple[RetentionPoint, ...]:
        """Return the points ordered by timestamp."""
        return tuple(sorted(self.points, key=lambda point: point.timestamp))


class HookLeak(BaseModel):
    """The retention lost during the hook, with timestamps.

    ``leak_points`` is ``start_retention - end_retention`` and ``leak_percent``
    is that loss expressed as a share of the starting retention. When the
    starting retention is zero the percent leak is ``0.0``.
    """

    model_config = ConfigDict(frozen=True)

    hook_seconds: float
    start_timestamp: float
    end_timestamp: float
    start_retention: float
    end_retention: float
    leak_points: float
    leak_percent: float


class SteepestDrop(BaseModel):
    """A consecutive pair of points where retention falls the most.

    ``drop_per_second`` normalises the drop by the elapsed time between the two
    points so short, sharp drops rank above long, gradual ones.
    """

    model_config = ConfigDict(frozen=True)

    from_timestamp: float
    to_timestamp: float
    drop_points: float
    drop_per_second: float


class ClosingSlide(BaseModel):
    """The retention at the close of the video, with timestamps.

    ``slide_seconds`` is the length of the closing window examined.
    """

    model_config = ConfigDict(frozen=True)

    start_timestamp: float
    end_timestamp: float
    start_retention: float
    end_retention: float
    slide_seconds: float


def _retention_at(points: tuple[RetentionPoint, ...], timestamp: float) -> float:
    """Return the retention at *timestamp*.

    Uses the point with the greatest timestamp not exceeding *timestamp*, falling
    back to the first point's retention when *timestamp* precedes every point.
    Returns ``0.0`` when there are no points.
    """
    if not points:
        return 0.0
    chosen = points[0]
    for point in points:
        if point.timestamp <= timestamp:
            chosen = point
        else:
            break
    return chosen.retention


def _column_index(header: list[str], candidates: tuple[str, ...]) -> int | None:
    lower = [name.strip().lower() for name in header]
    for candidate in candidates:
        if candidate in lower:
            return lower.index(candidate)
    return None


def _parse_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"retention export {path!s} has no header row")
        ts_index = _column_index(list(reader.fieldnames), _TIMESTAMP_COLUMNS)
        ret_index = _column_index(list(reader.fieldnames), _RETENTION_COLUMNS)
        if ts_index is None or ret_index is None:
            raise ValueError(
                f"retention export {path!s} needs a timestamp and a retention "
                f"column; found {reader.fieldnames!r}"
            )
        ts_name = reader.fieldnames[ts_index]
        ret_name = reader.fieldnames[ret_index]
        points: list[RetentionPoint] = []
        for row in reader:
            points.append(
                RetentionPoint(
                    timestamp=float(row[ts_name]),
                    retention=float(row[ret_name]),
                )
            )
    return {"points": tuple(points)}


def _parse_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        points = [
            RetentionPoint(
                timestamp=float(item["timestamp"]), retention=float(item["retention"])
            )
            for item in data
        ]
        return {"points": tuple(points)}
    if not isinstance(data, dict):
        raise ValueError(f"retention export {path!s} must be a JSON object or array")
    points = [
        RetentionPoint(
            timestamp=float(item["timestamp"]), retention=float(item["retention"])
        )
        for item in data.get("points", [])
    ]
    return {
        "video_id": data.get("video_id"),
        "duration": data.get("duration"),
        "points": tuple(points),
    }


def import_retention_export(path: str | Path) -> RetentionExport:
    """Import a retention export from *path* and return a :class:`RetentionExport`.

    CSV files need a timestamp column and a retention column. JSON files may be
    an array of ``{"timestamp", "retention"}`` objects or an object with an
    optional ``video_id`` and ``duration`` plus a ``points`` array.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"retention export not found: {path!s}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        payload = _parse_csv(path)
    elif suffix == ".json":
        payload = _parse_json(path)
    else:
        raise ValueError(
            f"unsupported retention export format {suffix!r}; use .csv or .json"
        )
    return RetentionExport(**payload)


def hook_leak(export: RetentionExport, hook_seconds: float = 30.0) -> HookLeak:
    """Report the retention lost during the hook.

    The hook runs from the first timestamp to ``first_timestamp + hook_seconds``.
    ``leak_points`` is the drop in retention across the hook and
    ``leak_percent`` is that drop as a share of the starting retention.
    """
    points = export.sorted_points
    if not points:
        raise ValueError("cannot compute a hook leak from an empty retention export")
    if hook_seconds <= 0:
        raise ValueError("hook_seconds must be greater than zero")

    start_timestamp = points[0].timestamp
    end_timestamp = start_timestamp + hook_seconds
    start_retention = _retention_at(points, start_timestamp)
    end_retention = _retention_at(points, end_timestamp)
    leak_points = start_retention - end_retention
    leak_percent = (
        (leak_points / start_retention * 100.0) if start_retention > 0 else 0.0
    )

    return HookLeak(
        hook_seconds=hook_seconds,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        start_retention=start_retention,
        end_retention=end_retention,
        leak_points=leak_points,
        leak_percent=leak_percent,
    )


def steepest_drops(export: RetentionExport, count: int = 5) -> list[SteepestDrop]:
    """Return the *count* consecutive segment pairs with the steepest retention drops.

    Only segments where retention falls are considered. Results are ordered by
    ``drop_per_second`` descending, then by ``from_timestamp`` ascending for a
    deterministic order.
    """
    if count <= 0:
        raise ValueError("count must be greater than zero")
    points = export.sorted_points
    drops: list[SteepestDrop] = []
    for earlier, later in itertools.pairwise(points):
        elapsed = later.timestamp - earlier.timestamp
        if elapsed <= 0:
            continue
        drop_points = earlier.retention - later.retention
        if drop_points <= 0:
            continue
        drops.append(
            SteepestDrop(
                from_timestamp=earlier.timestamp,
                to_timestamp=later.timestamp,
                drop_points=drop_points,
                drop_per_second=drop_points / elapsed,
            )
        )
    drops.sort(key=lambda drop: (-drop.drop_per_second, drop.from_timestamp))
    return drops[:count]


def closing_slide(export: RetentionExport, slide_seconds: float = 30.0) -> ClosingSlide:
    """Report the retention at the close of the video.

    The closing window is the last ``slide_seconds`` of the video. When the
    video is shorter than the window the whole export is reported.
    """
    points = export.sorted_points
    if not points:
        raise ValueError(
            "cannot compute a closing slide from an empty retention export"
        )
    if slide_seconds <= 0:
        raise ValueError("slide_seconds must be greater than zero")

    end_timestamp = points[-1].timestamp
    start_timestamp = max(points[0].timestamp, end_timestamp - slide_seconds)
    start_retention = _retention_at(points, start_timestamp)
    end_retention = _retention_at(points, end_timestamp)

    return ClosingSlide(
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        start_retention=start_retention,
        end_retention=end_retention,
        slide_seconds=slide_seconds,
    )
