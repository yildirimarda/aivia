"""``aivia review``: list, approve and reject reviews without a notifier.

The manual fallback for human review when no notifier provider is configured.
Reviews persist as a JSON file in the user data directory so a review survives
across CLI invocations and can be acted on later.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aivia.cli.registry import register_command
from aivia.core import paths
from aivia.review.review_state import (
    IllegalReviewTransition,
    Review,
    ReviewState,
    transition_review,
)

__all__ = ["configure", "review_approve", "review_list", "review_reject"]

_REVIEW_FILENAME = "reviews.json"
_DEFAULT_PROFILE = "default"


def _reviews_path(args: argparse.Namespace) -> Path:
    return (
        paths.resolve_paths(getattr(args, "config", None)).data_dir / _REVIEW_FILENAME
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_instant(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return datetime.now(UTC)
    return datetime.fromisoformat(text)


def _dump_reviews(reviews: dict[str, Review]) -> list[dict[str, Any]]:
    return [
        {
            "id": review.id,
            "profile": review.profile,
            "state": review.state.value,
            "created_at": _iso(review.created_at),
            "updated_at": _iso(review.updated_at),
            "answered_at": _iso(review.answered_at),
            "answered_by": review.answered_by,
            "answer_reason": review.answer_reason,
            "timeout_at": _iso(review.timeout_at),
            "timeout_action": review.timeout_action,
            "regenerated_count": review.regenerated_count,
        }
        for review in reviews.values()
    ]


def _load_reviews(path: Path) -> dict[str, Review]:
    reviews: dict[str, Review] = {}
    if not path.is_file():
        return reviews
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return reviews
    items = data.get("reviews") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return reviews
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            review = Review(
                id=str(item.get("id", "") or "").strip(),
                profile=(
                    str(item.get("profile") or _DEFAULT_PROFILE).strip()
                    or _DEFAULT_PROFILE
                ),
                state=ReviewState(str(item.get("state") or "pending").strip()),
                created_at=_parse_instant(item.get("created_at")),
                updated_at=_parse_instant(item["updated_at"])
                if item.get("updated_at")
                else None,
                answered_at=_parse_instant(item["answered_at"])
                if item.get("answered_at")
                else None,
                answered_by=str(item.get("answered_by") or ""),
                answer_reason=str(item.get("answer_reason") or ""),
                timeout_at=_parse_instant(item["timeout_at"])
                if item.get("timeout_at")
                else None,
                timeout_action=str(item.get("timeout_action") or ""),
                regenerated_count=int(item.get("regenerated_count") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if review.id:
            reviews[review.id] = review
    return reviews


def _save_reviews(path: Path, reviews: dict[str, Review]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"reviews": _dump_reviews(reviews)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def configure(parser: argparse.ArgumentParser) -> None:
    """Add the review command and its list, approve and reject subcommands."""
    subparsers = parser.add_subparsers(
        dest="review_command", required=True, metavar="COMMAND"
    )
    list_ = subparsers.add_parser(
        "list", help="List reviews", description="List every stored review"
    )
    list_.set_defaults(_review_handler=review_list)
    approve = subparsers.add_parser(
        "approve",
        help="Approve a review",
        description="Move a pending review to approved",
    )
    approve.add_argument(
        "review_id",
        nargs="?",
        default=None,
        metavar="REVIEW_ID",
        help="review to act on",
    )
    approve.add_argument("--actor", default="", metavar="NAME", help="who is acting")
    approve.add_argument(
        "--reason", default="", metavar="TEXT", help="reason for the decision"
    )
    approve.set_defaults(_review_handler=review_approve)
    reject = subparsers.add_parser(
        "reject",
        help="Reject a review",
        description="Move a pending review to rejected",
    )
    reject.add_argument(
        "review_id",
        nargs="?",
        default=None,
        metavar="REVIEW_ID",
        help="review to act on",
    )
    reject.add_argument("--actor", default="", metavar="NAME", help="who is acting")
    reject.add_argument(
        "--reason", default="", metavar="TEXT", help="reason for the decision"
    )
    reject.set_defaults(_review_handler=review_reject)


def review_list(args: argparse.Namespace) -> int:
    """Print every stored review."""
    path = _reviews_path(args)
    try:
        reviews = _load_reviews(path)
    except OSError as exc:
        print(f"error: cannot read reviews: {exc}", file=sys.stderr)
        return 1
    if not reviews:
        print("No reviews.")
        return 0
    print("ID        STATE      PROFILE   CREATED AT")
    for review in sorted(reviews.values(), key=lambda item: item.created_at):
        created_at = _iso(review.created_at) or "-"
        line = (
            f"{review.id:<12} {review.state.value:<10} "
            f"{review.profile:<10} {created_at}"
        )
        print(line)
    return 0


def _act(args: argparse.Namespace, target: ReviewState, verb: str) -> int:
    review_id = (getattr(args, "review_id", None) or "").strip()
    if not review_id:
        print("error: a review id is required", file=sys.stderr)
        return 2
    path = _reviews_path(args)
    try:
        reviews = _load_reviews(path)
    except OSError as exc:
        print(f"error: cannot read reviews: {exc}", file=sys.stderr)
        return 1
    review = reviews.get(review_id)
    if review is None:
        print(f"error: review not found: {review_id}", file=sys.stderr)
        return 1
    try:
        updated = transition_review(
            review,
            target,
            actor=str(getattr(args, "actor", "") or "").strip(),
            reason=str(getattr(args, "reason", "") or "").strip(),
        )
    except IllegalReviewTransition as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    reviews[review_id] = updated
    try:
        _save_reviews(path, reviews)
    except OSError as exc:
        print(f"error: cannot write reviews: {exc}", file=sys.stderr)
        return 1
    print(f"{verb.capitalize()}: {review_id}")
    return 0


def review_approve(args: argparse.Namespace) -> int:
    """Move a pending review to approved."""
    return _act(args, ReviewState.APPROVED, "approved")


def review_reject(args: argparse.Namespace) -> int:
    """Move a pending review to rejected."""
    return _act(args, ReviewState.REJECTED, "rejected")


def run_review(args: argparse.Namespace) -> int:
    """Dispatch to the selected review subcommand."""
    handler = getattr(args, "_review_handler", None)
    if handler is None:
        print("error: a review command is required", file=sys.stderr)
        return 2
    return int(handler(args))


register_command(
    "review",
    run_review,
    help="List, approve and reject reviews without a notifier",
    configure=configure,
)
