"""Optional review of an extracted style spec before any paid generation.

When a reference is analysed into a style spec, the spec can be shown to a
person for edit or confirmation before a provider is called. Review is opt-in:
without it the extracted spec flows straight into generation. Nothing here
calls a provider, touches the network or writes to the ledger; it only renders
the spec and records whether a person confirmed it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

import yaml

from aivia.reference.schema import StyleSpec

__all__ = ["SpecReview", "render_style_spec", "review_style_spec"]


@dataclass(frozen=True, slots=True)
class SpecReview:
    """The outcome of reviewing one style spec.

    ``confirmed`` is true when generation may proceed: either review is
    disabled, or the spec was shown and accepted. ``shown`` is true when the
    spec was rendered for a person to read. ``declined`` is true when review
    was enabled and not accepted, which means no paid generation should happen.
    """

    confirmed: bool
    shown: bool
    declined: bool


def render_style_spec(spec: StyleSpec) -> str:
    """Return *spec* as human-readable YAML for review.

    Only fields that carry information are included, so a partially learned
    spec shows what was actually extracted instead of a wall of defaults.
    """

    payload = spec.to_dict(exclude_none=True)
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=1000,
    )


def review_style_spec(
    spec: StyleSpec,
    *,
    enabled: bool = False,
    confirm: bool | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> SpecReview:
    """Show *spec* for edit or confirmation before paid generation.

    When *enabled* is false the spec is not shown and review passes without
    touching the streams. When enabled the spec is rendered to *stdout* and
    confirmation is requested: *confirm* overrides stdin, otherwise one line is
    read from *stdin* and anything starting with ``y`` or ``Y`` confirms.

    Returns a :class:`SpecReview`. ``confirmed`` is true only when generation
    may proceed.
    """

    out = stdout if stdout is not None else sys.stdout
    if not enabled:
        return SpecReview(confirmed=True, shown=False, declined=False)

    out.write("Style spec extracted from reference — review before paid generation:\n")
    out.write(render_style_spec(spec))
    out.write("---\n")
    out.flush()

    if confirm is None:
        active = stdin if stdin is not None else sys.stdin
        line = active.readline()
        confirm = bool(line) and line.strip().lower().startswith("y")

    confirmed = bool(confirm)
    return SpecReview(confirmed=confirmed, shown=True, declined=not confirmed)
