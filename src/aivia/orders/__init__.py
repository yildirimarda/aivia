"""Personalized orders: lifecycle, validation, delivery and pronunciation.

The orders module turns an occasion, a recipient name and optional variables
into generated media. Name pronunciation is verified by transcription and the
spoken name is regenerated with phonetic hints up to the configured retry
limit.
"""

from __future__ import annotations

from aivia.orders.pronunciation import (
    NamePronunciationError,
    NamePronunciationResult,
    verify_name_pronunciation,
)

__all__ = [
    "NamePronunciationError",
    "NamePronunciationResult",
    "verify_name_pronunciation",
]
