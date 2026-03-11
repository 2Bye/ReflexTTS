"""PII masker — mask personally identifiable information before TTS.

Detects and replaces PII (emails, phones, IDs, credit cards) with
safe placeholders before the text reaches the LLM or TTS engine.
Supports restore after synthesis for logging purposes.

Usage:
    masked, mapping = mask_pii("Call me at +7-999-123-4567")
    # masked = "Call me at [PHONE_1]"
    # mapping = {"[PHONE_1]": "+7-999-123-4567"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.log import get_logger

logger = get_logger(__name__)


@dataclass
class PIIMaskResult:
    """Result of PII masking.

    Attributes:
        masked_text: Text with PII replaced by placeholders.
        mapping: Placeholder → original value mapping.
        pii_count: Number of PII items found.
        pii_types: Types of PII detected.
    """

    masked_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    pii_count: int = 0
    pii_types: list[str] = field(default_factory=list)


# PII detection patterns (name, regex, placeholder_prefix)
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )),
    ("PHONE", re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{2,4}\b"
    )),
    ("CREDIT_CARD", re.compile(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    )),
    ("PASSPORT", re.compile(
        r"\b\d{2}\s?\d{2}\s?\d{6}\b"
    )),
    ("INN", re.compile(
        r"\b\d{10,12}\b"
    )),
    ("IP_ADDRESS", re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    )),
]


def mask_pii(text: str) -> PIIMaskResult:
    """Detect and mask PII in text.

    Replaces each PII occurrence with a numbered placeholder
    like [EMAIL_1], [PHONE_2], etc.

    Args:
        text: Input text potentially containing PII.

    Returns:
        PIIMaskResult with masked text and reverse mapping.
    """
    result_text = text
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    types_found: set[str] = set()

    for pii_type, pattern in _PII_PATTERNS:
        matches = list(pattern.finditer(result_text))
        for match in reversed(matches):  # Reverse to preserve indices
            original = match.group()
            count = counters.get(pii_type, 0) + 1
            counters[pii_type] = count
            placeholder = f"[{pii_type}_{count}]"

            mapping[placeholder] = original
            types_found.add(pii_type)

            result_text = (
                result_text[:match.start()]
                + placeholder
                + result_text[match.end():]
            )

    pii_count = len(mapping)

    if pii_count > 0:
        logger.info(
            "pii_masked",
            count=pii_count,
            types=sorted(types_found),
        )

    return PIIMaskResult(
        masked_text=result_text,
        mapping=mapping,
        pii_count=pii_count,
        pii_types=sorted(types_found),
    )


def restore_pii(masked_text: str, mapping: dict[str, str]) -> str:
    """Restore original PII values from placeholders.

    Args:
        masked_text: Text with PII placeholders.
        mapping: Placeholder → original value mapping.

    Returns:
        Text with original PII restored.
    """
    result = masked_text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)
    return result
