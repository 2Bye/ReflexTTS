"""Input sanitizer — prompt injection guard.

Validates and sanitizes user input before it reaches the LLM.
Uses pattern matching to detect common injection techniques
and enforces length limits.

Usage:
    result = sanitize_input("Hello, synthesize this text")
    if not result.is_safe:
        raise ValueError(result.reason)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.log import get_logger

logger = get_logger(__name__)

# Maximum allowed input length (characters)
MAX_INPUT_LENGTH = 2000

# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", "ignore_previous"),
    (r"disregard\s+(all\s+)?(previous|above|prior)", "disregard_previous"),
    (r"you\s+are\s+now\s+(a|an|the)\b", "role_override"),
    (r"act\s+as\s+(a|an|the|if)\b", "act_as"),
    (r"pretend\s+(to\s+be|you\s+are)", "pretend"),
    (r"system\s*:\s*", "system_prompt_inject"),
    (r"\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>", "chat_template_inject"),
    (r"```\s*(system|assistant|function)", "code_block_inject"),
    (r"<\s*script\b", "xss_attempt"),
    (r"(\{\{|\}\}|<%|%>)", "template_inject"),
]

# Compiled patterns for performance
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), name) for pattern, name in _INJECTION_PATTERNS
]


@dataclass
class SanitizeResult:
    """Result of input sanitization.

    Attributes:
        is_safe: Whether the input passed all checks.
        sanitized_text: Cleaned text (whitespace normalized).
        reason: Reason for rejection (if not safe).
        matched_patterns: Names of patterns that matched.
    """

    is_safe: bool = True
    sanitized_text: str = ""
    reason: str = ""
    matched_patterns: list[str] = field(default_factory=list)


def sanitize_input(
    text: str,
    *,
    max_length: int = MAX_INPUT_LENGTH,
    strict: bool = True,
) -> SanitizeResult:
    """Validate and sanitize user input text.

    Checks:
    1. Non-empty input
    2. Length within limits
    3. No prompt injection patterns
    4. Whitespace normalization

    Args:
        text: Raw user input.
        max_length: Maximum allowed character length.
        strict: If True, reject on pattern match. If False, log warning only.

    Returns:
        SanitizeResult with safety status and cleaned text.
    """
    # Check empty
    if not text or not text.strip():
        return SanitizeResult(is_safe=False, reason="Empty input")

    # Normalize whitespace
    cleaned = " ".join(text.split())

    # Check length
    if len(cleaned) > max_length:
        logger.warning(
            "input_too_long",
            length=len(cleaned),
            max_length=max_length,
        )
        return SanitizeResult(
            is_safe=False,
            sanitized_text=cleaned[:max_length],
            reason=f"Input exceeds maximum length ({len(cleaned)} > {max_length})",
        )

    # Check injection patterns
    matched: list[str] = []
    for pattern, name in _COMPILED_PATTERNS:
        if pattern.search(cleaned):
            matched.append(name)

    if matched:
        logger.warning(
            "injection_detected",
            patterns=matched,
            text_preview=cleaned[:50],
        )
        if strict:
            return SanitizeResult(
                is_safe=False,
                sanitized_text=cleaned,
                reason=f"Prompt injection detected: {', '.join(matched)}",
                matched_patterns=matched,
            )

    logger.debug("input_sanitized", length=len(cleaned))
    return SanitizeResult(is_safe=True, sanitized_text=cleaned)


def strip_control_chars(text: str) -> str:
    """Remove non-printable control characters (except newline/tab).

    Args:
        text: Input text.

    Returns:
        Text with control characters removed.
    """
    return "".join(
        c for c in text
        if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) != 127)
    )
