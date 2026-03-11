"""Voice whitelist — enforce allowed voices only.

Restricts TTS synthesis to a predefined set of whitelisted
voice IDs. Rejects unauthorized voice cloning attempts.

Usage:
    validate_voice("speaker_1")  # OK
    validate_voice("custom_voice")  # raises VoiceNotAllowedError
"""

from __future__ import annotations

from src.config import AppConfig
from src.log import get_logger

logger = get_logger(__name__)


class VoiceNotAllowedError(Exception):
    """Raised when a non-whitelisted voice is requested."""


def get_allowed_voices(config: AppConfig | None = None) -> list[str]:
    """Get the list of currently allowed voice IDs.

    Args:
        config: App config. If None, uses defaults.

    Returns:
        List of allowed voice ID strings.
    """
    if config is not None:
        return list(config.security.whitelisted_voices)
    return ["speaker_1", "speaker_2", "speaker_3"]


def validate_voice(
    voice_id: str,
    config: AppConfig | None = None,
) -> str:
    """Validate that a voice ID is in the whitelist.

    Args:
        voice_id: Requested voice ID.
        config: App config for whitelist lookup.

    Returns:
        The validated voice_id (unchanged).

    Raises:
        VoiceNotAllowedError: If voice is not whitelisted.
    """
    allowed = get_allowed_voices(config)

    if voice_id not in allowed:
        logger.warning(
            "voice_rejected",
            voice_id=voice_id,
            allowed=allowed,
        )
        raise VoiceNotAllowedError(
            f"Voice '{voice_id}' is not allowed. "
            f"Allowed voices: {', '.join(allowed)}"
        )

    logger.debug("voice_validated", voice_id=voice_id)
    return voice_id


def validate_ref_audio(
    ref_audio_path: str | None,
    *,
    allow_cloning: bool = False,
) -> None:
    """Validate reference audio for voice cloning.

    By default, voice cloning with custom reference audio is
    disabled for security (voice impersonation risk).

    Args:
        ref_audio_path: Path to reference audio file.
        allow_cloning: Whether to allow custom voice cloning.

    Raises:
        VoiceNotAllowedError: If cloning is not allowed.
    """
    if ref_audio_path and not allow_cloning:
        logger.warning(
            "clone_rejected",
            ref_audio_path="[REDACTED]",
        )
        raise VoiceNotAllowedError(
            "Voice cloning with custom reference audio is disabled. "
            "Use one of the whitelisted voices instead."
        )
