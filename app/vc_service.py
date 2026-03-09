"""OpenVoice voice conversion service (optional, disabled by default).

Requires ENABLE_VC=true environment variable to activate.
Uses OpenVoice ToneColorConverter with CUDA for voice cloning.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENABLE_VC = os.environ.get("ENABLE_VC", "false").lower() in ("true", "1", "yes")

_converter = None


def is_enabled() -> bool:
    return ENABLE_VC


def is_ready() -> bool:
    return _converter is not None


def get_converter():
    """Lazy-init OpenVoice ToneColorConverter."""
    global _converter
    if _converter is not None:
        return _converter

    if not ENABLE_VC:
        raise RuntimeError("Voice conversion is disabled. Set ENABLE_VC=true to enable.")

    # TODO: implement OpenVoice ToneColorConverter initialization
    # from openvoice import ToneColorConverter
    # _converter = ToneColorConverter(...)
    raise NotImplementedError("OpenVoice VC not yet implemented for Jetson deployment")


def convert(wav_bytes: bytes) -> bytes:
    """Convert voice timbre of input audio."""
    converter = get_converter()
    # TODO: implement conversion
    raise NotImplementedError("OpenVoice VC not yet implemented")
