"""Small language helpers shared by text/audio backends."""

from __future__ import annotations

from typing import Optional


_AUTO_VALUES = {"", "auto", "detect", "default"}


def normalize_auto_language(language: Optional[str]) -> Optional[str]:
    """Return None when the caller asked the backend to auto-detect."""
    if language is None:
        return None
    lang = str(language).strip()
    if lang.lower() in _AUTO_VALUES:
        return None
    return lang


def detect_zh_en(text: str, language: Optional[str] = None) -> str:
    """Detect the TTS language used by bilingual Matcha-style backends.

    The zh-en Matcha model can handle embedded English in Chinese text. For
    mixed input we therefore anchor to ``zh`` when any CJK character exists,
    and only return ``en`` for pure Latin/non-CJK input.
    """
    explicit = normalize_auto_language(language)
    if explicit:
        lowered = explicit.lower()
        if lowered in {"chinese", "mandarin", "cn", "zh-cn", "zh_hans"}:
            return "zh"
        if lowered in {"english", "en-us", "en_us", "us"}:
            return "en"
        return explicit

    for ch in text:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            return "zh"
    return "en"
