"""Concrete WakeSource implementations.

Only HTTPWakeSource is fully wired (it relies on the debug_dashboard
plugin's /api/control/wake route). The others are documented stubs
intended for users to subclass or replace.
"""
from __future__ import annotations

from .http import HTTPWakeSource
from .stub_ext import (
    LocalKeywordWakeSource,
    MQTTWakeSource,
    SerialWakeSource,
)

__all__ = [
    "HTTPWakeSource",
    "MQTTWakeSource",
    "SerialWakeSource",
    "LocalKeywordWakeSource",
]
