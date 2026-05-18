"""Conversation state machine for BaseApp.

A single ConvState value reflects what the agent is currently doing in a
turn. Transitions are driven from BaseApp's _update_vad / _dispatch_one
single-loop coroutines, so no lock is required.

States:
    IDLE       - quiet; nothing happening
    LISTENING  - client VAD has detected speech; audio is streaming to SLV
    THINKING   - utterance final received, awaiting LLM tokens / TTS
    SPEAKING   - first TTS audio frame has played; playback in progress
    BARGED_IN  - user spoke while assistant was SPEAKING; playback aborted
    SLEEPING   - pipeline_mode != always_on: agent is gated; mic audio is
                 dropped until an external wake signal (wake_word /
                 push_to_talk) transitions the agent out.
"""
from __future__ import annotations

from enum import Enum


class ConvState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    BARGED_IN = "barged_in"
    SLEEPING = "sleeping"


__all__ = ["ConvState"]
