"""Test bootstrap: expose MultiModeApp under a stable import path.

The standard voice app lives under `agent/apps/multi_mode/`, which is
outside the `openvoicestream_agent` namespace. To keep tests
independent of cwd, we register it at
`openvoicestream_agent.apps_multi_mode_shim`. The legacy
`apps_dialogue_shim` alias is kept for any pre-existing tests that
still import DialogueApp (they should treat it as MultiModeApp now).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `apps.*` importable for the CLI loader too.
_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from apps.multi_mode.app import MultiModeApp  # noqa: E402

import openvoicestream_agent as _ovs  # noqa: E402

import types as _types  # noqa: E402

# Primary stable alias.
shim_mod_name = "openvoicestream_agent.apps_multi_mode_shim"
_shim = _types.ModuleType(shim_mod_name)
_shim.MultiModeApp = MultiModeApp
sys.modules[shim_mod_name] = _shim

# Backwards-compat alias: any test that still imports DialogueApp from
# the old shim path gets MultiModeApp instead. The two satisfy the
# same external contract (BaseApp + on_user_utterance via ChatMode).
legacy_shim = "openvoicestream_agent.apps_dialogue_shim"
_legacy = _types.ModuleType(legacy_shim)
_legacy.DialogueApp = MultiModeApp
sys.modules[legacy_shim] = _legacy


@pytest.fixture
def multi_mode_cls():
    return MultiModeApp


@pytest.fixture
def dialogue_cls():
    # Back-compat fixture name; returns MultiModeApp now.
    return MultiModeApp
