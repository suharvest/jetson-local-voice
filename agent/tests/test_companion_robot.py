"""CompanionRobotApp skeleton: instantiates + carries robot slots."""
from __future__ import annotations

import pytest

from openvoicestream_agent.config import Config


@pytest.fixture
def test_config():
    # Minimal config; no network calls are made until run() is awaited.
    return Config(
        slv_url="ws://127.0.0.1:1/v2v/stream",
        llm_base_url="http://127.0.0.1:1/v1",
        metadata={"dashboard_port": 0},
    )


def test_companion_robot_app_instantiates(test_config):
    from apps.companion_robot.app import CompanionRobotApp

    app = CompanionRobotApp(test_config)
    assert app.reachy is None
    # head_target_bus is the no-op stub.
    assert app.head_target_bus is not None
    assert app.head_target_bus.publish("anything") is None
    assert app.head_target_bus.get_fused_target() is None
    assert app.current_emotion == "neutral"
    assert app.motor_enabled is True


def test_companion_robot_registers_default_modes(test_config):
    from apps.companion_robot.app import CompanionRobotApp

    app = CompanionRobotApp(test_config)
    names = {m["name"] for m in app.modes.list_all()}
    # Inherits MultiModeApp's four built-in modes.
    assert {"chat", "interpreter", "monologue", "transcribe"}.issubset(names)


def test_companion_robot_subclassable(test_config):
    from apps.companion_robot.app import CompanionRobotApp

    class MyCompanion(CompanionRobotApp):
        def __init__(self, config):
            super().__init__(config)
            self.extra = "mine"

    app = MyCompanion(test_config)
    assert app.extra == "mine"
    assert app.reachy is None
    # base CompanionRobotApp invariants still hold
    assert app.current_emotion == "neutral"


def test_app_alias_resolves(test_config):
    """The CLI loader looks for `App` symbol; ensure alias works."""
    from apps.companion_robot import App

    app = App(test_config)
    assert hasattr(app, "head_target_bus")
