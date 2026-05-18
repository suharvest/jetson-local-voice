"""Built-in AppModes shipped with the agent.

Each mode lives in its own file. Users adding custom modes typically
create their own package and `modes.register(MyMode())` in their app.
"""
from .chat import ChatMode
from .interpreter import InterpreterMode
from .monologue import MonologueMode
from .transcribe import TranscribeMode

__all__ = ["ChatMode", "InterpreterMode", "MonologueMode", "TranscribeMode"]
