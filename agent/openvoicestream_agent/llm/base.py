"""Abstract LLM backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class LLMBackend(ABC):
    """Streaming LLM backend."""

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]], **kw: Any) -> AsyncIterator[str]:
        """Yield text deltas (already-decoded strings) from the model."""
        # Concrete implementations must be async generators -- this stub
        # exists so the base class can be ABC-instantiated for typing.
        if False:  # pragma: no cover
            yield ""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any held network/transport resources. Default: no-op."""
        return None
