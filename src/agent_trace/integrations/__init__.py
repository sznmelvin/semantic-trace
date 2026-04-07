"""Third-party integrations for agent-trace.

Optional integrations that require additional dependencies.
Install with ``pip install agent-trace[langgraph]``.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from agent_trace.integrations.langgraph import (  # noqa: F401
        TraceCallbackHandler,
    )

    __all__.append("TraceCallbackHandler")
except ImportError:
    pass
