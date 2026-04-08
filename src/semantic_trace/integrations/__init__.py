"""Third-party integrations for semantic-trace.

Optional integrations that require additional dependencies.
Install with ``pip install semantic-trace[langgraph]``.
"""

from __future__ import annotations

__all__: list[str] = []

try:
    from semantic_trace.integrations.langgraph import (  # noqa: F401
        TraceCallbackHandler,
    )

    __all__.append("TraceCallbackHandler")
except ImportError:
    pass
