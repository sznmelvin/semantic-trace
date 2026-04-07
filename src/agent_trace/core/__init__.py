"""Core data models and serialization for agent-trace.

Provides Pydantic models (TraceModel, Span, IntentInvariant), the Trace
context manager, and JSONL serialization utilities.
"""

from agent_trace.core.schema import (
    ActionType,
    IntentInvariant,
    InvariantResult,
    InvariantType,
    ReplayReport,
    Span,
    Trace,
    TraceMetadata,
    TraceModel,
)
from agent_trace.core.serializer import (
    read_trace_from_jsonl,
    write_metadata_to_jsonl,
    write_span_to_jsonl,
)

__all__ = [
    # Models
    "ActionType",
    "IntentInvariant",
    "InvariantResult",
    "InvariantType",
    "ReplayReport",
    "Span",
    "Trace",
    "TraceMetadata",
    "TraceModel",
    # Serialization
    "read_trace_from_jsonl",
    "write_metadata_to_jsonl",
    "write_span_to_jsonl",
]
