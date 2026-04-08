"""Core data models and serialization for semantic-trace.

Provides Pydantic models (TraceModel, Span, IntentInvariant), the Trace
context manager, and JSONL serialization utilities.
"""

from semantic_trace.core.schema import (
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
from semantic_trace.core.serializer import (
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
