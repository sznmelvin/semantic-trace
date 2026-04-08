"""semantic-trace: Semantic tracing primitive for AI agents.

Minimal, composable, and zero-bloat by design.

Quick start:
    from semantic_trace import Trace, IntentInvariant, InvariantType, semantic_replay

    invariants = [
        IntentInvariant(
            id="valid-json",
            description="Output must be valid JSON",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"summary"'},
            fidelity_threshold=1.0,
        ),
    ]

    with Trace(name="my-agent", invariants=invariants, output_file="traces/run.jsonl") as trace:
        trace.add_span(span)

    report = semantic_replay("traces/run.jsonl")
    print(report.summary())
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
from semantic_trace.engine.invariants import (
    BaseInvariantChecker,
    InvariantViolation,
    LLMAsJudgeChecker,
    SchemaInvariantChecker,
    SubstringInvariantChecker,
)
from semantic_trace.engine.replay import (
    mechanical_replay,
    semantic_replay,
    validate_trace,
)

__version__ = "0.1.0"

__all__ = [
    "ActionType",
    "BaseInvariantChecker",
    "IntentInvariant",
    "InvariantResult",
    "InvariantType",
    "InvariantViolation",
    "LLMAsJudgeChecker",
    "ReplayReport",
    "SchemaInvariantChecker",
    "Span",
    "SubstringInvariantChecker",
    "Trace",
    "TraceMetadata",
    "TraceModel",
    "__version__",
    "mechanical_replay",
    "read_trace_from_jsonl",
    "semantic_replay",
    "validate_trace",
    "write_metadata_to_jsonl",
    "write_span_to_jsonl",
]
