"""Pydantic v2 data models and Trace context manager for semantic-trace.

All models use strict typing and Pydantic v2 validation. No mutable defaults.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from semantic_trace.engine.invariants import InvariantViolation


class InvariantType(str, Enum):
    """Enumeration of built-in invariant checker types.

    Attributes:
        SCHEMA_MATCH: Validates output against a Pydantic-compatible schema.
        SUBSTRING_CHECK: Checks for a target substring in the JSON-serialized output.
        LLM_AS_JUDGE: Uses an LLM to semantically evaluate the output.
        CUSTOM: Placeholder for user-defined checkers via the ABC.
    """

    SCHEMA_MATCH = "SCHEMA_MATCH"
    SUBSTRING_CHECK = "SUBSTRING_CHECK"
    LLM_AS_JUDGE = "LLM_AS_JUDGE"
    CUSTOM = "CUSTOM"


class ActionType(str, Enum):
    """Enumeration of span action types.

    Attributes:
        LLM_CALL: A call to a language model.
        TOOL_CALL: A call to an external tool.
        AGENT_STEP: A high-level agent reasoning step.
        CUSTOM: A user-defined action type.
    """

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    AGENT_STEP = "agent_step"
    CUSTOM = "custom"


class TraceMetadata(BaseModel):
    """Metadata header for a trace.

    Written as the first line of every JSONL trace file.

    Attributes:
        trace_id: Unique identifier for this trace.
        session_id: Identifier for the agent session that produced this trace.
        agent_name: Human-readable name of the agent.
        start_time: UTC timestamp when the trace was created.
        end_time: UTC timestamp when the trace was finalized. None until closed.
    """

    trace_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: str
    agent_name: str
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None


class IntentInvariant(BaseModel):
    """A declarative intent assertion attached to a span.

    Invariants express what the span's output *should* satisfy. They are
    evaluated during semantic replay.

    Attributes:
        id: Unique identifier for this invariant.
        description: Human-readable description of the intent.
        invariant_type: The checker type to use.
        config: Rule parameters passed to the checker (e.g., ``{"substring": "..."}``).
        fidelity_threshold: Minimum score (0.0-1.0) for the check to pass.
    """

    id: str
    description: str
    invariant_type: InvariantType
    config: dict[str, Any] = Field(default_factory=dict)
    fidelity_threshold: float = Field(ge=0.0, le=1.0, default=1.0)


class Span(BaseModel):
    """A single unit of execution within a trace.

    Each span represents one atomic action (LLM call, tool invocation, etc.)
    and may carry attached invariants for later verification.

    Attributes:
        span_id: Unique identifier for this span.
        parent_id: Optional parent span UUID for nested execution.
        trace_id: The trace this span belongs to.
        timestamp: UTC timestamp when the span started.
        action_type: The kind of action this span represents.
        input_data: The input payload for the action.
        output_data: The output payload produced by the action.
        duration_ms: Execution duration in milliseconds, if available.
        attached_invariants: Intent invariants to verify during replay.
        invariant_results: Post-execution scores, populated by semantic replay.
    """

    span_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: uuid.UUID | None = None
    trace_id: uuid.UUID
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action_type: ActionType
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    attached_invariants: list[IntentInvariant] = Field(default_factory=list)
    invariant_results: dict[str, float] | None = None


class TraceModel(BaseModel):
    """A complete trace consisting of metadata and ordered spans.

    This is the core Pydantic model. For the high-level context manager API,
    use ``Trace`` instead.

    Attributes:
        metadata: The trace header with session and agent information.
        spans: Ordered list of spans captured during execution.
    """

    metadata: TraceMetadata
    spans: list[Span] = Field(default_factory=list)


@dataclass
class InvariantResult:
    """Result of a single invariant check during replay.

    Attributes:
        invariant_id: Identifier of the invariant that was checked.
        span_id: UUID of the span that was evaluated.
        score: The actual score returned by the checker (0.0-1.0).
        threshold: The minimum score required for the check to pass.
        passed: Whether the score meets or exceeds the threshold.
    """

    invariant_id: str
    span_id: str
    score: float
    threshold: float
    passed: bool


@dataclass
class ReplayReport:
    """Comprehensive report from semantic or mechanical replay.

    Provides summary statistics, violation details, and a human-readable
    summary string. Iterable over violations for backward compatibility.

    Attributes:
        trace_file: Path to the trace file that was replayed.
        trace_id: UUID of the trace.
        agent_name: Name of the agent that produced the trace.
        total_spans: Number of spans in the trace.
        total_invariants: Total number of invariant checks performed.
        violations: List of invariant violations (score below threshold).
        results: All invariant check results (passing and failing).
        structural_errors: Structural errors from mechanical validation.
    """

    trace_file: Path
    trace_id: uuid.UUID
    agent_name: str
    total_spans: int
    total_invariants: int
    violations: list[InvariantViolation]
    results: list[InvariantResult]
    structural_errors: list[str] = field(default_factory=list)

    def __iter__(self):
        return iter(self.violations)

    def __len__(self) -> int:
        return len(self.violations)

    def __getitem__(self, index: int):
        return self.violations[index]

    def __bool__(self) -> bool:
        return bool(self.violations) or bool(self.structural_errors)

    @property
    def is_clean(self) -> bool:
        """True if no violations and no structural errors."""
        return not self.violations and not self.structural_errors

    @property
    def pass_rate(self) -> float:
        """Fraction of invariant checks that passed."""
        if self.total_invariants == 0:
            return 1.0
        passed = sum(1 for r in self.results if r.passed)
        return passed / self.total_invariants

    def summary(self) -> str:
        """Return a human-readable summary with boxed layout."""
        width = 52
        border_h = "─" * width
        border_v = "│"

        pct = int(self.pass_rate * 100)
        if self.is_clean:
            status = "ALL CLEAR"
            status_prefix = ""
        else:
            status = f"{len(self.violations)} VIOLATIONS"
            status_prefix = "✗ "

        trace_id = str(self.trace_id)[:36]
        file_name = self.file_basename[:36]

        agent_name = f"Replay Report: {self.agent_name}"
        agent_pad = width - len(agent_name) - 2

        trace_line = f"Trace ID   : {trace_id}"
        trace_pad = width - len(trace_line) - 2

        file_line = f"File       : {file_name}"
        file_pad = width - len(file_line) - 2

        spans_line = f"Spans      : {self.total_spans}"
        spans_pad = width - len(spans_line) - 2

        inv_line = f"Invariants : {self.total_invariants} checked"
        inv_pad = width - len(inv_line) - 2

        status_line = f"Status     : {status_prefix}{status}"
        status_pad = width - len(status_line) - 2

        pass_line = f"Pass rate  : {pct}%"
        pass_pad = width - len(pass_line) - 2

        lines = [
            f"┌{border_h}┐",
            f"{border_v} {agent_name}{' ' * agent_pad} {border_v}",
            f"├{border_h}┤",
            f"{border_v} {trace_line}{' ' * trace_pad} {border_v}",
            f"{border_v} {file_line}{' ' * file_pad} {border_v}",
            f"{border_v} {spans_line}{' ' * spans_pad} {border_v}",
            f"{border_v} {inv_line}{' ' * inv_pad} {border_v}",
            f"├{border_h}┤",
            f"{border_v} {status_line}{' ' * status_pad} {border_v}",
            f"{border_v} {pass_line}{' ' * pass_pad} {border_v}",
            f"└{border_h}┘",
        ]

        return "\n".join(lines)

    @property
    def file_basename(self) -> str:
        """Return just the filename without full path."""
        if not self.trace_file:
            return "N/A"
        import os

        return os.path.basename(self.trace_file)

    def print_violations(self) -> None:
        """Print all violations with helpful context."""
        if not self.violations and not self.structural_errors:
            return

        if self.structural_errors:
            print("Structural errors:")
            for err in self.structural_errors:
                print(f"  - {err}")
            print()

        if self.violations:
            print("Invariant Violations:")
            print("─" * 52)

            for v in self.violations:
                span_short = _shorten_uuid(v.span_id)
                msg = _format_violation_message(
                    v.invariant_id,
                    v.expected_score,
                    v.actual_score,
                )
                print(
                    f"✗ {v.invariant_id:<14} → Span {span_short} ({v.actual_score:.2f} / {v.expected_score:.2f})"
                )
                print(f"  {msg}")


class Trace:
    """Context manager for capturing agent traces.

    Creates a trace header, collects spans, and writes to a JSONL file
    in real-time. On exit, finalizes the trace with an end timestamp.

    Usage:
        with Trace(name="my-agent", output_file="traces/run.jsonl") as trace:
            trace.add_span(span)

    Or with default invariants attached to every span:
        invariants = [IntentInvariant(...)]
        with Trace(name="my-agent", invariants=invariants, output_file="traces/run.jsonl") as trace:
            # spans automatically get the invariants
            trace.add_span(span)

    For backward compatibility, can also be initialized with a metadata object:
        trace = Trace(metadata=TraceMetadata(...))
        trace.add_span(span)
        trace.save("traces/run.jsonl")

    Args:
        name: Human-readable name of the agent (required unless metadata is provided).
        metadata: Pre-built TraceMetadata (alternative to name).
        invariants: Default invariants to attach to every span.
        output_file: Path to the JSONL file for real-time streaming.
        session_id: Identifier for the agent session. Auto-generated if omitted.
        trace_id: Explicit trace UUID. Auto-generated if omitted.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        metadata: TraceMetadata | None = None,
        invariants: list[IntentInvariant] | None = None,
        output_file: str | Path | None = None,
        session_id: str | None = None,
        trace_id: uuid.UUID | None = None,
    ) -> None:
        if name is not None and metadata is not None:
            raise ValueError("Provide either 'name' or 'metadata', not both")

        if metadata is not None:
            self._model = TraceModel(metadata=metadata, spans=[])
            self._output_file: Path | None = None
            self._default_invariants: list[IntentInvariant] = []
        elif name is not None:
            self._model = TraceModel(
                metadata=TraceMetadata(
                    trace_id=trace_id or uuid.uuid4(),
                    session_id=session_id or f"session-{uuid.uuid4().hex[:8]}",
                    agent_name=name,
                ),
                spans=[],
            )
            self._output_file = Path(output_file).resolve() if output_file else None
            self._default_invariants = list(invariants) if invariants else []
        else:
            raise ValueError("Either 'name' or 'metadata' must be provided")

        self._active = False

    def __enter__(self) -> Trace:
        if self._active:
            raise RuntimeError("Trace is already active. Create a new instance.")
        self._active = True
        if self._output_file:
            from semantic_trace.core.serializer import write_metadata_to_jsonl

            write_metadata_to_jsonl(self._output_file, self._model)
        return self

    def __exit__(self, *args: Any) -> bool:
        self._active = False
        self._model.metadata.end_time = datetime.now(timezone.utc)
        if self._output_file:
            from semantic_trace.core.serializer import write_metadata_to_jsonl

            write_metadata_to_jsonl(self._output_file, self._model)
        return False

    def __repr__(self) -> str:
        status = "active" if self._active else "inactive"
        return (
            f"Trace(name={self._model.metadata.agent_name!r}, "
            f"trace_id={self.trace_id!r}, "
            f"spans={len(self.spans)}, "
            f"status={status!r})"
        )

    @property
    def model(self) -> TraceModel:
        """The underlying TraceModel. Modifications affect the trace."""
        return self._model

    @property
    def trace_id(self) -> uuid.UUID:
        return self._model.metadata.trace_id

    @property
    def session_id(self) -> str:
        return self._model.metadata.session_id

    @property
    def spans(self) -> list[Span]:
        return list(self._model.spans)

    def add_span(self, span: Span) -> None:
        """Add a span to the trace.

        Attaches default invariants and writes to the output file if
        the context manager is active.

        Args:
            span: The span to add. Must have a matching trace_id.

        Raises:
            ValueError: If the span's trace_id does not match this trace.
        """
        if span.trace_id != self.trace_id:
            raise ValueError(
                f"Span trace_id {span.trace_id} does not match trace {self.trace_id}"
            )
        span.attached_invariants.extend(self._default_invariants)
        self._model.spans.append(span)
        if self._active and self._output_file:
            from semantic_trace.core.serializer import write_span_to_jsonl

            write_span_to_jsonl(self._output_file, span)

    def save(self, output_file: str | Path | None = None) -> None:
        """Save the complete trace to a JSONL file.

        Args:
            output_file: Path to write. Uses the constructor value if omitted.

        Raises:
            ValueError: If no output file is specified.
        """
        from semantic_trace.core.serializer import (
            write_metadata_to_jsonl,
            write_span_to_jsonl,
        )

        path = Path(output_file).resolve() if output_file else self._output_file
        if not path:
            raise ValueError("No output file specified")

        self._model.metadata.end_time = datetime.now(timezone.utc)
        write_metadata_to_jsonl(path, self._model)
        for span in self._model.spans:
            write_span_to_jsonl(path, span)


def _shorten_uuid(uuid_str: str, prefix_len: int = 8) -> str:
    """Shorten UUID for display, showing only first few characters."""
    if len(uuid_str) <= 12:
        return uuid_str
    return uuid_str[:prefix_len] + "..."


def _format_violation_message(
    invariant_id: str,
    expected: float,
    actual: float,
) -> str:
    """Generate human-readable violation message based on invariant ID patterns."""
    invariant_id_lower = invariant_id.lower()

    if "summary" in invariant_id_lower:
        if actual == 0.0:
            return "Output did not contain any summary section"
        return "Summary section failed validation"

    if "citation" in invariant_id_lower or "reference" in invariant_id_lower:
        if actual == 0.0:
            return "No citations or references found in output"
        return "Citations failed validation"

    if "metadata" in invariant_id_lower:
        if actual == 0.0:
            return "Missing metadata field in output"
        return "Metadata validation failed"

    if "schema" in invariant_id_lower:
        return "Output did not match expected schema"

    if "hallucination" in invariant_id_lower:
        if actual == 0.0:
            return "Potential hallucination detected in output"
        return "Hallucination check failed"

    if "format" in invariant_id_lower:
        if actual == 0.0:
            return "Output format did not match requirements"
        return "Format validation failed"

    if actual == 0.0:
        return "Invariant check failed - expected pattern not found"
    return f"Invariant {invariant_id} returned score {actual:.2f}"
