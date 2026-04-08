"""LangGraph callback handler for real-time trace streaming.

This module is lazily loaded -- it will only import langgraph when the
class is actually instantiated. Install with ``pip install semantic-trace[langgraph]``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from semantic_trace.core.schema import ActionType, IntentInvariant, Span
from semantic_trace.core.serializer import write_metadata_to_jsonl, write_span_to_jsonl

try:
    from langgraph.callbacks.base import BaseCallbackHandler

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    BaseCallbackHandler = object  # type: ignore[misc,assignment]


class _SpanContext:
    """Internal container for in-progress span metadata."""

    __slots__ = (
        "action_type",
        "input_data",
        "invariants",
        "parent_id",
        "span_id",
        "start_time",
    )

    def __init__(
        self,
        span_id: uuid.UUID,
        parent_id: uuid.UUID | None,
        start_time: datetime,
        action_type: ActionType,
        input_data: dict[str, Any],
        invariants: list[IntentInvariant],
    ) -> None:
        self.span_id = span_id
        self.parent_id = parent_id
        self.start_time = start_time
        self.action_type = action_type
        self.input_data = input_data
        self.invariants = invariants


class TraceCallbackHandler(BaseCallbackHandler):
    """LangGraph callback handler that streams execution to a JSONL trace file.

    Intercepts LLM and tool call events, converts them to Span objects,
    and appends them to the trace file in real-time.

    Example:
        >>> handler = TraceCallbackHandler(
        ...     trace_file="traces/run.jsonl",
        ...     session_id="session-1",
        ...     agent_name="my-agent",
        ... )
        >>> graph = create_react_agent(..., callbacks=[handler])
    """

    def __init__(
        self,
        trace_file: str | Path,
        session_id: str,
        agent_name: str,
        trace_id: uuid.UUID | None = None,
        default_invariants: list[IntentInvariant] | None = None,
    ) -> None:
        """Initialize the callback handler.

        Args:
            trace_file: Path to the JSONL file to write spans to.
            session_id: Unique identifier for the agent session.
            agent_name: Human-readable name of the agent.
            trace_id: Optional explicit trace UUID. Generated if omitted.
            default_invariants: Invariants to attach to every span by default.

        Raises:
            ImportError: If langgraph is not installed.
        """
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required to use TraceCallbackHandler. "
                "Install it with: pip install semantic-trace[langgraph]"
            )

        super().__init__()

        from semantic_trace.core.schema import TraceMetadata, TraceModel

        self.trace_file = Path(trace_file)
        self._trace_id = trace_id or uuid.uuid4()
        self._session_id = session_id
        self._agent_name = agent_name
        self._default_invariants = default_invariants or []
        self._active_spans: dict[str, _SpanContext] = {}

        metadata = TraceMetadata(
            trace_id=self._trace_id,
            session_id=self._session_id,
            agent_name=self._agent_name,
        )
        write_metadata_to_jsonl(self.trace_file, TraceModel(metadata=metadata))

    def _finalize_span(self, run_id: str, output_data: dict[str, Any]) -> None:
        """Complete an in-progress span and write it to the trace file.

        Args:
            run_id: The LangGraph run identifier.
            output_data: The captured output from the LLM or tool call.
        """
        ctx = self._active_spans.pop(run_id, None)
        if ctx is None:
            return

        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - ctx.start_time).total_seconds() * 1000

        span = Span(
            span_id=ctx.span_id,
            parent_id=ctx.parent_id,
            trace_id=self._trace_id,
            timestamp=ctx.start_time,
            action_type=ctx.action_type,
            input_data=ctx.input_data,
            output_data=output_data,
            duration_ms=duration_ms,
            attached_invariants=ctx.invariants,
        )

        write_span_to_jsonl(self.trace_file, span)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture the start of an LLM call."""
        self._active_spans[run_id] = _SpanContext(
            span_id=uuid.uuid4(),
            parent_id=uuid.UUID(parent_run_id) if parent_run_id else None,
            start_time=datetime.now(timezone.utc),
            action_type=ActionType.LLM_CALL,
            input_data={"prompts": prompts, "serialized": serialized},
            invariants=list(self._default_invariants),
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Capture the end of an LLM call and finalize the span."""
        output_data: dict[str, Any] = {}
        if hasattr(response, "generations"):
            output_data["generations"] = [
                [{"text": gen.text, "type": gen.type} for gen in gen_list]
                for gen_list in response.generations
            ]
        if hasattr(response, "llm_output"):
            output_data["llm_output"] = response.llm_output

        self._finalize_span(run_id, output_data)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: str,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture the start of a tool call."""
        self._active_spans[run_id] = _SpanContext(
            span_id=uuid.uuid4(),
            parent_id=uuid.UUID(parent_run_id) if parent_run_id else None,
            start_time=datetime.now(timezone.utc),
            action_type=ActionType.TOOL_CALL,
            input_data={"input_str": input_str, "serialized": serialized},
            invariants=list(self._default_invariants),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: str,
        **kwargs: Any,
    ) -> None:
        """Capture the end of a tool call and finalize the span."""
        output_data: dict[str, Any] = {
            "output": str(output) if not isinstance(output, dict) else output
        }
        self._finalize_span(run_id, output_data)


__all__ = ["TraceCallbackHandler"]
