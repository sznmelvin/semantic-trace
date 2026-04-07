"""Tests for the Trace context manager."""

import uuid
from pathlib import Path

import pytest

from agent_trace import (
    ActionType,
    IntentInvariant,
    InvariantType,
    Span,
    Trace,
    TraceMetadata,
)
from agent_trace.core.serializer import read_trace_from_jsonl


class TestTraceContextManager:
    def test_basic_context_manager(self, tmp_path: Path):
        output = tmp_path / "trace.jsonl"
        with Trace(name="test-agent", output_file=output) as trace:
            span = Span(
                trace_id=trace.trace_id,
                action_type=ActionType.LLM_CALL,
                output_data={"response": "hello"},
            )
            trace.add_span(span)

        assert output.exists()
        loaded = read_trace_from_jsonl(output)
        assert loaded.metadata.agent_name == "test-agent"
        assert len(loaded.spans) == 1
        assert loaded.metadata.end_time is not None

    def test_default_invariants_attached(self, tmp_path: Path):
        output = tmp_path / "trace.jsonl"
        inv = IntentInvariant(
            id="test-inv",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": "test"},
        )
        with Trace(name="test-agent", invariants=[inv], output_file=output) as trace:
            span = Span(
                trace_id=trace.trace_id,
                action_type=ActionType.LLM_CALL,
            )
            trace.add_span(span)

        assert len(trace.spans[0].attached_invariants) == 1
        assert trace.spans[0].attached_invariants[0].id == "test-inv"

    def test_trace_id_access(self):
        with Trace(name="test") as trace:
            assert isinstance(trace.trace_id, uuid.UUID)

    def test_session_id_auto_generated(self):
        with Trace(name="test") as trace:
            assert trace.session_id.startswith("session-")

    def test_repr(self):
        with Trace(name="my-agent") as trace:
            r = repr(trace)
            assert "my-agent" in r
            assert "active" in r

    def test_explicit_trace_id(self):
        tid = uuid.uuid4()
        with Trace(name="test", trace_id=tid) as trace:
            assert trace.trace_id == tid

    def test_explicit_session_id(self):
        with Trace(name="test", session_id="my-session") as trace:
            assert trace.session_id == "my-session"


class TestTraceWithoutContextManager:
    def test_metadata_mode(self):
        meta = TraceMetadata(session_id="s1", agent_name="manual")
        trace = Trace(metadata=meta)
        assert trace.trace_id == meta.trace_id
        assert trace.session_id == "s1"

    def test_add_span_without_output_file(self):
        meta = TraceMetadata(session_id="s1", agent_name="manual")
        trace = Trace(metadata=meta)
        span = Span(trace_id=trace.trace_id, action_type=ActionType.LLM_CALL)
        trace.add_span(span)
        assert len(trace.spans) == 1

    def test_save(self, tmp_path: Path):
        meta = TraceMetadata(session_id="s1", agent_name="manual")
        trace = Trace(metadata=meta)
        span = Span(trace_id=trace.trace_id, action_type=ActionType.LLM_CALL)
        trace.add_span(span)

        output = tmp_path / "saved.jsonl"
        trace.save(output)
        assert output.exists()

        loaded = read_trace_from_jsonl(output)
        assert len(loaded.spans) == 1

    def test_save_no_file_raises(self):
        meta = TraceMetadata(session_id="s1", agent_name="manual")
        trace = Trace(metadata=meta)
        with pytest.raises(ValueError, match="No output file"):
            trace.save()


class TestTraceErrors:
    def test_name_and_metadata_raises(self):
        meta = TraceMetadata(session_id="s1", agent_name="test")
        with pytest.raises(ValueError, match="not both"):
            Trace(name="test", metadata=meta)

    def test_neither_name_nor_metadata_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            Trace()

    def test_mismatched_trace_id_raises(self):
        with Trace(name="test") as trace:
            bad_span = Span(
                trace_id=uuid.uuid4(),
                action_type=ActionType.LLM_CALL,
            )
            with pytest.raises(ValueError, match="does not match"):
                trace.add_span(bad_span)

    def test_double_enter_raises(self):
        with (
            Trace(name="test") as trace,
            pytest.raises(RuntimeError, match="already active"),
        ):
            trace.__enter__()
