"""Tests for semantic-trace schema models."""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from semantic_trace import (
    ActionType,
    IntentInvariant,
    InvariantResult,
    InvariantType,
    ReplayReport,
    Span,
    TraceMetadata,
    TraceModel,
)
from semantic_trace.engine.invariants import InvariantViolation


class TestInvariantType:
    def test_enum_values(self):
        assert InvariantType.SCHEMA_MATCH.value == "SCHEMA_MATCH"
        assert InvariantType.SUBSTRING_CHECK.value == "SUBSTRING_CHECK"
        assert InvariantType.LLM_AS_JUDGE.value == "LLM_AS_JUDGE"
        assert InvariantType.CUSTOM.value == "CUSTOM"


class TestActionType:
    def test_enum_values(self):
        assert ActionType.LLM_CALL.value == "llm_call"
        assert ActionType.TOOL_CALL.value == "tool_call"
        assert ActionType.AGENT_STEP.value == "agent_step"
        assert ActionType.CUSTOM.value == "custom"


class TestTraceMetadata:
    def test_auto_generated_fields(self):
        meta = TraceMetadata(session_id="s1", agent_name="test")
        assert isinstance(meta.trace_id, uuid.UUID)
        assert isinstance(meta.start_time, datetime)
        assert meta.end_time is None

    def test_explicit_fields(self):
        tid = uuid.uuid4()
        meta = TraceMetadata(
            trace_id=tid,
            session_id="s1",
            agent_name="test",
            end_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert meta.trace_id == tid
        assert meta.end_time is not None


class TestIntentInvariant:
    def test_valid_invariant(self):
        inv = IntentInvariant(
            id="test",
            description="test desc",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": "hello"},
            fidelity_threshold=0.9,
        )
        assert inv.id == "test"
        assert inv.fidelity_threshold == 0.9

    def test_threshold_validation(self):
        with pytest.raises(ValidationError):
            IntentInvariant(
                id="bad",
                description="bad",
                invariant_type=InvariantType.SUBSTRING_CHECK,
                fidelity_threshold=1.5,
            )

    def test_default_config(self):
        inv = IntentInvariant(
            id="test",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
        )
        assert inv.config == {}


class TestSpan:
    def test_auto_generated_fields(self):
        tid = uuid.uuid4()
        span = Span(trace_id=tid, action_type=ActionType.LLM_CALL)
        assert isinstance(span.span_id, uuid.UUID)
        assert span.trace_id == tid
        assert span.attached_invariants == []
        assert span.invariant_results is None

    def test_full_span(self):
        tid = uuid.uuid4()
        span = Span(
            trace_id=tid,
            action_type=ActionType.TOOL_CALL,
            input_data={"query": "hello"},
            output_data={"result": "world"},
            duration_ms=150.0,
        )
        assert span.duration_ms == 150.0
        assert span.output_data["result"] == "world"


class TestTraceModel:
    def test_empty_trace(self):
        meta = TraceMetadata(session_id="s1", agent_name="test")
        trace = TraceModel(metadata=meta)
        assert trace.spans == []

    def test_trace_with_spans(self):
        meta = TraceMetadata(session_id="s1", agent_name="test")
        span = Span(trace_id=meta.trace_id, action_type=ActionType.LLM_CALL)
        trace = TraceModel(metadata=meta, spans=[span])
        assert len(trace.spans) == 1


class TestInvariantResult:
    def test_passing_result(self):
        r = InvariantResult("inv1", "span1", 0.95, 0.9, True)
        assert r.passed
        assert r.score == 0.95

    def test_failing_result(self):
        r = InvariantResult("inv1", "span1", 0.5, 0.9, False)
        assert not r.passed


class TestReplayReport:
    def test_empty_report_is_clean(self):
        report = ReplayReport(
            trace_file="test.jsonl",
            trace_id=uuid.uuid4(),
            agent_name="test",
            total_spans=1,
            total_invariants=0,
            violations=[],
            results=[],
        )
        assert report.is_clean
        assert report.pass_rate == 1.0
        assert len(report) == 0

    def test_report_with_violations(self):
        report = ReplayReport(
            trace_file="test.jsonl",
            trace_id=uuid.uuid4(),
            agent_name="test",
            total_spans=1,
            total_invariants=2,
            violations=[
                InvariantViolation("s1", "inv1", 0.9, 0.5),
            ],
            results=[
                InvariantResult("inv1", "s1", 0.5, 0.9, False),
                InvariantResult("inv2", "s1", 1.0, 0.9, True),
            ],
        )
        assert not report.is_clean
        assert len(report) == 1
        assert report.pass_rate == 0.5
        assert report[0].invariant_id == "inv1"

    def test_report_is_iterable(self):
        violations = [
            InvariantViolation("s1", "inv1", 0.9, 0.3),
            InvariantViolation("s2", "inv2", 0.9, 0.1),
        ]
        report = ReplayReport(
            trace_file="test.jsonl",
            trace_id=uuid.uuid4(),
            agent_name="test",
            total_spans=2,
            total_invariants=2,
            violations=violations,
            results=[],
        )
        collected = list(report)
        assert len(collected) == 2

    def test_summary_string(self):
        report = ReplayReport(
            trace_file="test.jsonl",
            trace_id=uuid.uuid4(),
            agent_name="my-agent",
            total_spans=5,
            total_invariants=10,
            violations=[],
            results=[],
        )
        summary = report.summary()
        assert "my-agent" in summary
        assert "5" in summary
        assert "ALL CLEAR" in summary

    def test_structural_errors_make_report_dirty(self):
        report = ReplayReport(
            trace_file="test.jsonl",
            trace_id=uuid.uuid4(),
            agent_name="test",
            total_spans=1,
            total_invariants=0,
            violations=[],
            results=[],
            structural_errors=["bad span"],
        )
        assert not report.is_clean
        assert bool(report)
