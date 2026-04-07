"""Tests for replay engines."""

import uuid
from pathlib import Path

from agent_trace import (
    ActionType,
    IntentInvariant,
    InvariantType,
    Span,
    TraceMetadata,
    TraceModel,
)
from agent_trace.core.serializer import write_metadata_to_jsonl, write_span_to_jsonl
from agent_trace.engine.replay import mechanical_replay, semantic_replay, validate_trace


def write_test_trace(
    path: Path,
    meta: TraceMetadata | None = None,
    spans: list[Span] | None = None,
) -> TraceModel:
    meta = meta or TraceMetadata(session_id="test", agent_name="test")
    trace = TraceModel(metadata=meta)
    write_metadata_to_jsonl(path, trace)
    for span in spans or []:
        write_span_to_jsonl(path, span)
    return trace


class TestMechanicalReplay:
    def test_valid_trace_no_errors(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        span = Span(trace_id=meta.trace_id, action_type=ActionType.LLM_CALL)
        write_test_trace(tmp_path / "valid.jsonl", meta, [span])

        errors = mechanical_replay(tmp_path / "valid.jsonl")
        assert errors == []

    def test_mismatched_trace_id(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        span = Span(
            trace_id=uuid.uuid4(),
            action_type=ActionType.LLM_CALL,
        )
        write_test_trace(tmp_path / "bad.jsonl", meta, [span])

        errors = mechanical_replay(tmp_path / "bad.jsonl")
        assert len(errors) == 1
        assert "mismatched trace_id" in errors[0]

    def test_missing_parent_reference(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        parent_id = uuid.uuid4()
        span = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            parent_id=parent_id,
        )
        write_test_trace(tmp_path / "orphan.jsonl", meta, [span])

        errors = mechanical_replay(tmp_path / "orphan.jsonl")
        assert len(errors) == 1
        assert "missing parent" in errors[0]

    def test_valid_parent_reference(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        parent = Span(trace_id=meta.trace_id, action_type=ActionType.AGENT_STEP)
        child = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            parent_id=parent.span_id,
        )
        write_test_trace(tmp_path / "nested.jsonl", meta, [parent, child])

        errors = mechanical_replay(tmp_path / "nested.jsonl")
        assert errors == []


class TestSemanticReplay:
    def test_passing_invariant(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        inv = IntentInvariant(
            id="has-response",
            description="must have response",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"response"'},
        )
        span = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            output_data={"response": "hello"},
            attached_invariants=[inv],
        )
        write_test_trace(tmp_path / "pass.jsonl", meta, [span])

        report = semantic_replay(tmp_path / "pass.jsonl")
        assert report.is_clean
        assert len(report.violations) == 0
        assert report.pass_rate == 1.0

    def test_failing_invariant(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        inv = IntentInvariant(
            id="has-summary",
            description="must have summary",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"summary"'},
        )
        span = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            output_data={"response": "no summary here"},
            attached_invariants=[inv],
        )
        write_test_trace(tmp_path / "fail.jsonl", meta, [span])

        report = semantic_replay(tmp_path / "fail.jsonl")
        assert not report.is_clean
        assert len(report.violations) == 1
        assert report.violations[0].invariant_id == "has-summary"

    def test_no_invariants_empty_report(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        span = Span(trace_id=meta.trace_id, action_type=ActionType.LLM_CALL)
        write_test_trace(tmp_path / "empty.jsonl", meta, [span])

        report = semantic_replay(tmp_path / "empty.jsonl")
        assert report.is_clean
        assert report.total_invariants == 0
        assert report.pass_rate == 1.0


class TestValidateTrace:
    def test_combined_validation_pass(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        inv = IntentInvariant(
            id="check",
            description="check",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"response"'},
        )
        span = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            output_data={"response": "ok"},
            attached_invariants=[inv],
        )
        write_test_trace(tmp_path / "good.jsonl", meta, [span])

        report = validate_trace(tmp_path / "good.jsonl")
        assert report.is_clean
        assert report.structural_errors == []
        assert report.violations == []

    def test_combined_validation_fail(self, tmp_path: Path):
        meta = TraceMetadata(session_id="test", agent_name="test")
        span = Span(
            trace_id=uuid.uuid4(),
            action_type=ActionType.LLM_CALL,
            output_data={"response": "ok"},
        )
        write_test_trace(tmp_path / "bad.jsonl", meta, [span])

        report = validate_trace(tmp_path / "bad.jsonl")
        assert not report.is_clean
        assert len(report.structural_errors) > 0
