"""Tests for JSONL serialization."""

import uuid
from pathlib import Path

import orjson
import pytest

from agent_trace import (
    ActionType,
    IntentInvariant,
    InvariantType,
    Span,
    TraceMetadata,
    TraceModel,
)
from agent_trace.core.serializer import (
    read_trace_from_jsonl,
    write_metadata_to_jsonl,
    write_span_to_jsonl,
)


@pytest.fixture
def tmp_trace_file(tmp_path: Path) -> Path:
    return tmp_path / "test.jsonl"


@pytest.fixture
def sample_trace() -> TraceModel:
    meta = TraceMetadata(session_id="test-session", agent_name="test-agent")
    return TraceModel(metadata=meta)


@pytest.fixture
def sample_span(trace_id: uuid.UUID) -> Span:
    return Span(
        trace_id=trace_id,
        action_type=ActionType.LLM_CALL,
        input_data={"prompt": "hello"},
        output_data={"response": "hi"},
        duration_ms=100.0,
    )


@pytest.fixture
def trace_id() -> uuid.UUID:
    return uuid.uuid4()


class TestWriteMetadata:
    def test_writes_metadata_line(self, tmp_trace_file: Path, sample_trace: TraceModel):
        write_metadata_to_jsonl(tmp_trace_file, sample_trace)
        assert tmp_trace_file.exists()
        content = tmp_trace_file.read_text().strip()
        assert "__metadata__" in content

    def test_creates_parent_dirs(self, tmp_path: Path, sample_trace: TraceModel):
        nested = tmp_path / "a" / "b" / "trace.jsonl"
        write_metadata_to_jsonl(nested, sample_trace)
        assert nested.exists()


class TestWriteSpan:
    def test_writes_span_line(self, tmp_trace_file: Path, sample_span: Span):
        write_span_to_jsonl(tmp_trace_file, sample_span)
        assert tmp_trace_file.exists()
        content = tmp_trace_file.read_text().strip()
        assert "span_id" in content

    def test_appends_multiple_spans(self, tmp_trace_file: Path, trace_id: uuid.UUID):
        span1 = Span(trace_id=trace_id, action_type=ActionType.LLM_CALL)
        span2 = Span(trace_id=trace_id, action_type=ActionType.TOOL_CALL)
        write_span_to_jsonl(tmp_trace_file, span1)
        write_span_to_jsonl(tmp_trace_file, span2)
        lines = tmp_trace_file.read_text().strip().splitlines()
        assert len(lines) == 2


class TestReadWriteRoundtrip:
    def test_full_roundtrip(self, tmp_trace_file: Path):
        meta = TraceMetadata(session_id="roundtrip", agent_name="test")
        trace = TraceModel(metadata=meta)

        write_metadata_to_jsonl(tmp_trace_file, trace)

        span = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            input_data={"prompt": "test"},
            output_data={"response": "ok"},
            duration_ms=50.0,
        )
        write_span_to_jsonl(tmp_trace_file, span)

        loaded = read_trace_from_jsonl(tmp_trace_file)
        assert loaded.metadata.session_id == "roundtrip"
        assert len(loaded.spans) == 1
        assert loaded.spans[0].duration_ms == 50.0

    def test_roundtrip_with_invariants(self, tmp_trace_file: Path):
        meta = TraceMetadata(session_id="inv-test", agent_name="test")
        trace = TraceModel(metadata=meta)
        write_metadata_to_jsonl(tmp_trace_file, trace)

        inv = IntentInvariant(
            id="test-inv",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": "test"},
        )
        span = Span(
            trace_id=meta.trace_id,
            action_type=ActionType.LLM_CALL,
            attached_invariants=[inv],
        )
        write_span_to_jsonl(tmp_trace_file, span)

        loaded = read_trace_from_jsonl(tmp_trace_file)
        assert len(loaded.spans[0].attached_invariants) == 1
        assert loaded.spans[0].attached_invariants[0].id == "test-inv"


class TestReadErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_trace_from_jsonl("/nonexistent/path.jsonl")

    def test_no_metadata_raises(self, tmp_path: Path):
        bad_file = tmp_path / "bad.jsonl"
        span = Span(
            trace_id=uuid.uuid4(),
            action_type=ActionType.LLM_CALL,
        )
        write_span_to_jsonl(bad_file, span)
        with pytest.raises(ValueError, match="No metadata"):
            read_trace_from_jsonl(bad_file)

    def test_invalid_json_raises(self, tmp_path: Path):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text("not valid json\n")
        with pytest.raises(orjson.JSONDecodeError):
            read_trace_from_jsonl(bad_file)
