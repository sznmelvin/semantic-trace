"""Tests for the CLI entry point."""

import subprocess
import sys
from pathlib import Path

import pytest

from agent_trace import (
    ActionType,
    IntentInvariant,
    InvariantType,
    Span,
    TraceMetadata,
    TraceModel,
)
from agent_trace.core.serializer import write_metadata_to_jsonl, write_span_to_jsonl


def make_trace_file(path: Path) -> Path:
    """Create a valid trace file for CLI testing."""
    meta = TraceMetadata(session_id="cli-test", agent_name="cli-agent")
    trace = TraceModel(metadata=meta)
    write_metadata_to_jsonl(path, trace)

    inv = IntentInvariant(
        id="has-response",
        description="must have response",
        invariant_type=InvariantType.SUBSTRING_CHECK,
        config={"substring": '"response"'},
    )
    span = Span(
        trace_id=meta.trace_id,
        action_type=ActionType.LLM_CALL,
        input_data={"prompt": "hello"},
        output_data={"response": "hi"},
        duration_ms=100.0,
        attached_invariants=[inv],
    )
    write_span_to_jsonl(path, span)
    return path


@pytest.fixture
def valid_trace(tmp_path: Path) -> Path:
    return make_trace_file(tmp_path / "trace.jsonl")


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agent_trace.cli", *args],
        capture_output=True,
        text=True,
    )


class TestCLIInfo:
    def test_info_command(self, valid_trace: Path):
        result = run_cli(["info", str(valid_trace)])
        assert result.returncode == 0
        assert "cli-agent" in result.stdout
        assert "cli-test" in result.stdout

    def test_info_json(self, valid_trace: Path):
        result = run_cli(["info", str(valid_trace), "--json"])
        assert result.returncode == 0
        import json

        data = json.loads(result.stdout)
        assert data["agent_name"] == "cli-agent"


class TestCLIValidate:
    def test_validate_pass(self, valid_trace: Path):
        result = run_cli(["validate", str(valid_trace)])
        assert result.returncode == 0
        assert "passed" in result.stdout

    def test_validate_json(self, valid_trace: Path):
        result = run_cli(["validate", str(valid_trace), "--json"])
        assert result.returncode == 0
        import json

        data = json.loads(result.stdout)
        assert data["valid"] is True


class TestCLIReplay:
    def test_replay_pass(self, valid_trace: Path):
        result = run_cli(["replay", str(valid_trace)])
        assert result.returncode == 0
        assert "ALL CLEAR" in result.stdout

    def test_replay_json(self, valid_trace: Path):
        result = run_cli(["replay", str(valid_trace), "--json"])
        assert result.returncode == 0
        import json

        data = json.loads(result.stdout)
        assert data["clean"] is True


class TestCLISpans:
    def test_spans_command(self, valid_trace: Path):
        result = run_cli(["spans", str(valid_trace)])
        assert result.returncode == 0
        assert "LLM_CALL" in result.stdout

    def test_spans_json(self, valid_trace: Path):
        result = run_cli(["spans", str(valid_trace), "--json"])
        assert result.returncode == 0
        import json

        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 1


class TestCLIErrors:
    def test_no_command_shows_help(self):
        result = run_cli([])
        assert result.returncode != 0

    def test_file_not_found(self):
        result = run_cli(["info", "/nonexistent.jsonl"])
        assert result.returncode != 0
        assert "not found" in result.stderr

    def test_unknown_command(self):
        result = run_cli(["bogus", "test.jsonl"])
        assert result.returncode != 0
