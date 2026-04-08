"""JSONL serialization for trace files with file locking and path validation."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import orjson

from semantic_trace.core.schema import Span, TraceMetadata, TraceModel

logger = logging.getLogger(__name__)

_FILE_LOCK_AVAILABLE = sys.platform != "win32"
if _FILE_LOCK_AVAILABLE:
    import fcntl


def _serialize(obj: Any) -> bytes:
    """Serialize a Python object to JSON bytes using orjson.

    Args:
        obj: The object to serialize.

    Returns:
        JSON-encoded bytes with naive UTC and UUID serialization options.
    """
    return orjson.dumps(obj, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_UUID)


def _validate_path(trace_file: str | Path) -> Path:
    """Resolve and validate a trace file path.

    Logs a warning if the resolved path is outside the current working directory.
    This is informational only -- users may legitimately write to /tmp or any
    absolute path.

    Args:
        trace_file: The path to validate.

    Returns:
        The resolved absolute Path object.
    """
    path = Path(trace_file).resolve()
    cwd = Path.cwd().resolve()
    if not path.is_relative_to(cwd):
        logger.warning(
            "Trace file path %s is outside the current working directory %s",
            path,
            cwd,
        )
    return path


def _lock_file(f, exclusive: bool = True) -> None:
    """Acquire a file lock if the platform supports it.

    Args:
        f: The file object to lock.
        exclusive: True for exclusive (write) lock, False for shared (read) lock.
    """
    if _FILE_LOCK_AVAILABLE:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(f.fileno(), lock_type)


def _unlock_file(f) -> None:
    """Release a file lock if the platform supports it.

    Args:
        f: The file object to unlock.
    """
    if _FILE_LOCK_AVAILABLE:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_span_to_jsonl(trace_file: str | Path, span: Span) -> None:
    """Append a single Span to a JSONL trace file.

    Creates parent directories if they do not exist. Uses exclusive file
    locking to ensure safe concurrent writes.

    Args:
        trace_file: Path to the JSONL file.
        span: The Span object to write.
    """
    path = _validate_path(trace_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _serialize(span.model_dump(mode="json"))
    with open(path, "ab") as f:
        _lock_file(f, exclusive=True)
        try:
            f.write(data + b"\n")
        finally:
            _unlock_file(f)


def write_metadata_to_jsonl(trace_file: str | Path, trace: TraceModel) -> None:
    """Write trace metadata as a line in a JSONL trace file.

    When called on a new file, writes the header. When called on an existing
    file, appends a metadata line (the reader uses the last metadata line found).

    Creates parent directories if they do not exist. Uses exclusive file
    locking to ensure safe concurrent writes.

    Args:
        trace_file: Path to the JSONL file.
        trace: The TraceModel whose metadata will be written.
    """
    path = _validate_path(trace_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"__metadata__": trace.metadata.model_dump(mode="json")}
    data = _serialize(header)
    with open(path, "ab") as f:
        _lock_file(f, exclusive=True)
        try:
            f.write(data + b"\n")
        finally:
            _unlock_file(f)


def read_trace_from_jsonl(trace_file: str | Path) -> TraceModel:
    """Read a complete Trace from a JSONL file.

    The first ``__metadata__`` line provides the trace header. All subsequent
    lines are parsed as Span objects. If multiple metadata lines exist, the
    last one wins (allowing the context manager to append end_time on exit).

    Args:
        trace_file: Path to the JSONL file.

    Returns:
        A fully reconstructed TraceModel.

    Raises:
        FileNotFoundError: If the trace file does not exist.
        orjson.JSONDecodeError: If a line contains invalid JSON.
        ValueError: If no metadata line is found in the file.
    """
    path = _validate_path(trace_file)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    metadata: TraceMetadata | None = None
    spans: list[Span] = []

    with open(path, "rb") as f:
        _lock_file(f, exclusive=False)
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = orjson.loads(line)
                if "__metadata__" in obj:
                    metadata = TraceMetadata.model_validate(obj["__metadata__"])
                else:
                    spans.append(Span.model_validate(obj))
        finally:
            _unlock_file(f)

    if metadata is None:
        raise ValueError(f"No metadata found in trace file: {trace_file}")

    return TraceModel(metadata=metadata, spans=spans)


__all__ = [
    "read_trace_from_jsonl",
    "write_metadata_to_jsonl",
    "write_span_to_jsonl",
]
