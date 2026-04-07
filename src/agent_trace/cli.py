"""CLI entry point for inspecting and validating trace files.

Usage:
    trace <command> <trace_file> [options]

Commands:
    info       Show trace metadata summary
    validate   Run mechanical (structural) validation
    replay     Run full semantic replay with invariant checks
    spans      List all spans with their durations and invariant results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent_trace.core.serializer import read_trace_from_jsonl
from agent_trace.engine.replay import mechanical_replay, validate_trace


def cmd_info(args: argparse.Namespace) -> None:
    """Print trace metadata summary to stdout."""
    trace = read_trace_from_jsonl(args.trace_file)
    if args.json:
        print(json.dumps(trace.metadata.model_dump(mode="json"), indent=2))
        return

    print(f"Trace ID:    {trace.metadata.trace_id}")
    print(f"Session ID:  {trace.metadata.session_id}")
    print(f"Agent:       {trace.metadata.agent_name}")
    print(f"Start:       {trace.metadata.start_time.isoformat()}")
    end = trace.metadata.end_time.isoformat() if trace.metadata.end_time else "N/A"
    print(f"End:         {end}")
    print(f"Spans:       {len(trace.spans)}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Run mechanical validation and exit non-zero on errors."""
    errors = mechanical_replay(args.trace_file)
    if args.json:
        print(json.dumps({"valid": len(errors) == 0, "errors": errors}, indent=2))
        if errors:
            sys.exit(1)
        return

    if errors:
        print("Structural errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print("Structural validation passed.")


def cmd_replay(args: argparse.Namespace) -> None:
    """Run full semantic replay and exit non-zero on violations."""
    report = validate_trace(args.trace_file)

    if args.json:
        output = {
            "clean": report.is_clean,
            "trace_id": str(report.trace_id),
            "agent_name": report.agent_name,
            "total_spans": report.total_spans,
            "total_invariants": report.total_invariants,
            "pass_rate": report.pass_rate,
            "structural_errors": report.structural_errors,
            "violations": [
                {
                    "span_id": v.span_id,
                    "invariant_id": v.invariant_id,
                    "expected_score": v.expected_score,
                    "actual_score": v.actual_score,
                }
                for v in report.violations
            ],
        }
        print(json.dumps(output, indent=2))
        if not report.is_clean:
            sys.exit(1)
        return

    print(report.summary())
    if not report.is_clean:
        report.print_violations()
        sys.exit(1)


def cmd_spans(args: argparse.Namespace) -> None:
    """List all spans with action type, UUID, duration, and invariant results."""
    trace = read_trace_from_jsonl(args.trace_file)

    if args.json:
        spans = []
        for span in trace.spans:
            span_data = span.model_dump(mode="json")
            spans.append(span_data)
        print(json.dumps(spans, indent=2))
        return

    for span in trace.spans:
        if span.duration_ms is not None:
            print(f"[{span.action_type}] {span.span_id} ({span.duration_ms:.0f}ms)")
        else:
            print(f"[{span.action_type}] {span.span_id}")
        if span.invariant_results:
            for inv_id, score in span.invariant_results.items():
                status = "PASS" if score >= 1.0 else "FAIL"
                print(f"    {inv_id}: {score:.2f} [{status}]")


def main() -> None:
    """CLI entry point. Parses arguments and dispatches to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="trace",
        description="Semantic tracing CLI for AI agents. Inspect, validate, and replay trace files.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="agent-trace 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    for name, help_text, handler in [
        ("info", "Show trace metadata summary", cmd_info),
        ("validate", "Run structural validation", cmd_validate),
        ("replay", "Run full semantic replay", cmd_replay),
        ("spans", "List all spans", cmd_spans),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("trace_file", type=str, help="Path to the JSONL trace file")
        sub.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format for machine consumption",
        )
        sub.set_defaults(handler=handler)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"Error: file not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    args.handler(args)


if __name__ == "__main__":
    main()
