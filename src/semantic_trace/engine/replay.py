"""Mechanical and semantic replay engines for trace files."""

from __future__ import annotations

from pathlib import Path

from semantic_trace.core.schema import (
    IntentInvariant,
    InvariantResult,
    ReplayReport,
)
from semantic_trace.core.serializer import read_trace_from_jsonl
from semantic_trace.engine.invariants import (
    BaseInvariantChecker,
    InvariantViolation,
    get_checker,
)


def mechanical_replay(trace_file: str | Path) -> list[str]:
    """Validate the structural integrity of a trace file.

    Checks that all span UUIDs are valid, parent references point to
    existing spans, and every span's trace_id matches the trace metadata.

    Args:
        trace_file: Path to the JSONL trace file.

    Returns:
        A list of human-readable error strings. Empty if the trace is valid.
    """
    errors: list[str] = []
    trace = read_trace_from_jsonl(trace_file)

    span_ids: set[str] = {str(s.span_id) for s in trace.spans}

    for span in trace.spans:
        if span.parent_id is not None and str(span.parent_id) not in span_ids:
            errors.append(
                f"Span {span.span_id} references missing parent {span.parent_id}"
            )

    for span in trace.spans:
        if span.trace_id != trace.metadata.trace_id:
            errors.append(f"Span {span.span_id} has mismatched trace_id")

    return errors


def semantic_replay(
    trace_file: str | Path,
    checkers: list[BaseInvariantChecker] | None = None,
) -> ReplayReport:
    """Re-run all attached invariants against each span's output data.

    For every span, the invariants declared at capture time are re-evaluated.
    If a custom checker list is provided, checkers whose class name matches
    the invariant type will override the built-in checker.

    Args:
        trace_file: Path to the JSONL trace file.
        checkers: Optional list of custom checker instances to use instead of
            built-in checkers when the class name matches the invariant type.

    Returns:
        A ReplayReport containing all results, violations, and summary stats.
        The report is iterable over violations for backward compatibility.
    """
    trace = read_trace_from_jsonl(trace_file)
    violations: list[InvariantViolation] = []
    results: list[InvariantResult] = []
    total_invariants = 0

    for span in trace.spans:
        for invariant in span.attached_invariants:
            total_invariants += 1
            checker = _resolve_checker(invariant, checkers)
            score = checker.check(span, invariant)

            result = InvariantResult(
                invariant_id=invariant.id,
                span_id=str(span.span_id),
                score=score,
                threshold=invariant.fidelity_threshold,
                passed=score >= invariant.fidelity_threshold,
            )
            results.append(result)

            if not result.passed:
                violations.append(
                    InvariantViolation(
                        span_id=str(span.span_id),
                        invariant_id=invariant.id,
                        expected_score=invariant.fidelity_threshold,
                        actual_score=score,
                    )
                )

    return ReplayReport(
        trace_file=Path(trace_file).resolve(),
        trace_id=trace.metadata.trace_id,
        agent_name=trace.metadata.agent_name,
        total_spans=len(trace.spans),
        total_invariants=total_invariants,
        violations=violations,
        results=results,
    )


def validate_trace(trace_file: str | Path) -> ReplayReport:
    """Run mechanical and semantic validation on a trace file.

    Combines structural validation (mechanical_replay) with semantic
    invariant checks (semantic_replay) into a single comprehensive report.

    Args:
        trace_file: Path to the JSONL trace file.

    Returns:
        A ReplayReport with both structural errors and invariant violations.
    """
    report = semantic_replay(trace_file)
    report.structural_errors = mechanical_replay(trace_file)
    return report


def _resolve_checker(
    invariant: IntentInvariant,
    checkers: list[BaseInvariantChecker] | None,
) -> BaseInvariantChecker:
    """Select the appropriate checker for an invariant.

    If a matching custom checker is found in the provided list, it takes
    precedence over the built-in checker.

    Args:
        invariant: The invariant to find a checker for.
        checkers: Optional list of custom checkers.

    Returns:
        The selected checker instance.
    """
    if checkers is not None:
        custom = _find_custom_checker(checkers, invariant)
        if custom is not None:
            return custom
    return get_checker(invariant)


def _find_custom_checker(
    checkers: list[BaseInvariantChecker],
    invariant: IntentInvariant,
) -> BaseInvariantChecker | None:
    """Find a custom checker whose class name matches the invariant type.

    Args:
        checkers: List of custom checker instances.
        invariant: The invariant whose type to match against.

    Returns:
        The first matching checker, or None if no match is found.
    """
    for checker in checkers:
        if type(checker).__name__ == invariant.invariant_type.value:
            return checker
    return None


__all__ = [
    "mechanical_replay",
    "semantic_replay",
    "validate_trace",
]
