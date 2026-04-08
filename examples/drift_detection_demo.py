"""Drift detection demo: catching agent regressions with semantic replay.

This demo shows the core value proposition of semantic-trace:
1. Capture a "golden" trace when your agent works correctly
2. Simulate a model/prompt change that degrades output quality
3. Replay the golden trace against the new output to catch the regression

Run: python examples/drift_detection_demo.py
"""

from semantic_trace import (
    ActionType,
    IntentInvariant,
    InvariantType,
    Span,
    Trace,
    semantic_replay,
)


def main() -> None:
    # 1. Define invariants that codify "what good looks like"
    invariants = [
        IntentInvariant(
            id="has-summary",
            description="Output must contain a summary field",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"summary"'},
            fidelity_threshold=1.0,
        ),
        IntentInvariant(
            id="has-citations",
            description="Output must contain citations field",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"citations"'},
            fidelity_threshold=1.0,
        ),
        IntentInvariant(
            id="structured-format",
            description="Output must contain metadata field",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"metadata"'},
            fidelity_threshold=1.0,
        ),
    ]

    # 2. Capture a golden trace (agent working correctly)
    print("=== Step 1: Capturing golden trace ===\n")

    with Trace(
        name="research-assistant",
        invariants=invariants,
        output_file="traces/golden.jsonl",
    ) as trace:
        span = Span(
            trace_id=trace.trace_id,
            action_type=ActionType.LLM_CALL,
            input_data={"prompt": "Research transformers and return structured output"},
            output_data={
                "summary": "Transformers are the dominant architecture in NLP...",
                "citations": ["Vaswani et al. 2017", "Dosovitskiy et al. 2020"],
                "metadata": {"model": "gpt-4", "tokens": 1200},
            },
            duration_ms=450.0,
        )
        trace.add_span(span)
        print(f"Golden trace captured: {trace.trace_id}")

    # 3. Replay golden trace: should pass
    print("\n=== Step 2: Replay golden trace (should pass) ===\n")

    report = semantic_replay("traces/golden.jsonl")
    print(report.summary())
    report.print_violations()

    # 4. Simulate a model regression (agent output degraded)
    print("\n=== Step 3: Simulating model regression ===\n")

    with Trace(
        name="research-assistant-v2",
        invariants=invariants,
        output_file="traces/regressed.jsonl",
    ) as trace:
        # Simulated: model changed, output format broke
        span = Span(
            trace_id=trace.trace_id,
            action_type=ActionType.LLM_CALL,
            input_data={"prompt": "Research transformers and return structured output"},
            output_data={
                "text": "Transformers are great. They use attention. Very cool stuff.",
            },
            duration_ms=200.0,
        )
        trace.add_span(span)
        print("Regressed trace captured (missing summary, citations, metadata)")

    # 5. Replay regressed trace: should catch violations
    print("\n=== Step 4: Replay regressed trace (should fail) ===\n")

    report = semantic_replay("traces/regressed.jsonl")
    print(report.summary())
    print()
    report.print_violations()

    # 6. Summary
    print("\n=== Result ===")
    print(f"Violations detected: {len(report.violations)}/{report.total_invariants}")
    print(f"Pass rate: {report.pass_rate:.0%}")
    print(
        "\nThis is the power of semantic-trace: you caught the regression"
        " without manually inspecting output."
    )


if __name__ == "__main__":
    main()
