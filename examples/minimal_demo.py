"""Minimal demo showing the full semantic-trace workflow.

Demonstrates:
1. Defining invariants (intent assertions)
2. Capturing a trace with the context manager
3. Running semantic replay to check invariants
4. Using the CLI to inspect traces

Run: python examples/minimal_demo.py
"""

from semantic_trace import (
    ActionType,
    IntentInvariant,
    InvariantType,
    Span,
    Trace,
    mechanical_replay,
    semantic_replay,
)


def main() -> None:
    invariants = [
        IntentInvariant(
            id="has-summary",
            description="Output must contain a 'summary' key",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"summary"'},
            fidelity_threshold=1.0,
        ),
        IntentInvariant(
            id="valid-response",
            description="Output must contain a 'response' field",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"response"'},
            fidelity_threshold=1.0,
        ),
    ]

    with Trace(
        name="research-assistant",
        invariants=invariants,
        output_file="traces/demo.jsonl",
    ) as trace:
        # Simulate agent execution
        # In real code: call your LangGraph / CrewAI / custom agent here

        span = Span(
            trace_id=trace.trace_id,
            action_type=ActionType.LLM_CALL,
            input_data={"prompt": "Summarize the latest AI research on transformers"},
            output_data={
                "summary": "Transformer architectures continue to scale effectively...",
                "response": "Here's a summary of recent transformer research...",
            },
            duration_ms=342.0,
        )
        trace.add_span(span)

        print(f"Trace ID: {trace.trace_id}")
        print(f"Spans captured: {len(trace.spans)}")

    print("\nTrace saved to traces/demo.jsonl")

    # Mechanical validation
    errors = mechanical_replay("traces/demo.jsonl")
    if errors:
        print("\nStructural errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("Structural validation passed.")

    # Semantic replay
    report = semantic_replay("traces/demo.jsonl")
    print(f"\n{report.summary()}")
    report.print_violations()


if __name__ == "__main__":
    main()
