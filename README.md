# agent-trace

[![PyPI - Version](https://img.shields.io/pypi/v/semantic-trace.svg?logo=pypi&labelColor=555555)](https://pypi.org/project/semantic-trace/)
[![Python - Versions](https://img.shields.io/pypi/pyversions/semantic-trace.svg?logo=python&labelColor=555555)](https://pypi.org/project/semantic-trace/)
[![License - MIT](https://img.shields.io/pypi/l/agent-trace.svg?logo=github&labelColor=555555)](https://github.com/sznmelvin/agent-trace/blob/main/LICENSE)
[![CI](https://github.com/sznmelvin/agent-trace/actions/workflows/ci.yml/badge.svg)](https://github.com/sznmelvin/agent-trace/actions/workflows/ci.yml)

**Semantic tracing primitive for AI agents.**

Intent-anchored execution. Deterministic replay. Runtime drift detection.

---

## Philosophy

Existing observability tools log **what** happened: every keystroke, token, and HTTP request. But they don't capture **intent**.

agent-trace flips this. Instead of dumping raw logs, you attach *invariants* to your agent's actions:

> "This LLM call should return a JSON object with `action` and `params` keys."
> "This tool output must contain the substring `success`."

When the agent runs, those invariants travel with the trace. Later, you replay the trace and check whether every invariant still holds. If a model upgrade, prompt change, or tool regression breaks an invariant, you catch it immediately.

**agent-trace is a primitive, not a platform.** No web servers. No databases. No UI frameworks. Just strictly-typed Python data structures and JSONL files you can version-control, diff, and grep.

## Installation

```bash
pip install semantic-trace
```

With optional integrations:

```bash
pip install semantic-trace[langgraph]    # LangGraph callback handler
pip install semantic-trace[llm-judge]    # LLM-as-Judge invariant checker
pip install semantic-trace[dev]          # pytest + ruff for contributors
```

## Quick Start

```python
from agent_trace import Trace, IntentInvariant, InvariantType, semantic_replay

# 1. Define invariants: what your agent's output MUST satisfy
invariants = [
    IntentInvariant(
        id="valid-json",
        description="Output must be valid JSON with 'summary' key",
        invariant_type=InvariantType.SUBSTRING_CHECK,
        config={"substring": '"summary"'},
        fidelity_threshold=1.0,
    ),
    IntentInvariant(
        id="no-hallucination",
        description="Must not invent fake citations",
        invariant_type=InvariantType.LLM_AS_JUDGE,
        config={"api_key": "your-key", "model": "qwen/qwen3.6-plus:free"},
        fidelity_threshold=0.85,
    ),
]

# 2. Capture trace: invariants auto-attach to every span
with Trace(
    name="research-assistant",
    invariants=invariants,
    output_file="traces/run.jsonl",
) as trace:
    # Your agent code here (LangGraph, CrewAI, custom, etc.)
    # Spans are captured automatically via integrations
    # or manually:
    from agent_trace import Span, ActionType

    trace.add_span(Span(
        trace_id=trace.trace_id,
        action_type=ActionType.LLM_CALL,
        input_data={"prompt": "Summarize this document..."},
        output_data={"summary": "The document discusses..."},
        duration_ms=342.0,
    ))

# 3. Later: replay and check all invariants
report = semantic_replay("traces/run.jsonl")
print(report.summary())
report.print_violations()
```

## Why agent-trace?

| Problem | agent-trace solution |
|---------|---------------------|
| Agent behavior changes silently after a model upgrade | Replay old traces with invariants to catch regressions |
| No way to codify "what good looks like" for agent output | Attach invariants as executable specifications |
| Observability platforms are expensive and complex | JSONL files you own, version-control, and grep |
| Testing agents is hard and non-deterministic | Semantic replay checks intent, not exact output |

## Invariant Types

| Type | What it does | Config |
|------|-------------|--------|
| `SUBSTRING_CHECK` | Checks for a target substring in JSON output | `{"substring": "..."}` |
| `SCHEMA_MATCH` | Validates output against a Pydantic type | `{"schema": dict[str, str]}` |
| `LLM_AS_JUDGE` | Uses an LLM to semantically evaluate | `{"api_key": "...", "model": "..."}` |
| `CUSTOM` | Your own checker via `BaseInvariantChecker` | Any |

## CLI

```bash
# Show trace metadata
trace info traces/run.jsonl

# Structural validation
trace validate traces/run.jsonl

# Full semantic replay (mechanical + invariant checks)
trace replay traces/run.jsonl

# List all spans with durations
trace spans traces/run.jsonl

# Machine-readable JSON output
trace replay traces/run.jsonl --json
```

## Architecture

```
agent-trace/
├── src/agent_trace/
│   ├── __init__.py              # Public API re-exports
│   ├── cli.py                   # `trace` CLI entry point
│   ├── core/
│   │   ├── schema.py            # Pydantic models + Trace context manager
│   │   └── serializer.py        # JSONL read/write with orjson + file locking
│   ├── engine/
│   │   ├── invariants.py        # ABC + built-in checkers
│   │   └── replay.py            # Mechanical + semantic replay
│   └── integrations/
│       └── langgraph.py         # Lazy-loaded LangGraph callback handler
├── examples/
│   ├── minimal_demo.py          # Full workflow demo
│   └── drift_detection_demo.py  # Catching regressions with replay
├── tests/                       # Comprehensive test suite
├── pyproject.toml
└── README.md
```

## Examples

- [**minimal_demo.py**](examples/minimal_demo.py): Complete workflow: define invariants, capture trace, replay
- [**drift_detection_demo.py**](examples/drift_detection_demo.py): Simulate a model regression and catch it with semantic replay

## Extending

### Custom Checker

Write your own checker by subclassing `BaseInvariantChecker`:

```python
from agent_trace import BaseInvariantChecker, Span, IntentInvariant

class EmbeddingSimilarityChecker(BaseInvariantChecker):
    def check(self, span: Span, invariant: IntentInvariant) -> float:
        # Your embedding-based logic here
        expected = invariant.config["expected_embedding"]
        actual = get_embedding(span.output_data["text"])
        return cosine_similarity(expected, actual)
```

### LLM-as-Judge

Use an LLM to semantically evaluate whether a span's output satisfies an invariant.
Requires `pip install agent-trace[llm-judge]`.

```python
from agent_trace import IntentInvariant, InvariantType

invariant = IntentInvariant(
    id="quality-check",
    description="The response should be helpful and well-structured",
    invariant_type=InvariantType.LLM_AS_JUDGE,
    config={
        "api_key": "sk-or-your-key",
        "model": "qwen/qwen3.6-plus:free",
    },
    fidelity_threshold=0.7,
)
```

The judge sends the span context to an LLM (default: OpenRouter) and parses a
structured JSON score. On any failure it returns `0.0` and logs a warning;
it never crashes your replay pipeline.

### LangGraph Integration

```python
from agent_trace.integrations.langgraph import TraceCallbackHandler

handler = TraceCallbackHandler(
    trace_file="traces/run.jsonl",
    session_id="session-1",
    agent_name="my-agent",
    default_invariants=[invariant],
)

graph = create_react_agent(..., callbacks=[handler])
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and how to submit PRs.

## License

MIT
