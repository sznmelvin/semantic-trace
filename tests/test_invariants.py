"""Tests for invariant checkers."""

import pytest

from agent_trace import ActionType, IntentInvariant, InvariantType, Span
from agent_trace.engine.invariants import (
    BaseInvariantChecker,
    SchemaInvariantChecker,
    SubstringInvariantChecker,
    get_checker,
)


def make_span(output_data: dict) -> Span:
    return Span(
        trace_id="00000000-0000-0000-0000-000000000001",
        action_type=ActionType.LLM_CALL,
        output_data=output_data,
    )


class TestSubstringInvariantChecker:
    def test_substring_found(self):
        checker = SubstringInvariantChecker()
        span = make_span({"summary": "hello world"})
        inv = IntentInvariant(
            id="has-summary",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"summary"'},
        )
        assert checker.check(span, inv) == 1.0

    def test_substring_not_found(self):
        checker = SubstringInvariantChecker()
        span = make_span({"response": "hello"})
        inv = IntentInvariant(
            id="has-summary",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
            config={"substring": '"summary"'},
        )
        assert checker.check(span, inv) == 0.0

    def test_missing_config_raises(self):
        checker = SubstringInvariantChecker()
        span = make_span({})
        inv = IntentInvariant(
            id="test",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
        )
        with pytest.raises(ValueError, match="requires 'substring'"):
            checker.check(span, inv)


class TestSchemaInvariantChecker:
    def test_valid_schema(self):
        checker = SchemaInvariantChecker()
        span = make_span({"name": "test", "value": 42})
        inv = IntentInvariant(
            id="valid-schema",
            description="test",
            invariant_type=InvariantType.SCHEMA_MATCH,
            config={"schema": dict[str, str | int]},
        )
        assert checker.check(span, inv) == 1.0

    def test_invalid_schema(self):
        checker = SchemaInvariantChecker()
        span = make_span({"name": 123})
        inv = IntentInvariant(
            id="valid-schema",
            description="test",
            invariant_type=InvariantType.SCHEMA_MATCH,
            config={"schema": dict[str, str]},
        )
        assert checker.check(span, inv) == 0.0

    def test_missing_config_raises(self):
        checker = SchemaInvariantChecker()
        span = make_span({})
        inv = IntentInvariant(
            id="test",
            description="test",
            invariant_type=InvariantType.SCHEMA_MATCH,
        )
        with pytest.raises(ValueError, match="requires 'schema'"):
            checker.check(span, inv)


class TestCustomChecker:
    def test_custom_checker_subclass(self):
        class AlwaysPassChecker(BaseInvariantChecker):
            def check(self, span: Span, invariant: IntentInvariant) -> float:
                return 1.0

        checker = AlwaysPassChecker()
        span = make_span({})
        inv = IntentInvariant(
            id="test",
            description="test",
            invariant_type=InvariantType.CUSTOM,
        )
        assert checker.check(span, inv) == 1.0


class TestGetChecker:
    def test_returns_correct_checker(self):
        inv = IntentInvariant(
            id="test",
            description="test",
            invariant_type=InvariantType.SUBSTRING_CHECK,
        )
        checker = get_checker(inv)
        assert isinstance(checker, SubstringInvariantChecker)

    def test_unknown_type_raises(self):
        inv = IntentInvariant(
            id="test",
            description="test",
            invariant_type=InvariantType.CUSTOM,
        )
        with pytest.raises(ValueError, match="No built-in checker"):
            get_checker(inv)
