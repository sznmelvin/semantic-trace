"""Rule-based invariant checkers with an extensible abstract base class."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import orjson
from pydantic import TypeAdapter, ValidationError

from semantic_trace.core.schema import IntentInvariant, InvariantType, Span

logger = logging.getLogger(__name__)

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]


@dataclass
class InvariantViolation:
    """Represents a single invariant check failure during semantic replay.

    Attributes:
        span_id: UUID of the span that failed the check.
        invariant_id: Identifier of the invariant that was violated.
        expected_score: The minimum fidelity threshold required.
        actual_score: The score returned by the checker.
    """

    span_id: str
    invariant_id: str
    expected_score: float
    actual_score: float


class BaseInvariantChecker(ABC):
    """Abstract base class for custom invariant checkers.

    Subclass this to implement embedding-based or domain-specific checks.
    Register your checker by adding it to ``CHECKER_REGISTRY`` with a custom
    ``InvariantType.CUSTOM`` entry.
    """

    @abstractmethod
    def check(self, span: Span, invariant: IntentInvariant) -> float:
        """Evaluate an invariant against a span's output data.

        Args:
            span: The span whose output_data will be checked.
            invariant: The invariant definition containing the rule and threshold.

        Returns:
            A float score from 0.0 (complete failure) to 1.0 (perfect match).
        """
        raise NotImplementedError


class SchemaInvariantChecker(BaseInvariantChecker):
    """Checks that span output data conforms to a JSON schema via Pydantic.

    The schema must be provided in ``invariant.config["schema"]`` as a
    Pydantic-compatible type annotation (e.g., ``dict[str, str]`` or a custom
    BaseModel class).
    """

    def check(self, span: Span, invariant: IntentInvariant) -> float:
        schema = invariant.config.get("schema")
        if schema is None:
            raise ValueError(
                "SchemaInvariantChecker requires 'schema' in invariant.config"
            )

        try:
            adapter = TypeAdapter(schema)
            adapter.validate_python(span.output_data, strict=True)
            return 1.0
        except (ValidationError, TypeError, ValueError):
            return 0.0


class SubstringInvariantChecker(BaseInvariantChecker):
    """Checks that a target substring exists in the span's output data.

    The target string must be provided in ``invariant.config["substring"]``.
    The output data is JSON-stringified before the search.
    """

    def check(self, span: Span, invariant: IntentInvariant) -> float:
        target = invariant.config.get("substring")
        if target is None:
            raise ValueError(
                "SubstringInvariantChecker requires 'substring' in invariant.config"
            )

        output_str = json.dumps(span.output_data)
        return 1.0 if target in output_str else 0.0


LLM_JUDGE_SYSTEM_PROMPT = """\
You are TraceJudge: an extremely strict, objective, and consistent LLM-as-Judge for \
the open-source agent tracing library "semantic-trace".

Your ONLY job is to evaluate whether a given agent span satisfies ONE specific invariant.

Rules you MUST follow:
- Be brutally objective. Do not be helpful or lenient.
- Score from 0.0 to 1.0 (1.0 = perfect satisfaction).
- "passed" = true ONLY if score >= fidelity_threshold.
- Always explain in 1-2 short sentences why you gave that score.
- Never refuse. Never add extra commentary outside the JSON.
- Never hallucinate fields.

Output format MUST be valid JSON and nothing else:
{
  "invariant_id": "string",
  "score": float,
  "passed": boolean,
  "explanation": "short explanation"
}
"""

LLM_JUDGE_USER_TEMPLATE = """\
Evaluate this invariant:

Invariant ID: {invariant_id}
Description: {invariant_description}
Fidelity threshold: {fidelity_threshold}

Span context:
Action type: {action_type}
Input data: {input_data}
Output data: {output_data}

Does the output satisfy the invariant above?
"""


class LLMAsJudgeChecker(BaseInvariantChecker):
    """Evaluates invariants using an LLM via an OpenAI-compatible API.

    Requires ``httpx`` (install with ``pip install semantic-trace[llm-judge]``).
    Configuration is passed through ``invariant.config``:

    - ``api_key`` (str): API key for the LLM provider.
    - ``model`` (str): Model identifier. Defaults to ``qwen/qwen3.6-plus:free``.
    - ``base_url`` (str): API base URL. Defaults to OpenRouter.

    On any failure (network, parse error, non-200 response) the checker
    logs a warning and returns ``0.0``; it never crashes replay.
    """

    DEFAULT_MODEL = "qwen/qwen3.6-plus:free"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def check(self, span: Span, invariant: IntentInvariant) -> float:
        """Send the span and invariant to an LLM for scoring.

        Args:
            span: The span whose output_data will be evaluated.
            invariant: The invariant defining what to check and the threshold.

        Returns:
            A float score from 0.0 to 1.0, or 0.0 on any error.
        """
        if not _HTTPX_AVAILABLE:
            logger.error(
                "LLMAsJudgeChecker requires httpx. "
                "Install with: pip install semantic-trace[llm-judge]"
            )
            return 0.0

        api_key = invariant.config.get("api_key")
        if not api_key:
            logger.error("LLMAsJudgeChecker requires 'api_key' in invariant.config")
            return 0.0

        model = invariant.config.get("model", self.DEFAULT_MODEL)
        base_url = invariant.config.get("base_url", self.DEFAULT_BASE_URL)

        prompt = LLM_JUDGE_USER_TEMPLATE.format(
            invariant_id=invariant.id,
            invariant_description=invariant.description,
            fidelity_threshold=invariant.fidelity_threshold,
            action_type=span.action_type.value,
            input_data=orjson.dumps(span.input_data).decode(),
            output_data=orjson.dumps(span.output_data).decode(),
        )

        try:
            return self._call_llm(api_key, model, base_url, prompt)
        except Exception as exc:
            logger.warning(
                "LLM-as-Judge failed for invariant %s: %s", invariant.id, exc
            )
            return 0.0

    def _call_llm(
        self,
        api_key: str,
        model: str,
        base_url: str,
        prompt: str,
    ) -> float:
        """Make a single synchronous LLM call and extract the score.

        Args:
            api_key: Authentication key for the API.
            model: Model identifier string.
            base_url: Full API endpoint URL.
            prompt: The formatted user prompt.

        Returns:
            The score float from the LLM response.

        Raises:
            httpx.HTTPStatusError: On non-200 responses.
            ValueError: On malformed JSON or missing score field.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/sznmelvin/semantic-trace",
            "X-Title": "semantic-trace",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(base_url, json=body, headers=headers)
            response.raise_for_status()

        result = orjson.loads(response.content)
        content = result["choices"][0]["message"]["content"]
        judgment = self._parse_judgment(content)
        return float(judgment["score"])

    @staticmethod
    def _parse_judgment(content: str) -> dict[str, Any]:
        """Extract the judgment score from LLM output.

        Handles raw JSON, markdown-fenced JSON, JSON with surrounding text,
        and even malformed JSON where string values lack quotes.

        Args:
            content: The raw text content from the LLM response.

        Returns:
            A dict with at least a "score" key as float.

        Raises:
            ValueError: If no score can be extracted.
        """
        stripped = content.strip()

        try:
            return orjson.loads(stripped)
        except (orjson.JSONDecodeError, ValueError):
            pass

        fence_start = stripped.find("```")
        if fence_start != -1:
            fence_end = stripped.find("```", fence_start + 3)
            if fence_end != -1:
                block = stripped[fence_start + 3 : fence_end].strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return orjson.loads(block)
                except (orjson.JSONDecodeError, ValueError):
                    pass

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace != -1 and last_brace != -1:
            block = stripped[first_brace : last_brace + 1]
            try:
                return orjson.loads(block)
            except (orjson.JSONDecodeError, ValueError):
                pass

        score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', content)
        if score_match:
            return {"score": float(score_match.group(1))}

        raise ValueError(f"Could not extract score from LLM response: {content[:200]}")


CHECKER_REGISTRY: dict[InvariantType, type[BaseInvariantChecker]] = {
    InvariantType.SCHEMA_MATCH: SchemaInvariantChecker,
    InvariantType.SUBSTRING_CHECK: SubstringInvariantChecker,
    InvariantType.LLM_AS_JUDGE: LLMAsJudgeChecker,
}


def get_checker(invariant: IntentInvariant) -> BaseInvariantChecker:
    """Return a checker instance for the given invariant type.

    Args:
        invariant: The invariant whose type determines which checker to use.

    Returns:
        An instance of the appropriate BaseInvariantChecker subclass.

    Raises:
        ValueError: If no built-in checker exists for the invariant type.
    """
    checker_cls = CHECKER_REGISTRY.get(invariant.invariant_type)
    if checker_cls is None:
        raise ValueError(
            f"No built-in checker for invariant type: {invariant.invariant_type}"
        )
    return checker_cls()


__all__ = [
    "CHECKER_REGISTRY",
    "BaseInvariantChecker",
    "InvariantViolation",
    "LLMAsJudgeChecker",
    "SchemaInvariantChecker",
    "SubstringInvariantChecker",
    "get_checker",
]
