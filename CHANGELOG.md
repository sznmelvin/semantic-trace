# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-07

Initial release.

### Added
- Core Pydantic models: `TraceModel`, `Span`, `IntentInvariant`, `TraceMetadata`
- `Trace` context manager for simplified capture workflow with auto-serialization
- `ReplayReport` class with `.summary()`, `.print_violations()`, `.is_clean`, `.pass_rate`
- `InvariantResult` dataclass for per-invariant check details
- Enum types: `InvariantType`, `ActionType`
- JSONL serialization with orjson and file locking
- Built-in invariant checkers: `SchemaInvariantChecker`, `SubstringInvariantChecker`, `LLMAsJudgeChecker`
- `BaseInvariantChecker` ABC for custom checkers
- Mechanical replay (structural validation)
- Semantic replay (invariant checking)
- `validate_trace()` for combined mechanical + semantic validation
- LangGraph callback handler integration
- CLI with `info`, `validate`, `replay`, `spans` commands and `--json` output
- Optional extras: `langgraph`, `llm-judge`, `dev`
- Comprehensive test suite (71 tests)
- GitHub Actions CI workflow

[Unreleased]: https://github.com/sznmelvin/semantic-trace/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/sznmelvin/semantic-trace/releases/tag/v0.1.0
