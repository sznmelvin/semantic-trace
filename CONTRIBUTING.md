# Contributing to agent-trace

Thank you for your interest in contributing! agent-trace is built on the belief that AI agent observability should be simple, composable, and open. Every contribution, whether a typo fix, a new checker, or a full integration, is welcome.

## How to Contribute

### Reporting Bugs

- Check the [issue tracker](https://github.com/sznmelvin/agent-trace/issues) to see if it's already reported.
- Open a new issue with a clear title, description, and minimal reproduction if possible.
- Include your Python version and `agent-trace` version.

### Suggesting Features

- Open an issue with the label `enhancement`.
- Describe the problem you're trying to solve, not just the solution you have in mind.
- Keep scope focused; agent-trace values minimalism.

### Pull Requests

1. **Fork** the repository and create your branch from `main`.
2. **Install dev dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
3. **Make your changes** with clear, minimal code and docstrings.
4. **Add tests** for new functionality. We use pytest.
5. **Run the checks**:
   ```bash
   ruff check src/agent_trace/ tests/ examples/
   ruff format src/agent_trace/ tests/ examples/
   pytest tests/ -v
   ```
6. **Open a PR** with a clear description of what and why.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.
- Line length: 88 characters.
- Type hints on all public functions and methods.
- Docstrings on all public classes and functions.

## Design Principles

1. **Minimalism first**: If it can be one line, don't make it three.
2. **No hidden magic**: What you see is what you get. No metaclass sorcery.
3. **Composable**: Small pieces that work well together.
4. **Backward compatible**: Don't break existing code without a major version bump.
5. **Zero unnecessary dependencies**: Core is `pydantic` + `orjson`. Everything else is optional.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/agent-trace.git
cd agent-trace

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev deps
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/agent_trace/ tests/ examples/
```

## Adding a New Invariant Checker

If you want to contribute a new built-in checker:

1. Add the checker class to `src/agent_trace/engine/invariants.py`.
2. Register it in `CHECKER_REGISTRY` with a new `InvariantType` (or reuse an existing one).
3. Add the type to the `InvariantType` enum in `src/agent_trace/core/schema.py`.
4. Write tests in `tests/test_invariants.py`.
5. Update the README invariant types table.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
