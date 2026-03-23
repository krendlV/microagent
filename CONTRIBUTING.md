# Contributing to microagent

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository and clone your fork.
2. Install [uv](https://docs.astral.sh/uv/).
3. Set up the development environment:

```bash
uv sync --dev
pre-commit install
```

## Development Workflow

- Create a feature branch from `main`.
- Make your changes with tests.
- Run the full check suite:

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run pytest
```

- Open a pull request against `main`.

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (line length: 99).
Run `uv run ruff format .` before committing.

## Reporting Issues

Use the GitHub issue templates for bug reports and feature requests.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
