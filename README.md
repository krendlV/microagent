# microagent

[![CI](https://github.com/krendlV/microagent/actions/workflows/ci.yml/badge.svg)](https://github.com/krendlV/microagent/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/microagent.svg)](https://pypi.org/project/microagent/)
[![Python](https://img.shields.io/pypi/pyversions/microagent.svg)](https://pypi.org/project/microagent/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**microagent** is an open-source, agentic microscopy image analysis tool built in Python.

## Features

- Automated segmentation with Cellpose, StarDist, and Segment Anything (micro_sam)
- FAIR-compliant metadata handling via bioimage.io
- Experiment tracking with MLflow and hyperparameter optimization with Optuna
- MCP server integration for AI-driven workflows
- Rich CLI powered by Typer

## Installation

```bash
pip install microagent
```

With optional extras:

```bash
pip install "microagent[stardist,sam,bioimage,tracking,mcp]"
```

## Quick Start

```bash
microagent --help
```

## Development

```bash
# Install uv: https://docs.astral.sh/uv/
uv sync --dev
uv run pytest
uv run ruff check src/
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).
