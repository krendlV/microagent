# MicroAgent

[![CI](https://github.com/krendlV/microagent/actions/workflows/ci.yml/badge.svg)](https://github.com/krendlV/microagent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/microagent.svg)](https://pypi.org/project/microagent/)
[![Python](https://img.shields.io/pypi/pyversions/microagent.svg)](https://pypi.org/project/microagent/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/krendlV/microagent)](https://codecov.io/gh/krendlV/microagent)

**Open-source Python CLI + MCP server for automated microscopy image segmentation, evaluation, training, and reporting.**

MicroAgent wraps CellPose, StarDist, and micro-SAM behind a single command-line interface and a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server—letting you run reproducible segmentation pipelines from the terminal or directly from an AI assistant like Claude.

---

## 30-Second Quickstart

```bash
pip install microagent
microagent demo           # synthetic data → segment → report.html
```

With your own images:

```bash
microagent init --data-dir /path/to/images   # create project.yaml
microagent inspect /path/to/images           # QC check
microagent segment /path/to/images           # run segmentation → masks/
microagent report                            # generate report.html
```

---

## Features

| | |
|---|---|
| **Multi-model segmentation** | CellPose (cyto2/cyto3/cpsam), StarDist (fluorescence/H&E), micro-SAM |
| **Auto model selection** | Picks the best model based on your `project.yaml` metadata |
| **Quantitative evaluation** | F1, precision, recall, mean F1 across thresholds, panoptic quality |
| **Fine-tuning** | One-command CellPose fine-tuning on your annotated data |
| **Hyperparameter optimization** | Optuna-powered search over diameter, flow threshold, and more |
| **HTML reports** | Self-contained reports with overlay images, charts, and metrics |
| **FAIR provenance** | Auto-captured run metadata: git hash, library versions, data hash |
| **MCP server** | Full pipeline accessible to any MCP-compatible AI assistant |

---

## Architecture

```
images ──► inspect ──► segment ──► masks ──► evaluate ──► metrics ──► report
                          │                                               │
                      project.yaml                               experiments.jsonl
```

```
src/microagent/
├── cli.py            CLI entry point (Typer)
├── mcp_server.py     MCP server (FastMCP)
├── core/
│   ├── inspect.py    Image loading, QC, statistics
│   ├── segment.py    CellPose / StarDist / μSAM inference
│   ├── evaluate.py   Metrics via StarDist matching
│   ├── train.py      CellPose-SAM fine-tuning
│   └── optimize.py   Optuna hyperparameter search
├── viz/              Overlays, plots, HTML reports
├── fair/             Provenance metadata, experiment tracking
└── project/          project.yaml parsing and creation
```

---

## Supported Models

| Model | Backend | Best For | Weights License |
|-------|---------|----------|-----------------|
| `cyto2` | CellPose | Phase contrast, brightfield | BSD-3-Clause |
| `cyto3` | CellPose | Fluorescence whole-cell | BSD-3-Clause |
| `cpsam` | CellPose-SAM | General purpose (default) | **CC-BY-NC** |
| `2D_versatile_fluo` | StarDist | Fluorescence nuclei | BSD-3-Clause |
| `2D_versatile_he` | StarDist | H&E tissue nuclei | BSD-3-Clause |
| `micro_sam` | micro-SAM | EM / organelles / large irregular objects | Apache-2.0 |

> **License note:** MicroAgent source code is BSD-3-Clause. The `cpsam` model weights are released under **CC-BY-NC**—they may not be used for commercial purposes. If you need commercial use, switch to `cyto3` or `2D_versatile_fluo`.

---

## MCP Integration

Connect MicroAgent to Claude Code (or any MCP client) with three lines:

```json
{
  "mcpServers": {
    "microagent": {
      "command": "uvx",
      "args": ["microagent[mcp]", "mcp-server"]
    }
  }
}
```

Add this to `~/.claude/settings.json`, then ask Claude to segment your images. See [docs/mcp-integration.md](docs/mcp-integration.md) for the full setup guide and tool reference.

---

## Installation

```bash
pip install microagent                                  # core (CellPose)
pip install "microagent[stardist]"                     # + StarDist
pip install "microagent[tracking]"                     # + Optuna + MLflow
pip install "microagent[mcp]"                          # + MCP server
pip install "microagent[stardist,tracking,mcp]"        # recommended full install
conda install -c conda-forge micro_sam                 # optional micro-SAM backend
```

Requires **Python ≥ 3.10**. GPU is optional but recommended for datasets larger than ~100 images.

From source:

```bash
git clone https://github.com/krendlV/microagent
cd microagent
pip install -e ".[stardist,tracking,mcp,dev]"
```

---

## Documentation

- [Getting Started](docs/getting-started.md) — installation, first run, understanding output
- [User Guide](docs/user-guide.md) — project.yaml, model selection, fine-tuning, batch processing
- [API Reference](docs/api-reference.md) — every CLI command and option
- [MCP Integration](docs/mcp-integration.md) — Claude Code, Cursor, tool reference
- [project.yaml Reference](docs/project-yaml-reference.md) — full schema documentation
- [Architecture](docs/architecture.md) — module design, data flow, extension points
- [FAQ](docs/faq.md) — common issues, GPU troubleshooting, positioning

---

## Contributing

```bash
git clone https://github.com/krendlV/microagent
cd microagent
uv sync --extra dev
uv run pytest
uv run ruff check src/
```

Open an issue before starting significant work. PRs welcome.

---

## Citation

```bibtex
@software{microagent2024,
  author  = {Your Name},
  title   = {MicroAgent: Agentic Microscopy Image Analysis},
  year    = {2024},
  url     = {https://github.com/krendlV/microagent},
  license = {BSD-3-Clause}
}
```

---

## License

MicroAgent source: [BSD-3-Clause](LICENSE)
`cpsam` model weights: CC-BY-NC (non-commercial only)
