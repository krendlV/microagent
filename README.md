# MicroAgent

[![CI](https://github.com/krendlV/microagent/actions/workflows/ci.yml/badge.svg)](https://github.com/krendlV/microagent/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-pre--release-orange.svg)](https://github.com/krendlV/microagent)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Open-source Python CLI + MCP server for automated microscopy image segmentation, evaluation, training, and reporting.**

MicroAgent wraps CellPose, StarDist, and micro-SAM behind a single command-line interface and a [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server—letting you run reproducible segmentation pipelines from the terminal or directly from an AI assistant like Claude.

---

> **Status:** pre-release (`0.1.0`). Not yet published to PyPI — install from source
> (see [Installation](#installation)). The pipeline, MCP server, and reproducibility
> export are fully working today.

## 30-Second Quickstart

```bash
git clone https://github.com/krendlV/microagent && cd microagent
uv sync                    # or: pip install -e .
uv run microagent demo     # synthetic data → segment → evaluate → report.html
```

With your own images:

```bash
uv run microagent init --data-dir /path/to/images   # create project.yaml (interactive)
uv run microagent inspect /path/to/images           # QC check
uv run microagent segment /path/to/images           # run segmentation → masks/
uv run microagent report                            # generate report.html
```

Every pipeline command logs a reproducible run to `experiments.jsonl`; bundle one up with
`microagent export --run <id> --format bundle`. Use `--no-track` to opt out.

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
| **FAIR provenance** | Auto-captured run metadata: git hash, library versions, data hash, GPU/CUDA, timing |
| **Reproducibility export** | One-command Docker / Apptainer / zip bundle to re-run any logged experiment |
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
| `micro_sam` | micro-SAM (conda-forge only — not on PyPI) | EM / organelles / large irregular objects | Apache-2.0 |

> **License note:** MicroAgent source code is BSD-3-Clause. The `cpsam` model weights are released under **CC-BY-NC**—they may not be used for commercial purposes. If you need commercial use, switch to `cyto3` or `2D_versatile_fluo`.

---

## MCP Integration

Connect MicroAgent to Claude Code (or any MCP client). After cloning and installing from source, add this to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "microagent": {
      "command": "uv",
      "args": ["--directory", "/path/to/microagent", "run", "microagent-mcp-server"]
    }
  }
}
```

Then ask Claude to segment your images. Once MicroAgent is published to PyPI, you can use `uvx` instead. See [docs/mcp-integration.md](docs/mcp-integration.md) for the full setup guide and tool reference.

---

## Installation

Requires **Python ≥ 3.10**. GPU is optional but recommended for datasets larger than ~100 images.

**Install from source (available now):**

```bash
git clone https://github.com/krendlV/microagent
cd microagent
uv sync                                        # core + dev deps (recommended)
# or: pip install -e ".[stardist,tracking,mcp,dev]"
```

Optional micro-SAM backend (not on PyPI — conda-forge only):

```bash
conda install -c conda-forge micro_sam
```

**Once published to PyPI (not yet live):**

```bash
pip install microagent                                  # core (CellPose)
pip install "microagent[stardist]"                     # + StarDist
pip install "microagent[tracking]"                     # + Optuna + MLflow
pip install "microagent[mcp]"                          # + MCP server
pip install "microagent[stardist,tracking,mcp]"        # recommended full install
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
@software{microagent2026,
  author  = {Krendl, Valentin},
  title   = {MicroAgent: Agentic Microscopy Image Analysis},
  year    = {2026},
  url     = {https://github.com/krendlV/microagent},
  license = {BSD-3-Clause}
}
```

---

## License

MicroAgent source: [BSD-3-Clause](LICENSE)
`cpsam` model weights: CC-BY-NC (non-commercial only)
