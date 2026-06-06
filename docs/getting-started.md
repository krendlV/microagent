# Getting Started

## Prerequisites

- **Python 3.10 or later**
- pip or [uv](https://docs.astral.sh/uv/) package manager
- GPU optional — CUDA-capable GPU with ≥4 GB VRAM recommended for large datasets

---

## Installation

### Standard install (pip)

```bash
pip install microagent
```

### With all recommended extras

```bash
pip install "microagent[stardist,tracking,mcp]"
```

| Extra | What it adds |
|-------|-------------|
| `stardist` | StarDist segmentation models |
| `tracking` | Optuna HPO + MLflow experiment tracking |
| `mcp` | MCP server for AI assistant integration |
| `dev` | pytest, ruff, mypy, pre-commit |

### With uv (recommended for development)

```bash
uv sync                       # core dependencies
uv sync --extra dev           # + dev tools
```

### From source

```bash
git clone https://github.com/krendlV/microagent
cd microagent
pip install -e ".[stardist,tracking,mcp,dev]"
```

---

## First Run: Demo Data

The fastest way to see MicroAgent in action:

```bash
microagent demo
```

This will:
1. Generate 10 synthetic fluorescence images with known ground truth
2. Run QC inspection
3. Segment cells with CellPose-SAM
4. Evaluate segmentation quality
5. Generate `report.html` and open it in your browser

Options:

```bash
microagent demo --output ./my-demo --n-images 20 --no-browser
```

---

## First Run: Your Own Data

### One command — the quick path

```bash
microagent run /path/to/images
```

This single command runs the full pipeline — inspect → segment → overlays → report — and
opens `microagent_output/report.html` in your browser when done.

Options:

```bash
# Save everything to a custom directory
microagent run /path/to/images --output ./results

# Also evaluate against ground-truth masks (adds F1, PQ metrics to the report)
microagent run /path/to/images --ground-truth /path/to/gt --output ./results

# Use a project.yaml for model selection and parameters
microagent run /path/to/images --project project.yaml

# Skip opening the browser (useful in scripts / CI)
microagent run /path/to/images --no-open
```

Output written under `--output` (default `microagent_output/`):

| File | Contents |
|------|----------|
| `masks/` | Labeled TIFF masks, one per input image |
| `overlays/` | Overlay montage PNG |
| `plots/` | Metric charts (if evaluation was run) |
| `inspection.json` | QC report |
| `segmentation.json` | Per-image cell counts, model info, provenance |
| `metrics.json` | F1, PQ per image (only when `--ground-truth` is given) |
| `report.html` | Self-contained HTML report |

---

### Step-by-step — advanced / scripting use

For finer control or scripting, each pipeline phase is also a separate command.

#### Step 1 — Initialize a project (optional)

```bash
microagent init --data-dir /path/to/images
```

This launches an interactive wizard that asks about your:
- Organism (human, mouse, etc.)
- Sample type (cell culture, tissue section, etc.)
- Imaging modality (fluorescence, brightfield, confocal, etc.)
- Structures of interest (nuclei, whole cells, etc.)

It then creates `project.yaml` with a recommended model and parameters. You can also extract project metadata from a methods document:

```bash
microagent init --data-dir /path/to/images --doc methods.md
```

#### Step 2 — Inspect your images

```bash
microagent inspect /path/to/images
```

Reports file count, dimensions, intensity statistics per channel, and flags QC warnings (mismatched dimensions, unusual intensity ranges, load failures). Save the report with `-o`:

```bash
microagent inspect /path/to/images -o qc_report.json
```

#### Step 3 — Segment

```bash
microagent segment /path/to/images
```

Uses the model recommended in `project.yaml` (or auto-select if no YAML). Masks are saved to `masks/` as TIFF files alongside `segmentation_metadata.json`.

```bash
# Explicit model and diameter
microagent segment /path/to/images --model cellpose --diameter 30

# Custom output directory
microagent segment /path/to/images -o /path/to/masks
```

#### Step 4 — Evaluate (if you have ground truth)

```bash
microagent evaluate masks/ /path/to/ground_truth/
```

Outputs F1, precision, recall, mean F1 across thresholds, and panoptic quality. Ground truth files are matched to predictions by filename stem.

#### Step 5 — Generate a report

```bash
microagent report
```

Creates `report.html` — a self-contained HTML file with overlay composites, metric charts, and provenance metadata.

---

## Understanding the Output

### Masks directory

```
masks/
├── image001_mask.tif       # integer-labeled mask (0 = background)
├── image002_mask.tif
└── segmentation_metadata.json
```

Each mask TIFF uses integer labels where 0 is background and each positive integer is a unique cell instance.

### segmentation_metadata.json

```json
{
  "model_info": {"model_name": "cpsam", "backend": "cellpose"},
  "parameters": {"diameter": null, "flow_threshold": 0.4},
  "per_image_stats": [
    {"filename": "image001.tif", "n_labels": 47, "elapsed_seconds": 1.2}
  ],
  "provenance": {
    "microagent_version": "0.1.0",
    "cellpose_version": "3.0.0",
    "git_commit": "abc1234",
    "data_hash": "sha256:...",
    "timestamp_utc": "2024-01-15T10:30:00Z"
  }
}
```

### report.html

A self-contained HTML file (no external dependencies) containing:
- Summary statistics table
- Per-image overlay composites
- Metric charts (F1 vs. IoU threshold, cell count distribution)
- Full provenance metadata for reproducibility

### experiments.jsonl

Append-only log of every run. Each line is a JSON record with parameters, metrics, and provenance. Useful for comparing runs over time.
