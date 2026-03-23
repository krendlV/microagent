# Architecture

## Overview

MicroAgent is a thin orchestration layer around best-in-class segmentation libraries (CellPose, StarDist, micro-SAM). It adds a unified CLI, a project metadata system, quantitative evaluation, FAIR provenance tracking, and an MCP server — without reimplementing any segmentation logic.

---

## Module Dependency Diagram

```
cli.py ──────────────────────────────────────────────────────┐
mcp_server.py ───────────────────────────────────────────────┤
                                                              ▼
                                                    core/inspect.py
                                                    core/segment.py ──► cellpose
                                                                   ──► stardist
                                                                   ──► micro_sam
                                                    core/evaluate.py
                                                    core/train.py ───► cellpose
                                                    core/optimize.py ► optuna

project/knowledge.py ◄──── all commands that accept -p flag

viz/overlays.py ◄────────── segment output
viz/plots.py ◄───────────── evaluate output
viz/report.py ◄──────────── overlays + plots + metadata

fair/provenance.py ◄───────── all commands (auto-collected)
fair/tracking.py ◄─────────── all commands (append-only log)
```

---

## Data Flow

```
Input images (TIFF, OME-TIFF, CZI, ND2, PNG, JPG)
        │
        ▼
core/inspect.py
  → InspectionReport (file_count, dimensions, channel_stats, qc_warnings)
        │
        ▼
core/segment.py
  → Segmenter.predict() per image
  → integer label masks saved as TIFF
  → SegmentationResult (mask_paths, model_info, per_image_stats)
        │
        ├──► viz/overlays.py → overlay composites (PNG)
        │
        ▼
core/evaluate.py  (if ground truth available)
  → StarDist matching (or scipy fallback)
  → EvaluationResult (per_image, summary: F1/mAP/PQ at multiple thresholds)
        │
        ├──► viz/plots.py → metric charts (PNG)
        │
        ▼
viz/report.py
  → Jinja2 template + base64-embedded images
  → self-contained report.html
        │
fair/provenance.py ──► RunMetadata (versions, data_hash, git_commit, timestamp)
fair/tracking.py ────► experiments.jsonl (append-only run log)
```

---

## Key Design Decisions

### Abstract Segmenter base class

```python
class Segmenter(ABC):
    def predict(self, image: np.ndarray, **kwargs) -> np.ndarray: ...
    def get_info(self) -> dict: ...
    def get_default_params(self, project: ProjectConfig) -> dict: ...
```

`CellPoseSegmenter` and `StarDistSegmenter` both implement this interface. Adding a new backend requires only implementing these three methods — no changes elsewhere.

### Typed dataclass results

Every significant function returns a typed dataclass (e.g. `EvaluationResult`, `SegmentationResult`). This makes JSON serialization trivial, enables type checking, and provides a stable interface between modules.

### Optional dependencies via try/except

```python
try:
    import stardist
    HAS_STARDIST = True
except ImportError:
    HAS_STARDIST = False
```

The core package works without StarDist, Optuna, or MCP installed. Features degrade gracefully with clear error messages.

### Project-driven configuration

`project.yaml` is the single source of truth for dataset metadata and model selection. Commands that accept `-p project.yaml` use it to auto-select models, parameters, and report metadata. Commands work without it (falling back to defaults), but quality improves with it.

### No GUI in core

`matplotlib` uses the `Agg` backend only — no display, no tkinter, no Qt. All visualization is to files or base64 strings. This keeps the library headless-server-safe.

### MCP = CLI in disguise

`mcp_server.py` wraps the same functions as `cli.py`. There is no separate code path. Adding a new CLI command only requires adding an MCP tool wrapper to expose it via MCP.

---

## FAIR Provenance Design

Every run automatically captures:

**Reproducibility:**
- `microagent_version`, `cellpose_version`, `stardist_version`, `numpy_version`
- `python_version`, `platform`
- `cuda_version`, `gpu_name`, `gpu_vram_mb`
- `git_commit` (HEAD SHA), `git_dirty` (uncommitted changes)
- `data_hash` (SHA-256 of input directory)
- `random_seed`
- `timestamp_utc` (ISO 8601)
- `command` (full CLI invocation string)
- `wall_clock_seconds`

This is embedded in every `segmentation_metadata.json`, `report.html`, and `experiments.jsonl` entry. The goal: given the raw data and metadata, another researcher can reproduce the exact result.

---

## Extension Points

### Adding a new segmentation backend

1. Create `core/segment_mymodel.py`
2. Implement the `Segmenter` abstract base class
3. Add `"mymodel"` to the `select_segmenter()` dispatch in `core/segment.py`
4. Add `"mymodel"` to model selection logic in `project/knowledge.py`
5. Add optional import guard in `core/segment.py`

### Adding new metrics

1. Add computation to `core/evaluate.py` in `_compute_image_metrics()`
2. Add the field to `ThresholdMetrics` or `ImageMetrics` dataclass
3. Update `viz/plots.py` and `viz/report.py` Jinja template to display it

### Adding new report sections

`viz/report.py` uses a Jinja2 template. Add new sections to the template and pass additional data from `generate_report()`.

---

## Testing Strategy

Tests use **synthetic images only** — no real microscopy data in the repository.

```
tests/
├── test_inspect.py    # QC on synthetic images
├── test_segment.py    # Segmentation with mock Segmenter
├── test_evaluate.py   # Metrics with known ground truth
├── test_train.py      # Training smoke test
├── test_optimize.py   # Optuna integration
└── conftest.py        # Shared fixtures (synthetic images, temp dirs)
```

Slow tests (real model inference) are marked `@pytest.mark.slow` and excluded from the default test run. Integration tests are marked `@pytest.mark.integration`.

```bash
uv run pytest                    # fast tests only
uv run pytest -m slow            # include slow tests
uv run pytest -m integration     # include integration tests
```
