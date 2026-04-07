# MicroAgent Development Plan

> A roadmap for further development with actionable prompts for each improvement area.

---

## Current State (v0.1.0)

MicroAgent provides a CLI + MCP server for microscopy image analysis with:
- Image inspection and QC (`inspect`)
- Segmentation via CellPose/StarDist (`segment`)
- Evaluation against ground truth (`evaluate`)
- CellPose fine-tuning (`train`)
- Hyperparameter optimization via Optuna (`optimize`)
- HTML reporting, experiment tracking, and reproducibility tooling

---

## Phase 1: Robustness & Testing (Near-term)

### 1.1 Improve Error Handling Specificity

**Status:** Partially addressed — MCP server now includes `error_type`, `_load_project()` validates YAML.

**Remaining work:**
- Replace remaining broad `except Exception` blocks in `mcp_server.py` tool functions with specific exception types (e.g., `FileNotFoundError`, `ImportError`, `ValueError`, `yaml.YAMLError`)
- Add structured error codes for programmatic consumption

**Prompt for future improvement:**
> Review all `except Exception` blocks across the codebase. For each one, identify the specific exceptions that could be raised and replace the broad catch with targeted handlers. Ensure that `KeyboardInterrupt` and `SystemExit` are never accidentally caught. Add logging at appropriate levels (debug, warning, error) for each handler.

### 1.2 Increase Test Coverage for Error Paths

**Status:** Tests primarily cover happy paths.

**Remaining work:**
- Add tests for invalid YAML input (empty files, non-dict content, malformed syntax)
- Add tests for missing directories passed to CLI commands
- Add tests for edge cases in mask suffix matching
- Add tests for the `_load_project` YAML validation

**Prompt for future improvement:**
> Add pytest test cases for all error paths in the core modules. Focus on: (1) invalid YAML files (empty, non-dict, syntax errors) passed to `_load_project()`, (2) mismatched image/mask pairs in evaluate and train, (3) corrupt image files in inspect, (4) missing optional dependencies (mock ImportError for cellpose, stardist, optuna). Each test should verify both the error message and the exit behavior.

### 1.3 Input Validation in CLI Commands

**Status:** Basic validation exists but is incomplete.

**Remaining work:**
- Add pre-flight directory existence checks in `segment`, `evaluate`, `train`, `optimize` commands
- Validate channel indices are non-negative integers
- Validate threshold values are between 0 and 1
- Add `typer.BadParameter` callbacks for path arguments

**Prompt for future improvement:**
> Add input validation callbacks to all CLI commands in `cli.py`. Each Path argument should verify the path exists and is of the correct type (file vs directory). Numeric parameters should validate ranges (e.g., thresholds between 0-1, epochs > 0, diameter > 0). Use `typer.BadParameter` for clear error messages. Add corresponding tests for each validation.

---

## Phase 2: Code Quality & Consistency (Short-term)

### 2.1 Standardize Logging

**Status:** Logging module added to `inspect.py` and `optimize.py`.

**Remaining work:**
- Add `logging.getLogger(__name__)` to all core modules
- Add logging configuration in CLI entry point (verbosity flag)
- Replace `console.print()` error messages with logger + console dual output
- Add `--verbose` / `--quiet` flags to CLI

**Prompt for future improvement:**
> Add Python's `logging` module to every module in `src/microagent/core/`, `src/microagent/fair/`, and `src/microagent/viz/`. Add a `--verbose` flag to the main CLI app that sets log level to DEBUG. In each exception handler, log the error at the appropriate level before displaying to the user via rich console. Ensure log messages include the module name and are useful for debugging.

### 2.2 Resolve Remaining Linting Issues

**Status:** `Optional` → `X | None` migration complete. B008 (typer defaults) intentionally kept.

**Remaining work:**
- Fix F401 unused imports in `evaluate.py` and `train.py`
- Fix E501 line length violations
- Fix UP037 quoted type annotations
- Consider adding `# noqa: B008` comments to Typer defaults to suppress known false positives

**Prompt for future improvement:**
> Run `ruff check src/ --fix` to auto-fix safe issues. For remaining issues: (1) remove unused imports flagged by F401, (2) break long lines (E501) into multiple lines or use intermediate variables, (3) add `# noqa: B008` to all `typer.Argument()` and `typer.Option()` defaults since this is the standard Typer pattern. Run the full test suite after fixes to verify nothing breaks.

### 2.3 Type Checking with mypy

**Status:** `mypy` is configured but `type: ignore` comments are scattered.

**Remaining work:**
- Add per-package overrides in `pyproject.toml` for untyped dependencies
- Run `mypy src/` and fix reported issues
- Remove unnecessary `type: ignore` comments where possible

**Prompt for future improvement:**
> Configure mypy in `pyproject.toml` with per-package overrides for untyped dependencies (yaml, anthropic, pypdf, etc.) instead of individual `# type: ignore` comments. Run `mypy src/microagent/` and fix all reported errors. Focus on: missing return types, incompatible types in assignments, and missing type stubs. Add mypy to the CI pipeline.

---

## Phase 3: Features & Architecture (Medium-term)

### 3.1 μSAM Backend Integration

**Status:** Placeholder exists in `select_segmenter()` but no implementation.

**Remaining work:**
- Implement `MicroSAMSegmenter(Segmenter)` class
- Add prompt-based segmentation support
- Add automatic point/box prompt generation from project metadata
- Test with EM and organelle datasets

**Prompt for future improvement:**
> Implement a `MicroSAMSegmenter` class in `src/microagent/core/segment.py` following the `Segmenter` abstract base class interface. μSAM should be imported conditionally (conda-forge only). Implement `predict()` using `micro_sam.inference.inference_with_iterative_prompting`. Add the backend to `select_segmenter()` for organelle/EM targets. Add unit tests using synthetic data that mock the μSAM model.

### 3.2 3D Volume Support

**Status:** Only 2D images are supported.

**Remaining work:**
- Extend `_load_image()` and `_to_channels_first()` to handle Z-stacks
- Add 3D segmentation support (CellPose 3D, StarDist 3D)
- Update evaluation metrics for 3D volumes
- Add Z-projection options for inspection

**Prompt for future improvement:**
> Extend the segmentation pipeline to support 3D microscopy volumes. In `inspect.py`, detect 3D images (ndim==4 or ndim==3 with Z>16) and report Z-depth. In `segment.py`, add a `Segmenter3D` base class and implement `CellPose3DSegmenter` using `cellpose.models.Cellpose` with `do_3D=True`. Update `evaluate.py` to compute 3D IoU using `scipy.ndimage.label`. Add CLI flags `--mode 2d|3d|auto` to relevant commands.

### 3.3 MLflow Integration

**Status:** MLflow is listed as optional dependency but not used.

**Remaining work:**
- Add MLflow tracking to training and optimization
- Log parameters, metrics, and artifacts automatically
- Add `--mlflow` flag to CLI commands
- Support remote MLflow servers

**Prompt for future improvement:**
> Integrate MLflow tracking into the training and optimization pipelines. In `train.py`, add `mlflow.start_run()` context manager that logs TrainConfig parameters, per-epoch losses, and the final model artifact. In `optimize.py`, log each trial's parameters and metrics. Add `--mlflow-uri` option to CLI commands. Update `ExperimentTracker` to optionally mirror logs to MLflow. Guard all MLflow code with `try/except ImportError`.

### 3.4 Batch Processing & Parallelization

**Status:** Images are processed sequentially.

**Remaining work:**
- Add `concurrent.futures` parallel processing for segmentation
- Add batch-level progress reporting
- Support multi-GPU inference
- Add `--workers` flag to CLI

**Prompt for future improvement:**
> Add parallel image processing to `run_segmentation()` in `segment.py`. Use `concurrent.futures.ProcessPoolExecutor` with a configurable `--workers` parameter (default: 1 for backward compatibility). Each worker should load, segment, and save one image. Use a thread-safe progress callback for the CLI progress bar. Ensure GPU models are not shared across processes. Add tests verifying that parallel results match sequential results.

---

## Phase 4: Distribution & Documentation (Long-term)

### 4.1 CI/CD Pipeline

**Status:** No CI workflow configured.

**Remaining work:**
- Add GitHub Actions workflow for testing (Python 3.10, 3.11, 3.12)
- Add linting step (ruff, mypy)
- Add test coverage reporting
- Add release automation (PyPI publish on tag)

**Prompt for future improvement:**
> Create `.github/workflows/ci.yml` with: (1) matrix testing on Python 3.10/3.11/3.12, (2) `ruff check src/` linting step, (3) `pytest tests/ --cov=microagent` with coverage threshold of 80%, (4) `mypy src/` type checking. Add a separate `.github/workflows/release.yml` that publishes to PyPI on version tags using `hatch build` and `twine upload`. Add badges to README.md.

### 4.2 Documentation Site

**Status:** mkdocs.yml exists but docs/ content is minimal.

**Remaining work:**
- Write API reference documentation
- Add tutorials with real microscopy examples
- Add architecture decision records
- Deploy to GitHub Pages or ReadTheDocs

**Prompt for future improvement:**
> Build out the MkDocs documentation site. Add: (1) Getting Started tutorial with synthetic data, (2) API Reference auto-generated from docstrings using `mkdocstrings`, (3) User Guide for each command (inspect, segment, evaluate, train, optimize, report), (4) Developer Guide explaining the Segmenter abstract base class and how to add new backends. Configure `mkdocs.yml` with the Material theme and deploy to GitHub Pages via CI.

### 4.3 Plugin System for Custom Backends

**Status:** Segmenters use an abstract base class but are hardcoded.

**Remaining work:**
- Add entry point-based plugin discovery
- Allow third-party segmenter registration
- Document the plugin API

**Prompt for future improvement:**
> Implement a plugin system for segmentation backends using Python entry points. In `pyproject.toml`, define `[project.entry-points."microagent.segmenters"]`. In `segment.py`, add `discover_segmenters()` that uses `importlib.metadata.entry_points()` to find and load registered segmenter classes. Update `select_segmenter()` to consider plugins alongside built-in backends. Add documentation and a minimal example plugin package.

### 4.4 BioImage.IO Model Zoo Integration

**Status:** `bioimageio.core` is listed as optional dependency.

**Remaining work:**
- Add model export to BioImage.IO format
- Add model import from the Model Zoo
- Include proper model cards with metadata

**Prompt for future improvement:**
> Add BioImage.IO Model Zoo integration. Implement `export_bioimage_model()` that wraps a fine-tuned CellPose model in the BioImage.IO model format with proper metadata (covers, documentation, test tensors). Implement `import_bioimage_model()` that loads a model from a BioImage.IO RDF and wraps it in the `Segmenter` interface. Add CLI commands `microagent export-model` and `microagent import-model`. Guard with `try/except ImportError` for `bioimageio.core`.

---

## Priority Summary

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 🔴 High | Error path tests (1.2) | Medium | High |
| 🔴 High | CLI input validation (1.3) | Low | High |
| 🟡 Medium | Standardize logging (2.1) | Medium | Medium |
| 🟡 Medium | CI/CD pipeline (4.1) | Medium | High |
| 🟡 Medium | Fix remaining lint (2.2) | Low | Medium |
| 🟢 Low | μSAM backend (3.1) | High | High |
| 🟢 Low | 3D support (3.2) | High | Medium |
| 🟢 Low | MLflow integration (3.3) | Medium | Medium |
| 🟢 Low | Plugin system (4.3) | Medium | Low |
| 🟢 Low | BioImage.IO (4.4) | High | Medium |
