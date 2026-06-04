# MicroAgent — Development Prompts for Claude Code

These are ready-to-paste prompts for Claude Code, ordered by priority. Each is
self-contained: it states the problem, the files involved, the fix, and the
acceptance check. Work top-to-bottom — the first three turn the test suite green.

> Conventions reminder (from `CLAUDE.md`): Python ≥3.10, type hints on public
> functions, `pathlib.Path`, `rich` for output, NumPy-style docstrings, optional
> deps guarded with `try/except ImportError`, tests use synthetic images only.
> After every change run: `uv run pytest -q` and `uv run ruff check src/`.

---

## P0 — Turn the test suite green

### Prompt 1 — Fix the `RunMetadata` API drift (4 failing report tests)

```
`tests/test_report.py` fails with two errors:
  - `RunMetadata.__init__() got an unexpected keyword argument 'timestamp'`
  - `type object 'RunMetadata' has no attribute 'collect'`

The dataclass in `src/microagent/fair/provenance.py` has a field `timestamp_utc`
(not `timestamp`) and exposes a module-level `collect_metadata(...)` function with
no `RunMetadata.collect` classmethod, but the tests expect `timestamp` and
`RunMetadata.collect(command=..., seed=...)`.

Decide on ONE canonical API and make code + tests agree. Preferred direction:
keep the dataclass field as `timestamp_utc` (more precise) but ADD a
`@classmethod collect(cls, command: str = "", seed: int = 0, **kwargs)` to
`RunMetadata` that simply delegates to `collect_metadata(command=command,
random_seed=seed, **kwargs)`. Then update `tests/test_report.py`'s
`_make_provenance` helper to pass `timestamp_utc=` instead of `timestamp=`.

Do not change any field semantics elsewhere; `viz/report.py` and `cli.py export`
already construct `RunMetadata` with `timestamp_utc` and must keep working.

Acceptance: `uv run pytest tests/test_report.py -q` passes.
```

### Prompt 2 — Make `optimize` tests skip cleanly without Optuna (11 failing tests)

```
All of `tests/test_optimize.py` fails with `ImportError: optuna is required` because
`optuna` is only declared under the optional `tracking` extra in `pyproject.toml`
and isn't installed in the dev environment.

Pick the cleaner of these and implement it consistently:
  (A) Add a module-level `optuna = pytest.importorskip("optuna")` at the top of
      `tests/test_optimize.py` so the whole module skips when Optuna is absent, OR
  (B) Add `optuna>=4.8` to the `dev` extra in `pyproject.toml` so it's always present
      for `uv sync --extra dev`.

Choose (A) — it keeps the dev install lean and matches how `stardist`-dependent code
is already optional. Verify no other test module imports optuna unguarded.

Acceptance: `uv run pytest -q` reports the optimize tests as skipped (not failed)
in an env without optuna, and as passing in an env with it.
```

### Prompt 3 — Validate training data instead of crashing on sparse masks

```
`tests/test_train.py::test_train_smoke` fails with `ZeroDivisionError: float division
by zero` originating inside CellPose: when training images contain fewer than
`min_train_masks` (5) labelled objects, CellPose removes them all, leaving `nimg == 0`.

In `src/microagent/core/train.py::train_cellpose`, after loading `train_images,
train_masks` (and before initialising the model), add a pre-flight check:
  - count distinct nonzero labels per mask
  - if the number of images with >= 5 objects is 0, raise a clear `ValueError`
    explaining that CellPose requires at least 5 labelled objects per image and
    reporting how many images were too sparse.
  - expose the threshold as a module constant `MIN_TRAIN_MASKS = 5`.

Also update the test fixture so the smoke test generates masks with >= 5 objects each
(so the happy path is actually exercised), and add a NEW test asserting the clear
`ValueError` is raised for an all-sparse dataset.

Acceptance: `uv run pytest tests/test_train.py -q` passes; the error path is covered.
```

---

## P1 — Wire the integration seams that are currently dead-ended

### Prompt 4 — Thread provenance/experiment tracking through the pipeline

```
`src/microagent/fair/tracking.py` (`ExperimentTracker`, `tracked_run`) and
`fair/provenance.py` (`collect_metadata`) are fully implemented and tested, but NO CLI
or MCP command ever calls them. As a result `experiments.jsonl` is never written and
`microagent export --run <id>` can never find a run.

Wire `tracked_run` into the `segment`, `train`, `evaluate`, and `optimize` commands in
`src/microagent/cli.py`. For each:
  - build a `params` dict of the user-facing arguments,
  - wrap the core call in `with tracked_run(ExperimentTracker(), command_str, params,
    data_path=<input dir>) as results:` and populate `results` with key outputs
    (e.g. mask count + elapsed for segment; summary F1/mean_f1/PQ for evaluate; best_params
    + improvement for optimize; model_path + best loss for train),
  - print the resulting 8-char `run_id` to the user (e.g. `[dim]run abc12345 logged →
    experiments.jsonl[/dim]`) so they can later `export --run abc12345`.

Keep it opt-out-able: add a global `--no-track` flag (default off) so users can disable
logging. Do not log on error paths that exited early via `typer.Exit`.

Add an integration test that runs `segment` (or a stubbed core call) and asserts a line
was appended to `experiments.jsonl` and that `export --run <that id>` resolves it.

Acceptance: after `microagent segment ...`, `experiments.jsonl` contains a record and
`microagent export --run <id> --format bundle` succeeds.
```

### Prompt 5 — Persist `optimization.json` from the optimize command

```
The `report` command (`cli.py:819`) and the MCP `generate_report` tool auto-detect a
file named `optimization.json`, but `run_optimization` in
`src/microagent/core/optimize.py` never writes one — it only pickles `optuna_study.pkl`.
So the optimization section can never appear in a report.

Add JSON persistence:
  - Give `OptimizationResult` a `save_json(self, path: Path)` method that serialises
    `best_params`, `best_value`, `baseline_value`, `improvement`, and a compact list of
    trial records (`number`, `params`, `value`).
  - In `cli.py::optimize`, after the run completes, write `optimization.json` next to the
    other result files (default cwd; add an `--output-json` option defaulting to
    `optimization.json`), and print the path.
  - Confirm `viz/report.py::load_report_data` already knows how to render this shape; if
    not, extend it to include an "Optimization" section (best params + improvement +
    trial sparkline/table).

Acceptance: `microagent optimize ... && microagent report` produces a report containing
the optimization results; add a test for `OptimizationResult.save_json` round-trip.
```

### Prompt 6 — Make `segment` actually use the project's recommended model

```
There are two divergent model-selection code paths:
  - `core/segment.py::select_segmenter` (used by `run_segmentation`)
  - `project/knowledge.py::recommend_model` (used by `init`, written into project.yaml
    as `recommended_model` / `recommended_params`)

They disagree (recommend_model knows cyto2/cyto3/2D_versatile_he with tuned params;
select_segmenter only ever yields default cpsam or fluo/he stardist), and
`run_segmentation` IGNORES the `recommended_model`/`recommended_params` saved in
project.yaml. A user who runs `init` then `segment --project project.yaml` does not get
the recommended model.

Reconcile them:
  - When `run_segmentation` is given a project that contains `recommended_model` and
    `recommended_params`, honour them (instantiate that backend with those params)
    unless the user explicitly overrode `--model`/`--diameter` on the CLI.
  - Make `select_segmenter` delegate to / share the decision matrix in `recommend_model`
    rather than maintaining a second, weaker copy. Extract the matrix into one function
    if helpful.
  - Preserve graceful fallback when the recommended backend's optional dep isn't installed.

Acceptance: a test that builds a project.yaml recommending stardist `2D_versatile_he`
and asserts `run_segmentation(..., project_path=...)` selects a `StarDistSegmenter` with
that model; CLI `--model`/`--diameter` still override.
```

---

## P2 — Correctness & honesty

### Prompt 7 — Completed: use `mean_f1` for threshold-mean F1

```
Resolved. Evaluation dataclasses, JSON, CLI, MCP output, reports, docs, and tests now
use `mean_f1` for the mean of F1 across IoU thresholds.

Acceptance: `uv run pytest -q` green; no stale labels for this threshold-mean F1 score.
```

### Prompt 8 — Surface the silently-ignored CellPose v4 args

```
CellPose >= 4.0.1 ignores `model_type=` (emits "model_type argument is not used in
v4.0.1+") and may ignore the `channels=` kwarg too. In
`src/microagent/core/train.py::train_cellpose` the `--pretrained` value is passed as
`model_type` and therefore has no effect; in `core/segment.py` `channels` is passed to
`model.eval`.

Make the behavior honest:
  - In training, if `pretrained` cannot actually be applied on the installed CellPose
    version, either use the correct v4 mechanism for selecting a base model or warn the
    user (`rich` warning) that `--pretrained` is ignored on this CellPose version.
  - In segmentation, verify whether `channels` is honoured on the installed version; if
    not, drop it or warn. Detect the version via `importlib.metadata.version("cellpose")`.
  - Add a small unit test (mock CellPose) asserting the warning fires on v4.

Acceptance: running train/segment on CellPose v4 no longer silently swallows a flag the
user set; behavior is logged.
```

### Prompt 9 — Implement μSAM backend or remove it from the docs

```
`CLAUDE.md` and `core/segment.py::select_segmenter`'s docstring advertise μSAM
(micro-sam) as a segmentation backend, but it is not implemented — "organelles / EM"
silently falls back to CellPose cpsam.

Pick one:
  (A) Implement `MicroSamSegmenter(Segmenter)` in `core/segment.py`, guarded by
      `try/except ImportError` for `micro_sam` (note: conda-forge only, document that),
      following the same `predict`/`get_info`/`get_default_params` contract as the other
      backends, and route EM/organelle targets to it in `select_segmenter` when available
      (falling back to cpsam when not). Add tests that skip when micro_sam is absent. OR
  (B) Remove μSAM from `CLAUDE.md`, the segment docstring, and any README/docs that
      mention it, and make the EM/organelle fallback to cpsam explicit and documented.

If unsure about (A)'s API surface, do (B) now and open a tracked follow-up for (A).

Acceptance: docs and code agree on which backends exist; suite stays green.
```

### Prompt 10 — Cleanup pass: dead code & filename consistency

```
Small hygiene fixes across the repo:

1. `cli.py::demo` (~lines 700–717): remove the unused `combined` array and the unused
   `import numpy as np` / `from skimage import measure as _measure` inside that block —
   `plot_object_size_distribution` only needs `combined_2d`.
2. Remove redundant local re-imports of `rich.table.Table` / `rich.live.Live` in the
   `optimize`/`train` commands where they're already imported or unused.
3. Unify result-JSON filenames so auto-detection works regardless of entry point:
   pick canonical names (suggest `inspection.json`, `segmentation.json`, `metrics.json`,
   `optimization.json`) and make `cli.py`, `mcp_server.py::generate_report`, and
   `viz/report.py::load_report_data` all agree. Currently MCP looks for
   `evaluation.json` while demo/CLI write `metrics.json`, and segmentation metadata is
   `masks/segmentation_metadata.json` vs auto-detected `segmentation.json`.
4. In `core/segment.py`, replace the private-attribute mutation in `_make_cellpose`
   (`seg._diameter = ...`) with a proper constructor or a `CellPoseSegmenter.from_project`
   classmethod; delete the no-op `_apply_project_params` or implement StarDist param
   application.

Keep each change small and run `uv run ruff check src/` + `uv run pytest -q` after.

Acceptance: ruff clean, suite green, report auto-detection works from both CLI and MCP.
```

---

## Suggested execution order

1. **Prompts 1–3** (P0) in one sitting → green suite.
2. **Prompts 4–6** (P1) → the documented end-to-end pipeline actually holds together.
3. **Prompts 7–10** (P2) → correctness, honesty, polish.

After P0+P1, run the demo end-to-end as a smoke test:
`uv run microagent demo --no-browser` and confirm a report with all sections plus an
`experiments.jsonl` entry that `export --run` can resolve.
