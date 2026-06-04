# MicroAgent — Code Review & Project Status

**Review date:** 2026-06-04
**Reviewed at commit:** `1699612` (main)
**Scope:** Full `src/microagent/` tree, test suite, packaging, and pipeline wiring.

---

## 1. Executive summary

MicroAgent is in a **strong "wide but shallow-in-places" state**. The architecture
described in `CLAUDE.md` is real and faithfully implemented: there is a clean Typer
CLI, a parallel MCP server, an abstract `Segmenter` base with CellPose/StarDist
backends, a genuine evaluation module with a scipy fallback, an Optuna HPO loop, a
synthetic-data demo, an HTML report generator, and a FAIR/provenance subsystem.
Code style is consistent — type hints, NumPy docstrings, `pathlib`, `rich` output,
optional deps guarded with `try/except ImportError` exactly as the conventions require.

The gap is **integration and correctness at the seams**, not breadth. The test suite
is substantial (~220 tests) but **16 are currently failing**, and several subsystems
are built but never connected to the pipeline that is supposed to use them. The most
important issues are: provenance/experiment-tracking is never invoked by any command
(so `export --run` can never find a run), the `optimize` command never persists the
JSON the `report` command looks for, the `RunMetadata` API has drifted away from its
tests, and training crashes on sparse masks.

**Overall grade: solid alpha.** The foundation is good enough to build on; it is not
yet releasable because the cross-module promises (provenance → export, optimize →
report) are unfulfilled and the suite is red.

---

## 2. Test suite status

```
16 failed, 203 passed, 1 skipped
```

| Failing group | Count | Root cause | Severity |
|---|---|---|---|
| `test_optimize.py::*` | 11 | `optuna` is an **optional** dep (`tracking` extra) and is not installed; tests fail instead of skipping | Medium (test hygiene) |
| `test_report.py::*` | 4 | `RunMetadata` API drift — tests use `timestamp=`/`RunMetadata.collect(...)`, code has `timestamp_utc` field and module-level `collect_metadata()` | High (real API mismatch) |
| `test_train.py::test_train_smoke` | 1 | `ZeroDivisionError` inside CellPose: synthetic masks have < `min_train_masks` (5) objects, all images get filtered out, `nimg == 0` | High (robustness bug) |

A green suite is the single highest-leverage thing to fix — three distinct root
causes, all small.

---

## 3. Architecture observations

### 3.1 What is well done
- **`core/evaluate.py`** is the strongest module: dual path (stardist `matching` when
  available, scipy Hungarian fallback otherwise), clean dataclasses, JSON round-trip
  with `load_json`, per-image + dataset aggregation, best/worst callouts, run-vs-run
  comparison. The fallback IoU matrix + `linear_sum_assignment` is correct.
- **CLI/MCP parity** — every CLI verb has an MCP tool wrapping the same core function.
  Errors are returned as `{"status": "error", ...}` rather than raised, which is the
  right shape for an MCP tool.
- **Optional-dependency discipline** is consistent and correct across `segment`,
  `evaluate`, `train`, `optimize`, `inspect`, and `report`.
- **`inspect.py`** QC heuristics (dtype mismatch, dimension mismatch, near-zero,
  near-saturation, single-image) are sensible and well-factored.

### 3.2 Structural issues

**(A) Provenance/tracking is dead-ended.** `fair/tracking.py` (`ExperimentTracker`,
`tracked_run`) and `fair/provenance.py` (`collect_metadata`) are fully implemented and
tested in isolation — but **no CLI or MCP command ever calls `tracked_run` or
`ExperimentTracker.log_run`.** Consequences:
- `experiments.jsonl` is never written by the pipeline.
- `microagent export --run <id>` therefore can never resolve a run ID in normal use.
- The "FAIR / reproducibility" story in the docs is aspirational, not wired.

This is the single biggest gap. The plumbing exists; it just needs to be threaded
through `segment`/`train`/`evaluate`/`optimize`.

**(B) `optimize` → `report` handoff is broken.** The `report` command auto-detects
`optimization.json` (`cli.py:819`) and the MCP `generate_report` tool looks for it too,
but **`run_optimization` never writes any JSON** — it only pickles an `optuna_study.pkl`.
So the optimization section can never appear in a report. Either `optimize` should emit
`optimization.json`, or the report wiring is pointing at a phantom file.

**(C) Two divergent model-selection implementations.** `core/segment.py::select_segmenter`
and `project/knowledge.py::recommend_model` both encode a "modality/structure → model"
decision matrix, but they **disagree**: `recommend_model` returns models like `cyto2`,
`cyto3`, `2D_versatile_he` with rich params; `select_segmenter` only ever produces a
default `cpsam` CellPose or a fluo/he StarDist and ignores most of that. The
`recommended_model`/`recommended_params` written into `project.yaml` by `init` are
**not consulted** by `run_segmentation`. A user who runs `init` then `segment --project`
does not get the model the interview recommended.

**(D) μSAM is vaporware in the tree.** `CLAUDE.md` and `select_segmenter`'s docstring
list μSAM as a backend; in reality "organelles / EM" silently falls back to CellPose
`cpsam`. Either implement a `MicroSamSegmenter` (guarded optional dep) or stop
advertising it.

### 3.3 Correctness / scientific concerns
- **Threshold-mean F1 has been renamed.** The evaluation summary now exposes
  `mean_f1` for the mean of F1 across IoU thresholds, which avoids implying a
  literature AP metric.
- **`pretrained` is silently ignored in training.** `train_cellpose` passes
  `model_type=config.pretrained` to `CellposeModel`, but CellPose ≥4.0.1 emits
  *"model_type argument is not used in v4.0.1+. Ignoring this argument"*. So `--pretrained`
  has no effect on cpsam; the flag is a no-op the user can't see.
- **`train_cellpose` crashes instead of validating.** When images carry fewer than
  CellPose's `min_train_masks`, CellPose strips them and divides by zero. MicroAgent
  should pre-check object counts and raise a clear `ValueError` ("N images have < 5
  labelled objects; CellPose needs at least 5") long before CellPose's internals fail.

### 3.4 Minor / cleanup
- `cli.py` `demo` command (≈ lines 705–714): `combined` array and the `numpy`/`skimage.measure`
  imports are computed/imported and never used — dead code.
- `cli.py` `optimize`/`train` re-import `rich.table.Table` and `rich.live.Live` locally
  even though some are unused in that scope.
- `segment.py::_make_cellpose` mutates private attributes (`seg._diameter = ...`) from
  outside the class. Prefer a constructor/`from_project` classmethod.
- `segment.py::_apply_project_params` is a no-op `pass` — either implement StarDist
  param application from `project.yaml` (`prob_thresh`, `nms_thresh`) or delete it.
- CellPose `eval(..., channels=...)` is passed in `segment.py`; `channels` is also a
  deprecated arg in CellPose v4 — verify it isn't silently ignored like `model_type`.
- Filename convention inconsistency: demo/CLI write `metrics.json`; MCP `generate_report`
  looks for `evaluation.json`; segmentation metadata is `segmentation_metadata.json` in
  `masks/` but `report` auto-detects `segmentation.json` at top level. These near-misses
  mean auto-detection quietly fails depending on entry point.
- `mcp_server.py::evaluate` uses `asdict(result)` directly; fine today, but if any
  dataclass later holds a `Path` it will break JSON serialization the way `train` already
  had to hand-patch. Consider a single `to_jsonable()` helper.

---

## 4. Packaging & tooling
- `pyproject.toml` is clean and well-organized; optional extras (`stardist`, `tracking`,
  `mcp`, `bioimage`, `all`, `dev`) are sensible.
- **Gap:** `dev`/`all` do not include `optuna` even though `test_optimize.py` requires it
  at import time. Either add `optuna` to `dev` or make the tests `pytest.importorskip`.
- CI (`.github/workflows/ci.yml`) exists — confirm it runs the *same* extras the tests
  need, or it will be red for the same reason locally observed here.
- `sample_data/` (37 MB) was untracked and is now git-ignored (this review's change),
  along with `demo_output/`, `experiments.jsonl`, and `optuna_study.pkl`.

---

## 5. Prioritized action list

| # | Action | Why | Effort |
|---|---|---|---|
| 1 | Fix `RunMetadata` API drift (`timestamp_utc`, add `collect` classmethod or fix tests) | Unblocks 4 report tests | S |
| 2 | Guard/skip `optimize` tests when `optuna` missing **or** add `optuna` to `dev` | Unblocks 11 tests | S |
| 3 | Validate mask object-counts in `train_cellpose`; raise clear error | Fixes crash, 1 test | S |
| 4 | Wire `tracked_run` into `segment`/`train`/`evaluate`/`optimize` | Makes FAIR/export real | M |
| 5 | Have `optimize` write `optimization.json` | Unblocks report's optimize section | S |
| 6 | Reconcile `select_segmenter` with `recommend_model`; consume `recommended_*` from `project.yaml` | Correct model actually used | M |
| 7 | Keep `mean_f1` naming consistent in new evaluation surfaces | Scientific correctness | S |
| 8 | Surface or remove the ignored `pretrained`/`channels` CellPose args | Honest behavior | S |
| 9 | Implement `MicroSamSegmenter` or drop μSAM from docs | Truth-in-advertising | M/L |
| 10 | Delete demo dead code; unify result-JSON filenames | Cleanup | S |

Items 1–3 turn the suite green and should land first. Items 4–6 are the integration
work that makes the documented pipeline actually hold together end to end.
