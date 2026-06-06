# MicroAgent — Manual Testing Guide

A step-by-step script for manually exercising MicroAgent end to end. Each section has a
**command**, **what you should see**, and a **pass/fail check**. Times assume a machine
with a GPU; CPU-only is slower but works.

> All commands assume you are in the repo root and using `uv`. If you've `pip install`-ed
> MicroAgent into an active environment, drop the `uv run` prefix.

---

## 0. Setup (once)

```bash
uv sync --extra dev          # core + test tooling (CellPose included)
# Optional backends / features:
uv pip install "mcp>=1.18.0" # MCP server
uv pip install optuna        # hyperparameter optimization
uv pip install stardist      # StarDist backend (nuclei / H&E)
# micro-SAM is conda-only:    conda install -c conda-forge micro_sam
```

**Check:** `uv run microagent --help` prints the command list
(`inspect segment evaluate train optimize report export init demo`).

---

## 1. Automated tests first (your safety net)

There **is** already an automated test suite — run it before manual testing.

```bash
uv run pytest -q                       # unit + most integration: ~2-3 min
uv run pytest -m integration -q        # end-to-end pipeline tests
uv run pytest --cov=microagent         # with coverage
uv run ruff check src/                 # lint
uv run ruff format --check src/        # format
```

**Pass:** `232 passed, 3 skipped` (skips = optional deps you didn't install). `ruff`
reports "All checks passed". If anything **fails** (not skips), stop and investigate
before manual testing.

---

## 2. The 60-second smoke test (no data needed)

This is the fastest "is it alive" check. It generates synthetic data and runs the whole
pipeline.

```bash
cd /tmp && rm -rf ma_demo && mkdir ma_demo && cd ma_demo
uv run --project <REPO> microagent demo --no-browser
```

(Replace `<REPO>` with the absolute repo path, or run from an installed env without
`--project`.)

**What you should see:** a sequence of green ✓ lines — 10 images → inspect → segment →
overlays → evaluate → plots → report — ending in a "Demo complete!" panel with
`F1@0.5 = 1.000`.

**Check the outputs exist:**
```bash
ls demo_output/                    # images masks plots overlays report.html *.json
open demo_output/report.html       # (macOS) or xdg-open / just open in a browser
```
**Pass:** `report.html` opens and shows Data Summary, Segmentation Results, Metrics
Dashboard, and Reproducibility sections with images and numbers (not blanks).

---

## 3. Real-data pipeline (using the sample data)

The repo ships three H&E-style RGB tiles in `sample_data/` (`1.tif`, `2.tif`, `3.tif`,
2048×2048). Copy a couple somewhere writable.

```bash
cd /tmp && rm -rf ma_real && mkdir -p ma_real/imgs && cd ma_real
cp <REPO>/sample_data/1.tif <REPO>/sample_data/2.tif imgs/
```

### 3a. Inspect (QC)
```bash
uv run --project <REPO> microagent inspect imgs -o inspection.json
```
**Pass:** prints a file table + a "Summary" panel (file count, channels, dtype,
intensity stats). `inspection.json` is created.
A `microagent_inspection/` thumbnail folder is written next to `inspection.json`, not
inside the input directory.

### 3b. Segment (auto model selection)
```bash
uv run --project <REPO> microagent segment imgs -o masks
```
**Pass:** a per-image table ("Cells found", "Time"), `masks/*_mask.tif` files, and
`masks/segmentation.json`. A `run … logged → experiments.jsonl` line appears.
> These are H&E tiles. With only CellPose installed, the auto-selected fallback may find
> few/zero cells (CellPose is a poor fit for H&E). Install `stardist` to segment H&E
> properly — see 3d.

### 3c. Report
```bash
uv run --project <REPO> microagent report          # auto-detects the JSON files
```
**Pass:** "Report Generated" panel listing the sections found; `report.html` exists.

### 3d. (If StarDist installed) H&E via project recommendation
```bash
cat > he.yaml <<'EOF'
name: he_demo
modality: H&E
structures: [nuclei]
recommended_model: stardist
recommended_params: {model_name: 2D_versatile_he, prob_thresh: 0.5, nms_thresh: 0.4}
imaging: {staining: he}
EOF
uv run --project <REPO> microagent segment imgs --project he.yaml -o masks_he
uv run --project <REPO> python -c "import json;d=json.load(open('masks_he/segmentation.json'));print(d['model_info'])"
```
**Pass:** `model_info` shows `backend: stardist`, `model_name: 2D_versatile_he`, and the
cell counts are now non-trivial. Without StarDist installed, `backend` will be `cellpose`
(graceful fallback) — also acceptable.

---

## 4. Evaluation (needs ground-truth masks)

Use the synthetic demo, which produces ground truth:

```bash
cd /tmp/ma_demo
uv run --project <REPO> microagent evaluate demo_output/masks demo_output/ground_truth
```
**Pass:** a per-image metrics table with F1 / Mean F1 / PQ columns and a SUMMARY row,
plus a "Worst Images" callout. Values should be high (synthetic data is easy).

**Run-vs-run comparison:** save one result, then compare:
```bash
uv run --project <REPO> microagent evaluate demo_output/masks demo_output/ground_truth -o run_a.json
uv run --project <REPO> microagent evaluate demo_output/masks demo_output/ground_truth --compare run_a.json
```
**Pass:** a "Comparison vs Baseline" panel with ↑/↓/→ deltas (all → since identical).

---

## 5. Hyperparameter optimization (needs optuna + GT)

```bash
cd /tmp/ma_demo
uv run --project <REPO> microagent optimize demo_output/images demo_output/ground_truth \
    --model cellpose --trials 5 --metric f1
```
**Pass:** a live trials table, a "Best Hyperparameters" panel (best vs baseline +
improvement), `optimization.json` written, and a `run … logged` line. If optuna isn't
installed you'll get a clear "optuna is required" message — install it and retry.

Then regenerate the report to see the optimization section picked up:
```bash
uv run --project <REPO> microagent report
grep -c "Optimization Summary" report.html     # should print 1
```

---

## 6. Training (CellPose fine-tuning) — optional, slower

```bash
cd /tmp/ma_demo
uv run --project <REPO> microagent train demo_output/images demo_output/ground_truth \
    --epochs 2 -o models
```
**Pass:** training progresses and a model file appears under `models/`. With very few
labelled objects you should get a **clear `ValueError`** about needing ≥5 objects per
image — that's the intended guardrail, not a crash. (The synthetic demo has enough.)

---

## 7. Reproducibility / FAIR export

After any tracked run (§3b/§5 created `experiments.jsonl`):

```bash
cd /tmp/ma_real      # or wherever experiments.jsonl was written
# grab the most recent run id:
RUNID=$(uv run --project <REPO> python -c "import json;print(json.loads(open('experiments.jsonl').readlines()[-1])['run_id'])")
echo "run id: $RUNID"
uv run --project <REPO> microagent export --run "$RUNID" --format bundle -o export
```
**Pass:** "Reproducibility Bundle" panel; `export/<id>_reproducibility.zip` exists and
contains `Dockerfile`, `requirements.txt`, `run_metadata.json`, `README.md`,
`experiments.jsonl`.

```bash
# Standalone Dockerfile from the current env (no run id needed):
uv run --project <REPO> microagent export --dockerfile -o export
```
**Pass:** `export/Dockerfile` is written.

**Disable tracking** for a one-off:
```bash
uv run --project <REPO> microagent --no-track segment imgs -o masks2
```
**Pass:** no `run … logged` line; `experiments.jsonl` not appended.

---

## 8. MCP server (for LLM clients)

Confirm the server registers tools:
```bash
uv run --project <REPO> python -c "
import asyncio; from microagent import mcp_server
print([t.name for t in asyncio.run(mcp_server.mcp.list_tools())])"
```
**Pass:** prints 8 tools: `inspect_data, segment, evaluate, train, optimize,
generate_report, get_project_info, create_project`.

Call a tool directly (no LLM needed):
```bash
cd /tmp/ma_demo
uv run --project <REPO> python -c "
from microagent import mcp_server
print(mcp_server.inspect_data('demo_output/images')['status'])"
```
**Pass:** prints `success`.

Run the actual stdio server (Ctrl-C to stop):
```bash
uv run --project <REPO> microagent-mcp-server
```
**Pass:** prints "Starting MicroAgent MCP server" and waits on stdin.

### Wire it into Claude Code
Add to `~/.claude/settings.json` (or `.mcp.json`), pointing at your checkout:
```json
{
  "mcpServers": {
    "microagent": {
      "command": "uv",
      "args": ["run", "--project", "/ABS/PATH/TO/microagent", "microagent-mcp-server"]
    }
  }
}
```
Restart Claude Code, then ask: *"Use microagent to inspect the images in <dir> and
segment them."* **Pass:** Claude calls the `inspect_data` / `segment` tools and reports
results.

---

## 9. The `init` interview (interactive)

```bash
cd /tmp/ma_real
uv run --project <REPO> microagent init --data-dir imgs
```
**Pass:** an interactive Q&A (organism, modality, structures, …) ends by writing
`project.yaml` with a `recommended_model` / `recommended_params` block. Inspect it:
```bash
cat project.yaml
```
Then `segment imgs --project project.yaml` should use the recommended model.

---

## 10. Known rough edges (don't file these as bugs)

- H&E tiles segmented with the **CellPose fallback** may return ~0 cells — install
  `stardist` for H&E/nuclei.
- MCP `segment` does **not** take a project file and MCP runs are **not** logged to
  `experiments.jsonl` (CLI runs are).
- `pip install microagent` is **not** live yet — use the source install.

---

## Quick reference — full happy path in one block

```bash
cd /tmp && rm -rf ma && mkdir ma && cd ma
R=/ABS/PATH/TO/microagent
uv run --project $R microagent demo --no-browser            # synthetic smoke test
mkdir imgs && cp $R/sample_data/1.tif imgs/
uv run --project $R microagent inspect imgs -o inspection.json
uv run --project $R microagent segment imgs -o masks
uv run --project $R microagent report
RUNID=$(uv run --project $R python -c "import json;print(json.loads(open('experiments.jsonl').readlines()[-1])['run_id'])")
uv run --project $R microagent export --run "$RUNID" --format bundle -o export
ls report.html export/*.zip
```
