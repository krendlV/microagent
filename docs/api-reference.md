# API Reference

Complete reference for all CLI commands. Run `microagent <command> --help` for the same information in the terminal.

---

## microagent demo

Run an end-to-end pipeline on synthetic data.

```bash
microagent demo [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output`, `-o` | PATH | `demo_output/` | Output directory |
| `--n-images` | INT | 10 | Number of synthetic images to generate |
| `--no-browser` | FLAG | false | Skip opening report.html in browser |

**What it does:**
1. Generates synthetic fluorescence images with known ground truth
2. Runs `inspect`, `segment`, `evaluate`
3. Creates overlay plots and HTML report
4. Opens `report.html` in your default browser

---

## microagent init

Create a `project.yaml` for your dataset.

```bash
microagent init [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data-dir` | PATH | — | Image directory for auto-detection |
| `--doc`, `-d` | PATH | — | Methods document to extract fields from (Markdown or PDF) |
| `--output`, `-o` | PATH | `project.yaml` | Save path |

Launches an interactive wizard. With `--data-dir`, scans the directory to detect image format and dimensions. With `--doc`, uses the document to pre-fill fields.

---

## microagent inspect

Quality-control check on an image directory.

```bash
microagent inspect <image_dir> [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `image_dir` | Directory containing images |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--channels` | TEXT | all | Comma-separated channel indices to analyze (e.g. `0,1`) |
| `--output`, `-o` | PATH | — | Save JSON report |

**Output:**
- File count, formats detected
- Image dimensions (checks for consistency)
- Per-channel statistics: min, max, mean, std
- QC warnings: mismatched sizes, unusual intensity ranges, load failures

---

## microagent segment

Run cell/nucleus segmentation.

```bash
microagent segment <image_dir> [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `image_dir` | Directory of input images |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output`, `-o` | PATH | `masks/` | Output directory for mask TIFFs |
| `--model`, `-m` | TEXT | `auto` | Model: `auto`, `cellpose`, `stardist`, `micro_sam` |
| `--diameter` | FLOAT | None | Cell diameter in pixels (CellPose; None = auto-detect) |
| `--project`, `-p` | PATH | — | Path to `project.yaml` |

**Outputs:**
- `masks/<stem>_mask.tif` — integer-labeled mask TIFF for each input image
- `masks/segmentation_metadata.json` — model info, parameters, per-image cell counts, provenance

**Model auto-selection** (used when `--model auto` or `-p` is given):
reads modality, structures, and compute config from `project.yaml` to pick the best model.

---

## microagent evaluate

Compute segmentation metrics against ground truth.

```bash
microagent evaluate <pred_dir> <gt_dir> [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `pred_dir` | Directory of predicted mask TIFFs |
| `gt_dir` | Directory of ground-truth mask TIFFs |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--thresholds` | TEXT | `0.5,0.75,0.9` | Comma-separated IoU thresholds |
| `--output`, `-o` | PATH | — | Save metrics JSON |
| `--compare` | PATH | — | Baseline metrics JSON to compare against |

**File matching:** Pairs predictions to ground truth by filename stem, stripping suffixes `_mask`, `_masks`, `_seg`, `_label`.

**Metrics computed per image and as dataset summary:**
- `precision`, `recall`, `f1` — at each IoU threshold
- `tp`, `fp`, `fn` — instance counts
- `mean_true_score` — mean IoU of matched pairs
- `mean_f1` — mean F1 across all thresholds
- `panoptic_quality` — SQ × RQ at IoU 0.5

When `--compare` is given, also outputs per-image deltas and lists of improved/regressed images.

---

## microagent train

Fine-tune CellPose on your labeled data.

```bash
microagent train <image_dir> <gt_dir> [OPTIONS]
```

| Argument | Description |
|----------|-------------|
| `image_dir` | Directory of training images |
| `gt_dir` | Directory of ground-truth mask TIFFs |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--epochs`, `-e` | INT | 100 | Number of training epochs |
| `--lr` | FLOAT | 1e-5 | Learning rate |
| `--weight-decay` | FLOAT | 0.1 | L2 regularization coefficient |
| `--pretrained` | TEXT | `cpsam` | Pretrained model to start from |
| `--test-split` | FLOAT | 0.2 | Fraction of data held out for validation |
| `--output`, `-o` | PATH | `models/` | Directory to save trained model |
| `--seed` | INT | 42 | Random seed |

**Outputs:**
- `models/<name>.pth` — fine-tuned model weights
- Training + test loss curves
- Evaluation metrics on the held-out test set

---

## microagent optimize

Hyperparameter optimization with Optuna.

```bash
microagent optimize <image_dir> <gt_dir> [OPTIONS]
```

Requires `pip install "microagent[tracking]"`.

| Argument | Description |
|----------|-------------|
| `image_dir` | Directory of images |
| `gt_dir` | Directory of ground-truth masks |

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--trials`, `-n` | INT | 20 | Number of Optuna trials |
| `--metric` | TEXT | `f1` | Metric to optimize: `f1`, `precision`, `recall`, `mean_f1`, `pq` |
| `--model`, `-m` | TEXT | `auto` | Backend: `auto`, `cellpose`, `stardist` |
| `--iou` | FLOAT | 0.5 | IoU threshold for metric evaluation |
| `--seed` | INT | 42 | Random seed |
| `--project`, `-p` | PATH | — | Path to `project.yaml` |

**Search space:**
- CellPose: `diameter` (10–80), `flow_threshold` (0.1–1.0), `cellprob_threshold` (-3.0–3.0)
- StarDist: `prob_thresh` (0.1–0.9), `nms_thresh` (0.1–0.7), `scale` (0.5–2.0)

**Output:** Best parameters, baseline score, best score, improvement, all trial records.

---

## microagent report

Generate an HTML report from previous run results.

```bash
microagent report [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output`, `-o` | PATH | `report.html` | Output HTML file path |

Reads results from the current directory (segmentation metadata, evaluation metrics, overlay images). Produces a self-contained HTML file with base64-embedded images.
