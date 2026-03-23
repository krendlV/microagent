# User Guide

## Working with project.yaml

`project.yaml` is the central configuration file for a MicroAgent project. It drives automatic model selection, default parameters, and report metadata.

Create one interactively:

```bash
microagent init --data-dir /path/to/images
```

Or from a methods document:

```bash
microagent init --data-dir /path/to/images --doc paper_methods.md
```

Example `project.yaml`:

```yaml
name: HeLa Nuclear Segmentation
organism: human
sample_type: cell_culture
modality: fluorescence
structures:
  - nuclei
channels:
  - name: DAPI
    index: 0
    target: nuclei
image_format: tiff
typical_dimensions: [1024, 1024]
bit_depth: 16
has_ground_truth: true
ground_truth_format: masks
analysis_goal: segment
data_dir: /data/hela_images
gt_dir: /data/hela_gt
recommended_model: 2D_versatile_fluo
recommended_params:
  prob_thresh: 0.479
  nms_thresh: 0.3
```

Pass it to any command with `-p`:

```bash
microagent segment /data/hela_images -p project.yaml
```

---

## Model Selection Guide

MicroAgent auto-selects a model when you run `microagent init`. Here's the logic:

| Sample/Modality | Recommended Model | Rationale |
|----------------|-------------------|-----------|
| Any EM | StarDist `2D_versatile_fluo` | Best for round nuclei in EM |
| H&E tissue | StarDist `2D_versatile_he` | Trained on stained tissue |
| Nuclei only (fluorescence) | StarDist `2D_versatile_fluo` | Optimized for round nuclei |
| Phase contrast / brightfield | CellPose `cyto2` | Works without fluorescence |
| Confocal fluorescence | CellPose `cpsam` | Handles complex morphology |
| General fluorescence | CellPose `cyto3` | Good default for whole cells |
| Low VRAM (<4 GB) | CellPose `cyto2` | CPU-compatible |
| Default | CellPose `cpsam` | Most capable general model |

Override the auto-selected model at any time:

```bash
microagent segment /data/images --model stardist
microagent segment /data/images --model cellpose --diameter 25
```

**When to use each model:**

- **cyto2** — reliable workhorse, fastest, works on CPU, best for phase contrast
- **cyto3** — better than cyto2 for fluorescence, good for multichannel images
- **cpsam** — highest quality but requires GPU and is CC-BY-NC (non-commercial)
- **StarDist** — best for circular/elliptical nuclei, fast, excellent for H&E

---

## Fine-Tuning Guide

Fine-tuning improves segmentation on images that differ significantly from the pretrained data (unusual cell morphology, novel staining, low SNR).

### Preparing training data

You need paired image + mask files:

```
train_images/
├── cell001.tif
├── cell002.tif
└── ...
train_masks/
├── cell001.tif    # integer-labeled mask
├── cell002.tif
└── ...
```

Masks must use integer labels (0 = background, 1/2/3/... = individual cells). Tools like Fiji/ImageJ, napari, or QuPath can export this format.

### Running fine-tuning

```bash
microagent train train_images/ train_masks/
```

With options:

```bash
microagent train train_images/ train_masks/ \
  --epochs 200 \
  --lr 5e-6 \
  --pretrained cpsam \
  --test-split 0.2 \
  --output models/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--lr` | 1e-5 | Learning rate |
| `--weight-decay` | 0.1 | L2 regularization |
| `--pretrained` | `cpsam` | Starting weights |
| `--test-split` | 0.2 | Fraction held out for validation |
| `--output` | `models/` | Where to save the trained model |
| `--seed` | 42 | Random seed for reproducibility |

### Evaluating the fine-tuned model

After training, the fine-tuned model is automatically evaluated on the held-out test set. To compare against the pretrained model:

```bash
# Evaluate fine-tuned
microagent evaluate masks_finetuned/ train_masks/ -o metrics_finetuned.json

# Evaluate pretrained
microagent evaluate masks_pretrained/ train_masks/ -o metrics_pretrained.json

# Compare
microagent evaluate masks_finetuned/ train_masks/ --compare metrics_pretrained.json
```

---

## Hyperparameter Optimization Guide

HPO uses [Optuna](https://optuna.org) to search for the best segmentation parameters for your data.

```bash
pip install "microagent[tracking]"   # requires tracking extra
microagent optimize train_images/ train_masks/
```

With options:

```bash
microagent optimize train_images/ train_masks/ \
  --trials 50 \
  --metric f1 \
  --model cellpose \
  --iou 0.5 \
  --seed 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `--trials` | 20 | Number of Optuna trials |
| `--metric` | `f1` | Metric to optimize (`f1`, `map`, `pq`, `precision`, `recall`) |
| `--model` | `auto` | Backend to tune (`cellpose`, `stardist`) |
| `--iou` | 0.5 | IoU threshold for metric computation |

**Parameters searched:**

- CellPose: `diameter`, `flow_threshold`, `cellprob_threshold`
- StarDist: `prob_thresh`, `nms_thresh`, `scale`

Results show the best parameters and improvement over the baseline:

```
Best parameters: diameter=28.5, flow_threshold=0.35, cellprob_threshold=-0.2
F1 improved: 0.72 → 0.81 (+0.09)
```

Apply the best parameters by updating `project.yaml` `recommended_params`.

---

## Generating Reports

```bash
microagent report
```

By default reads results from the current directory. Outputs `report.html` — a self-contained file with no external dependencies that can be shared or archived.

The report includes:
- Dataset summary (image count, mean cell count, QC flags)
- Metric tables (F1, precision, recall, mAP, PQ at all IoU thresholds)
- Per-image overlay composites (original + predicted mask)
- Charts: F1 vs. IoU threshold, cell count distribution
- Full provenance metadata (library versions, data hash, git commit)

---

## Batch Processing Tips

### Process multiple experiments

```bash
for dir in /data/experiment*/; do
  microagent segment "$dir" -o "${dir}masks/" -p "$dir/project.yaml"
done
```

### Parallel processing with xargs

```bash
ls /data/experiments/ | xargs -P 4 -I{} microagent segment /data/experiments/{} -o /data/masks/{}
```

### Monitoring progress

All commands emit structured output via `rich`. Progress bars, cell counts, and timing appear automatically. Use `--output` flags to save JSON results for scripting.

### Using experiments.jsonl for tracking

Every run appends a record to `experiments.jsonl`:

```bash
# View recent runs
python -c "
import json
with open('experiments.jsonl') as f:
    for line in f:
        r = json.loads(line)
        print(r['run_id'], r['results'].get('map', 'N/A'))
"
```
