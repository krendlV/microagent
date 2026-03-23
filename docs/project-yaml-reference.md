# project.yaml Reference

`project.yaml` is the central configuration file for a MicroAgent project. It stores dataset metadata, guides automatic model selection, and records recommended segmentation parameters.

Create one with:

```bash
microagent init --data-dir /path/to/images
```

---

## Complete Schema

```yaml
# ── Identity ──────────────────────────────────────────────
name: "HeLa Nuclear Segmentation"   # Human-readable project name

# ── Biology ───────────────────────────────────────────────
organism: "human"                   # See valid values below
sample_type: "cell_culture"         # See valid values below
modality: "fluorescence"            # See valid values below
structures:                         # List of structures to segment
  - "nuclei"
  - "whole_cells"

# ── Channels ──────────────────────────────────────────────
channels:
  - name: "DAPI"                    # Channel display name
    index: 0                        # 0-based channel index in image
    target: "nuclei"                # What this channel shows

# ── Image Format ──────────────────────────────────────────
image_format: "tiff"                # See valid values below
typical_dimensions: [1024, 1024]    # [height, width] in pixels; null if variable
bit_depth: 16                       # 8 or 16

# ── Ground Truth ──────────────────────────────────────────
has_ground_truth: true
ground_truth_format: "masks"        # "masks", "points", or "polygons"; null if none

# ── Analysis Goal ─────────────────────────────────────────
analysis_goal: "segment"            # See valid values below

# ── Data Paths ────────────────────────────────────────────
data_dir: "/data/hela_images"
gt_dir: "/data/hela_gt"             # Optional; null if no ground truth

# ── Compute ───────────────────────────────────────────────
compute:
  gpu_model: "NVIDIA RTX 3080"      # GPU name; null if CPU-only
  vram_gb: 10.0                     # Available VRAM in GB; null if CPU
  ram_gb: 32.0                      # System RAM in GB

# ── Model Selection (auto-populated by init) ──────────────
recommended_model: "2D_versatile_fluo"
recommended_params:
  prob_thresh: 0.479
  nms_thresh: 0.3
```

---

## Field Reference

### `organism`

Free-text string. Common values:

`human` · `mouse` · `rat` · `zebrafish` · `drosophila` · `c_elegans` · `yeast` · `plant`

### `sample_type`

| Value | Description |
|-------|-------------|
| `cell_culture` | Cells grown in a dish |
| `tissue_section` | Fixed tissue slice |
| `whole_mount` | Intact 3D tissue sample |
| `organoid` | 3D self-organized culture |

### `modality`

| Value | Typical Use |
|-------|-------------|
| `fluorescence` | Labeled cells, widefield or epifluorescence |
| `confocal` | Confocal fluorescence, z-stacks |
| `brightfield` | Unstained transmitted light |
| `phase_contrast` | Phase contrast transmitted light |
| `H&E` | Hematoxylin & eosin stained tissue |
| `EM` | Electron microscopy |

### `structures`

List of biological structures to segment. Examples:

`nuclei` · `whole_cells` · `mitochondria` · `microtubules` · `organoids` · `bacteria`

### `image_format`

| Value | Extension | Notes |
|-------|-----------|-------|
| `tiff` | `.tif`, `.tiff` | Recommended |
| `ome-tiff` | `.ome.tif` | With metadata |
| `png` | `.png` | 8-bit only |
| `jpg` | `.jpg`, `.jpeg` | Lossy, avoid for quantitative work |
| `czi` | `.czi` | Zeiss format (requires `aicsimageio`) |
| `nd2` | `.nd2` | Nikon format (requires `nd2reader`) |
| `lif` | `.lif` | Leica format |

### `analysis_goal`

| Value | Description |
|-------|-------------|
| `count` | Count objects only |
| `segment` | Instance segmentation (masks) |
| `track` | Track objects across time |
| `measure` | Measure morphological properties |
| `classify` | Classify cell types |

### `ground_truth_format`

| Value | Description |
|-------|-------------|
| `masks` | Integer-labeled TIFF masks (recommended) |
| `points` | CSV/JSON point annotations |
| `polygons` | GeoJSON or COCO-format polygons |

---

## Model Selection Decision Tree

```
What is your modality?
│
├── EM ──────────────────────────────► StarDist 2D_versatile_fluo
│
├── H&E ─────────────────────────────► StarDist 2D_versatile_he
│
├── Fluorescence, structures = nuclei ► StarDist 2D_versatile_fluo
│
├── Phase contrast / Brightfield ────► CellPose cyto2
│
├── Confocal ────────────────────────► CellPose cpsam
│
├── Fluorescence (whole cells) ──────► CellPose cyto3
│
└── Default ─────────────────────────► CellPose cpsam
         │
         └── VRAM < 4 GB ────────────► CellPose cyto2 (CPU mode)
```

---

## Using Multiple Channels

If your images have multiple channels (e.g. DAPI + phalloidin), specify each:

```yaml
channels:
  - name: "DAPI"
    index: 0
    target: "nuclei"
  - name: "Phalloidin"
    index: 1
    target: "actin"
```

For CellPose multichannel segmentation, `recommended_params` will include `channels: [1, 0]` (cytoplasm channel, nucleus channel).

---

## Minimal Valid project.yaml

Only a few fields are truly required:

```yaml
organism: human
modality: fluorescence
structures:
  - nuclei
data_dir: /data/images
```

MicroAgent will fill in defaults for everything else.

---

## Updating Recommended Parameters After Optimization

After running `microagent optimize`, copy the best parameters into `project.yaml`:

```yaml
recommended_model: cellpose
recommended_params:
  diameter: 28.5
  flow_threshold: 0.35
  cellprob_threshold: -0.2
```

These will be used automatically on the next `microagent segment` run with `-p project.yaml`.
