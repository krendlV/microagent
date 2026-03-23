# FAQ

## Installation

### `pip install microagent` fails with a CellPose error

CellPose requires PyTorch. Install PyTorch first, following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/), then install MicroAgent:

```bash
pip install torch torchvision        # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install microagent
```

### StarDist won't install on my platform

StarDist requires a C compiler for its extension. On macOS:

```bash
xcode-select --install
pip install "microagent[stardist]"
```

On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

### `microagent demo` runs but I get 0 cells detected

This usually means CellPose defaulted to GPU mode but no GPU is available. Try:

```bash
microagent segment /path/to/images --model cellpose --diameter 30
```

If that works, your GPU environment may not be configured correctly (see GPU section below).

---

## GPU Troubleshooting

### How do I check if MicroAgent is using my GPU?

```bash
microagent inspect /path/to/images -o qc.json
cat qc.json | python -c "import json,sys; d=json.load(sys.stdin); print(d.get('provenance', {}).get('gpu_name', 'no GPU'))"
```

Or check the provenance in any `segmentation_metadata.json`.

### CUDA out of memory

Reduce batch size or use a lighter model:

```bash
microagent segment /path --model cellpose      # cyto3 uses less VRAM than cpsam
microagent segment /path --model cellpose --diameter 30   # smaller diameter = less memory
```

For very large images, consider tiling (future feature — see [#12](https://github.com/krendlV/microagent/issues)).

### CellPose is slow even with a GPU

Verify PyTorch sees your GPU:

```python
import torch
print(torch.cuda.is_available())    # should be True
print(torch.cuda.get_device_name()) # should show your GPU
```

If False, reinstall PyTorch with the correct CUDA version for your driver.

---

## Segmentation Quality

### My cell counts seem too high / too low

1. Run `microagent inspect` to check image dimensions and intensity range
2. Try adjusting `--diameter` (CellPose): should match average cell diameter in pixels
3. Run `microagent optimize` with a small set of annotated images to find optimal parameters

### Cells at image borders are incomplete

This is expected behavior — CellPose and StarDist both handle borders conservatively. If border cells are important, use `microagent segment` with border-handling options (coming in a future release).

### StarDist gives an error about model weights

StarDist downloads pretrained weights on first use. Ensure you have internet access, or download manually:

```python
from stardist.models import StarDist2D
StarDist2D.from_pretrained("2D_versatile_fluo")
```

---

## File Formats

### What image formats are supported?

TIFF, OME-TIFF, PNG, JPEG. CZI (Zeiss), ND2 (Nikon), and LIF (Leica) are supported with extra packages:

```bash
pip install aicsimageio          # CZI, OME-TIFF
pip install nd2reader            # ND2
```

### Can I use 3D (z-stack) images?

MicroAgent currently operates on 2D images. For z-stacks, extract the best focal plane first (e.g. with Fiji's "Focus Stacking" plugin) or process slice-by-slice. 3D segmentation support is planned.

### My masks aren't matching to ground truth files

MicroAgent matches files by stem, stripping suffixes `_mask`, `_masks`, `_seg`, `_label`. So `cell001_mask.tif` matches `cell001.tif`. If your naming doesn't follow this pattern, rename your files or open an issue.

---

## Licensing

### Can I use MicroAgent in a commercial product?

MicroAgent source code is BSD-3-Clause — yes. However, the `cpsam` model weights are **CC-BY-NC** (non-commercial only). For commercial use, switch to `cyto3` or `cyto2`:

```bash
microagent segment /data/images --model cellpose
```

Then set `pretrained: cyto3` in your `project.yaml` to avoid `cpsam` entirely.

### What are the licenses for each model?

| Model | License | Commercial OK? |
|-------|---------|----------------|
| CellPose cyto2 | BSD-3-Clause | Yes |
| CellPose cyto3 | BSD-3-Clause | Yes |
| CellPose cpsam | **CC-BY-NC** | **No** |
| StarDist 2D_versatile_fluo | BSD-3-Clause | Yes |
| StarDist 2D_versatile_he | BSD-3-Clause | Yes |
| micro-SAM | Apache-2.0 | Yes |

---

## Positioning

### Why not use napari?

[napari](https://napari.org) is an excellent interactive viewer. MicroAgent is complementary — it's a scriptable, headless CLI for batch processing and automation. Use napari for manual review and annotation; use MicroAgent for running pipelines at scale. MicroAgent can generate overlays you view in napari.

### Why not use QuPath?

[QuPath](https://qupath.readthedocs.io) is excellent for H&E pathology analysis. MicroAgent focuses on fluorescence microscopy and scientific Python workflows, and integrates with the Python ecosystem (Optuna, MLflow, CellPose, StarDist).

### Why not use CellProfiler?

[CellProfiler](https://cellprofiler.org) is a mature GUI-based pipeline tool. MicroAgent's advantages: modern deep learning models (CellPose-SAM, StarDist), MCP server for AI assistant integration, FAIR provenance tracking, and a code-first workflow that integrates with Python notebooks and CI pipelines.

---

## MCP / AI Integration

### The MCP server isn't appearing in Claude

1. Check your settings file path (`~/.claude/settings.json`)
2. Verify `microagent` is in your PATH: `which microagent`
3. Test the server starts: `microagent mcp-server` (should run without error)
4. Restart Claude Code after editing settings

### Claude can't find my image files

MCP tools run in the shell environment where the server was started. Ensure:
- Paths are absolute (not relative)
- The server has read access to the image directory
- On Windows, use forward slashes or raw strings in paths
