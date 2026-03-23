"""Synthetic microscopy data generator for demos and integration tests.

Produces realistic-ish fluorescence microscopy images with matching
ground-truth label masks, without requiring any real biological data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import tifffile as _tifffile

    _HAS_TIFFFILE = True
except ImportError:  # pragma: no cover
    _HAS_TIFFFILE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _draw_objects(
    rng: np.random.Generator,
    H: int,
    W: int,
    n_objects_range: tuple[int, int],
    radius_range: tuple[int, int],
    *,
    allow_overlap: bool = False,
    allow_border: bool = False,
    intensity_variation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw synthetic nuclei-like objects and return (signal_f64, labels_int32).

    Parameters
    ----------
    rng:
        Seeded random generator.
    H, W:
        Image height and width in pixels.
    n_objects_range:
        (min, max) number of objects to attempt.
    radius_range:
        (min, max) radius in pixels.
    allow_overlap:
        If True, new objects may overlap existing ones.
    allow_border:
        If True, object centres may be partially outside the image boundary.
    intensity_variation:
        If True, each object gets a random intensity in [0.4, 1.0].

    Returns
    -------
    signal : np.ndarray
        Float64 array (H, W), values in [0, 1].
    labels : np.ndarray
        Int32 label array (H, W), 0 = background.
    """
    n_objects = int(rng.integers(n_objects_range[0], n_objects_range[1] + 1))
    r_min, r_max = radius_range

    signal = np.zeros((H, W), dtype=np.float64)
    labels = np.zeros((H, W), dtype=np.int32)

    label_id = 1
    for _ in range(n_objects):
        r = int(rng.integers(r_min, r_max + 1))
        margin = max(1, r // 4) if allow_border else r + 3

        cy = int(rng.integers(-r // 2 if allow_border else margin, H - margin + 1))
        cx = int(rng.integers(-r // 2 if allow_border else margin, W - margin + 1))

        yy, xx = np.ogrid[:H, :W]
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

        if not allow_overlap and (labels[circle] > 0).any():
            continue  # skip overlapping objects for clean dataset

        intensity = float(rng.uniform(0.4, 1.0)) if intensity_variation else 0.85
        signal[circle] = intensity
        labels[circle] = label_id
        label_id += 1

    return signal, labels


def _make_image_pair(
    rng: np.random.Generator,
    H: int,
    W: int,
    n_objects_range: tuple[int, int],
    radius_range: tuple[int, int],
    noise_level: float,
    *,
    allow_overlap: bool = False,
    allow_border: bool = False,
    intensity_variation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one synthetic image (2, H, W) uint16 and matching label mask (H, W) uint16.

    Channel 0 — DAPI-like: bright nuclei on dark background with PSF + Poisson noise.
    Channel 1 — dim background channel: low-level uniform noise.

    Returns
    -------
    img_u16 : np.ndarray
        Shape (2, H, W), dtype uint16.
    labels_u16 : np.ndarray
        Shape (H, W), dtype uint16.
    """
    from skimage.filters import gaussian

    signal, labels = _draw_objects(
        rng, H, W, n_objects_range, radius_range,
        allow_overlap=allow_overlap,
        allow_border=allow_border,
        intensity_variation=intensity_variation,
    )

    # Simulate optical PSF with Gaussian blur
    blurred = gaussian(signal, sigma=1.8)

    # Normalise to [0, 0.9] to avoid saturation
    if blurred.max() > 0:
        blurred = blurred / blurred.max() * 0.9

    # Poisson photon noise (simulate photon counting statistics)
    photon_scale = 800
    ch0 = rng.poisson(blurred * photon_scale).astype(np.float64) / photon_scale
    # Low-level background haze
    ch0 += rng.uniform(0, noise_level * 0.08, size=(H, W))
    ch0 = np.clip(ch0, 0.0, 1.0)

    # Dim uniform background channel (e.g. autofluorescence)
    ch1 = rng.uniform(0.0, noise_level * 0.25, size=(H, W))

    img_u16 = (np.stack([ch0, ch1], axis=0) * np.iinfo(np.uint16).max).astype(np.uint16)
    labels_u16 = labels.astype(np.uint16)
    return img_u16, labels_u16


def _save_project_yaml(
    output_dir: Path,
    image_dir: Path,
    gt_dir: Path,
    image_size: tuple[int, int],
    radius_range: tuple[int, int],
) -> Path:
    """Write project.yaml for the synthetic dataset.

    Provides segmentation hints (diameter) so microagent segment can
    call CellPose with the appropriate scale.
    """
    # mean diameter ≈ mean radius * 2
    avg_diameter = int(radius_range[0] + radius_range[1])
    yaml_path = output_dir / "project.yaml"

    # Best-effort PyYAML; fall back to manual string
    try:
        import yaml  # type: ignore[import-untyped]

        data: dict[str, Any] = {
            "name": "synthetic_demo",
            "organism": "synthetic",
            "sample_type": "cell_culture",
            "modality": "fluorescence",
            "structures": ["nuclei"],
            "channels": [
                {"name": "DAPI", "index": 0, "target": "nuclei"},
                {"name": "background", "index": 1, "target": "background"},
            ],
            "image_format": "tiff",
            "bit_depth": 16,
            "has_ground_truth": True,
            "ground_truth_format": "masks",
            "analysis_goal": "count",
            "data_dir": str(image_dir),
            "gt_dir": str(gt_dir),
            "imaging": {
                "segmentation_target": "nuclei",
                "staining": "fluorescence",
                "cell_diameter_pixels": avg_diameter,
                "channels": {"cytoplasm": 0, "nucleus": 0},
            },
            "recommended_model": "cellpose",
            "recommended_params": {
                "diameter": avg_diameter,
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
            },
        }
        yaml_path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    except ImportError:
        # Manual YAML fallback
        yaml_path.write_text(
            f"name: synthetic_demo\n"
            f"organism: synthetic\n"
            f"sample_type: cell_culture\n"
            f"modality: fluorescence\n"
            f"structures:\n  - nuclei\n"
            f"image_format: tiff\n"
            f"bit_depth: 16\n"
            f"has_ground_truth: true\n"
            f"data_dir: {image_dir}\n"
            f"gt_dir: {gt_dir}\n"
            f"imaging:\n"
            f"  segmentation_target: nuclei\n"
            f"  staining: fluorescence\n"
            f"  cell_diameter_pixels: {avg_diameter}\n"
            f"  channels:\n    cytoplasm: 0\n    nucleus: 0\n"
            f"recommended_model: cellpose\n"
            f"recommended_params:\n"
            f"  diameter: {avg_diameter}\n"
            f"  flow_threshold: 0.4\n"
            f"  cellprob_threshold: 0.0\n",
            encoding="utf-8",
        )

    return yaml_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    output_dir: Path,
    n_images: int = 10,
    image_size: tuple[int, int] = (512, 512),
    n_objects_range: tuple[int, int] = (5, 30),
    noise_level: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate a synthetic fluorescence microscopy dataset.

    Creates realistic-ish synthetic images with nucleus-like circular objects,
    optical PSF blur, and Poisson photon noise.  Matching ground-truth label
    masks are saved alongside the images.  A ``project.yaml`` is written to
    *output_dir* so the dataset can be used directly with ``microagent`` CLI
    commands.

    Parameters
    ----------
    output_dir : Path
        Root directory; ``images/`` and ``ground_truth/`` subdirs are created.
    n_images : int
        Number of image/mask pairs to generate.
    image_size : tuple[int, int]
        (height, width) in pixels.
    n_objects_range : tuple[int, int]
        (min, max) number of objects per image.
    noise_level : float
        Controls background noise intensity (0 = noiseless, 1 = very noisy).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    image_dir : Path
        Directory containing the synthetic TIFF images.
    gt_dir : Path
        Directory containing the ground-truth label mask TIFFs.
    """
    if not _HAS_TIFFFILE:
        raise ImportError("tifffile is required: pip install tifffile")

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    gt_dir = output_dir / "ground_truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    H, W = image_size
    # Scale radius range to image size; objects should be clearly visible
    r_min = max(8, min(H, W) // 20)
    r_max = max(r_min + 4, min(H, W) // 9)
    radius_range = (r_min, r_max)

    rng = np.random.default_rng(seed)

    for i in range(n_images):
        img, labels = _make_image_pair(
            rng, H, W, n_objects_range, radius_range, noise_level
        )
        _tifffile.imwrite(str(image_dir / f"image_{i:03d}.tif"), img)
        _tifffile.imwrite(str(gt_dir / f"image_{i:03d}_labels.tif"), labels)

    _save_project_yaml(output_dir, image_dir, gt_dir, image_size, radius_range)
    return image_dir, gt_dir


def generate_challenging_dataset(
    output_dir: Path,
    n_images: int = 10,
    image_size: tuple[int, int] = (512, 512),
    n_objects_range: tuple[int, int] = (10, 40),
    noise_level: float = 0.15,
    seed: int = 99,
) -> tuple[Path, Path]:
    """Generate a challenging synthetic dataset with touching/border objects.

    Like :func:`generate_synthetic_dataset` but with:
    - Touching and overlapping objects
    - High size variation (small to large)
    - Per-object intensity variation
    - Partial objects at image borders

    Intended for testing that metrics are not trivially perfect.

    Parameters
    ----------
    output_dir : Path
        Root directory; ``images/`` and ``ground_truth/`` subdirs are created.
    n_images : int
        Number of image/mask pairs to generate.
    image_size : tuple[int, int]
        (height, width) in pixels.
    n_objects_range : tuple[int, int]
        (min, max) number of objects per image.
    noise_level : float
        Background noise intensity.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    image_dir : Path
        Directory containing the synthetic TIFF images.
    gt_dir : Path
        Directory containing the ground-truth label mask TIFFs.
    """
    if not _HAS_TIFFFILE:
        raise ImportError("tifffile is required: pip install tifffile")

    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    gt_dir = output_dir / "ground_truth"
    image_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    H, W = image_size
    r_min = max(5, min(H, W) // 30)
    r_max = max(r_min + 8, min(H, W) // 5)
    radius_range = (r_min, r_max)

    rng = np.random.default_rng(seed)

    for i in range(n_images):
        img, labels = _make_image_pair(
            rng, H, W, n_objects_range, radius_range, noise_level,
            allow_overlap=True,
            allow_border=True,
            intensity_variation=True,
        )
        _tifffile.imwrite(str(image_dir / f"image_{i:03d}.tif"), img)
        _tifffile.imwrite(str(gt_dir / f"image_{i:03d}_labels.tif"), labels)

    _save_project_yaml(output_dir, image_dir, gt_dir, image_size, radius_range)
    return image_dir, gt_dir
