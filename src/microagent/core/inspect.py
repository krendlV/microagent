"""Image loading, QC, and statistics for microscopy data."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

logger = logging.getLogger(__name__)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import tifffile as _tifffile

    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

try:
    import imageio.v3 as _iio

    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


_TIFF_EXTS = frozenset({".tif", ".tiff"})
_IMAGE_EXTS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})

# ── QC and thumbnail constants ─────────────────────────────────────────────────

THUMBNAIL_MAX_IMAGES = 9
"""Maximum number of images to include in the thumbnail montage."""

THUMBNAIL_COLS = 3
"""Number of columns in the thumbnail montage grid."""

THUMBNAIL_DPI = 300
"""DPI resolution for the saved thumbnail montage."""

NEAR_ZERO_THRESHOLD = 0.01
"""Fraction of dtype max below which an image mean is considered near-zero."""

SATURATION_THRESHOLD = 0.10
"""Fraction of pixels at dtype max above which an image is considered saturated."""


@dataclass
class ChannelStats:
    """Intensity statistics for a single channel across all images."""

    channel: int
    min: float
    max: float
    mean: float
    std: float


@dataclass
class InspectionReport:
    """Results of inspecting a directory of microscopy images.

    Parameters
    ----------
    file_count : int
        Number of successfully loaded image files.
    file_paths : list[str]
        Absolute paths of all discovered image files.
    dimensions : list[list[int]]
        Unique image shapes (as lists) found in the directory.
    dtypes : list[str]
        Unique NumPy dtype strings found across images.
    channel_count : int
        Most common channel count across images.
    intensity_stats : list[ChannelStats]
        Per-channel aggregate intensity statistics.
    issues : list[str]
        QC warnings detected during inspection.
    thumbnail_paths : list[str]
        Paths to generated thumbnail images.
    """

    file_count: int
    file_paths: list[str]
    dimensions: list[list[int]]
    dtypes: list[str]
    channel_count: int
    intensity_stats: list[ChannelStats]
    issues: list[str]
    thumbnail_paths: list[str]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Write the report to *path* as formatted JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    """Load an image file, preferring tifffile for TIFF/OME-TIFF."""
    if path.suffix.lower() in _TIFF_EXTS and _HAS_TIFFFILE:
        return _tifffile.imread(str(path))
    if _HAS_IMAGEIO:
        return _iio.imread(str(path))
    raise RuntimeError(f"Cannot load {path.name}: neither tifffile nor imageio available")


def _to_channels_first(img: np.ndarray) -> np.ndarray:
    """Return image as (C, H, W) regardless of input layout.

    Heuristic: if ndim==2 → add channel dim.
    If ndim==3 and shape[0]<=16 → already (C, H, W).
    If ndim==3 and shape[-1]<=16 → assume (H, W, C) and move axis.
    Otherwise return as-is (caller gets whatever tifffile produced).
    """
    if img.ndim == 2:
        return img[np.newaxis]
    if img.ndim == 3:
        if img.shape[0] <= 16:
            return img
        if img.shape[-1] <= 16:
            return np.moveaxis(img, -1, 0)
    return img


def _dtype_max(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0  # normalised float images


def _generate_thumbnail(images: list[np.ndarray], out_path: Path) -> None:
    """Save a montage PNG of up to THUMBNAIL_MAX_IMAGES images."""
    n = min(len(images), THUMBNAIL_MAX_IMAGES)
    ncols = min(n, THUMBNAIL_COLS)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), squeeze=False)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.axis("off")
        if idx < n:
            img = images[idx]
            ch0 = img[0] if img.ndim >= 3 else img
            mn, mx = float(ch0.min()), float(ch0.max())
            display = (ch0.astype(np.float64) - mn) / (mx - mn) if mx > mn else np.zeros_like(ch0, dtype=float)
            ax.imshow(display, cmap="gray", interpolation="nearest")
            ax.set_title(f"#{idx}", fontsize=8, pad=2)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=THUMBNAIL_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inspect_directory(
    path: Path,
    channels: list[int] | None = None,
) -> InspectionReport:
    """Inspect a directory of microscopy images and return a QC report.

    Parameters
    ----------
    path : Path
        Directory containing image files (.tif, .tiff, .png, .jpg, .jpeg).
    channels : list[int] | None
        Channel indices to compute statistics for. If None, all channels
        are used.

    Returns
    -------
    InspectionReport
        Aggregated statistics and QC warnings for the directory.
    """
    path = Path(path)

    # Discover image files
    file_paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    if not file_paths:
        return InspectionReport(
            file_count=0,
            file_paths=[],
            dimensions=[],
            dtypes=[],
            channel_count=0,
            intensity_stats=[],
            issues=["No image files found in directory"],
            thumbnail_paths=[],
        )

    # Load images
    loaded: list[tuple[Path, np.ndarray]] = []
    issues: list[str] = []

    for fp in file_paths:
        try:
            img = _to_channels_first(_load_image(fp))
            loaded.append((fp, img))
        except Exception as exc:
            logger.debug("Failed to load %s: %s", fp.name, exc)
            issues.append(f"Failed to load {fp.name}: {exc}")

    if not loaded:
        return InspectionReport(
            file_count=0,
            file_paths=[str(fp) for fp in file_paths],
            dimensions=[],
            dtypes=[],
            channel_count=0,
            intensity_stats=[],
            issues=issues or ["All files failed to load"],
            thumbnail_paths=[],
        )

    # Aggregate shapes and dtypes
    all_shapes = [list(img.shape) for _, img in loaded]
    all_dtypes = [str(img.dtype) for _, img in loaded]

    unique_shapes = list({tuple(s): s for s in all_shapes}.values())
    unique_dtypes = list(set(all_dtypes))

    # Most common channel count
    channel_counts = [img.shape[0] for _, img in loaded]
    channel_count = max(set(channel_counts), key=channel_counts.count)

    # Channels to analyse
    chans = channels if channels is not None else list(range(channel_count))

    # Per-channel intensity statistics (across all images)
    intensity_stats: list[ChannelStats] = []
    for ch in chans:
        pixels: list[np.ndarray] = []
        for _, img in loaded:
            if img.ndim >= 3 and ch < img.shape[0]:
                pixels.append(img[ch].ravel().astype(np.float64))
            elif img.ndim == 2 and ch == 0:
                pixels.append(img.ravel().astype(np.float64))
        if pixels:
            flat = np.concatenate(pixels)
            intensity_stats.append(
                ChannelStats(
                    channel=ch,
                    min=float(flat.min()),
                    max=float(flat.max()),
                    mean=float(flat.mean()),
                    std=float(flat.std()),
                )
            )

    # QC checks ----------------------------------------------------------

    # 1. Dtype mismatch
    if len(unique_dtypes) > 1:
        issues.append(f"Dtype mismatch across images: {sorted(unique_dtypes)}")

    # 2. Inconsistent dimensions
    if len(unique_shapes) > 1:
        issues.append(f"Inconsistent image dimensions: {unique_shapes}")

    # 3. Near-zero images (mean < NEAR_ZERO_THRESHOLD of dtype max)
    for fp, img in loaded:
        threshold = _dtype_max(img.dtype) * NEAR_ZERO_THRESHOLD
        if float(img.mean()) < threshold:
            issues.append(f"Near-zero image (mean intensity below 1% of dtype max): {fp.name}")

    # 4. Near-saturated images (>SATURATION_THRESHOLD pixels at dtype max)
    for fp, img in loaded:
        dmax = _dtype_max(img.dtype)
        sat_frac = float((img >= dmax).sum()) / img.size
        if sat_frac > SATURATION_THRESHOLD:
            issues.append(f"Near-saturated image ({sat_frac:.1%} pixels at dtype max): {fp.name}")

    # 5. Single-file directory
    if len(loaded) == 1:
        issues.append("Only 1 image found — may need more data for training")

    # Thumbnail montage
    thumbnail_paths: list[str] = []
    thumb_imgs = [img for _, img in loaded[:THUMBNAIL_MAX_IMAGES]]
    if thumb_imgs:
        thumb_out = path / "microagent_inspection" / "thumbnail_montage.png"
        try:
            _generate_thumbnail(thumb_imgs, thumb_out)
            thumbnail_paths.append(str(thumb_out))
        except Exception as exc:
            logger.warning("Thumbnail generation failed: %s", exc)
            issues.append(f"Thumbnail generation failed: {exc}")

    return InspectionReport(
        file_count=len(loaded),
        file_paths=[str(fp) for fp in file_paths],
        dimensions=unique_shapes,
        dtypes=unique_dtypes,
        channel_count=channel_count,
        intensity_stats=intensity_stats,
        issues=issues,
        thumbnail_paths=thumbnail_paths,
    )
