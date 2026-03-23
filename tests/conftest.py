"""Shared pytest fixtures for microagent tests."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from skimage import measure


def _make_nucleus_image(
    seed: int,
    shape: tuple[int, ...] = (2, 256, 256),
    dtype: type = np.uint16,
) -> np.ndarray:
    """Synthetic (C, H, W) image with filled circles simulating nuclei."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=dtype)
    max_val = int(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1

    n_circles = int(rng.integers(3, 7))
    for _ in range(n_circles):
        cy = int(rng.integers(20, shape[-2] - 20))
        cx = int(rng.integers(20, shape[-1] - 20))
        r = int(rng.integers(10, 30))
        intensity = int(rng.integers(int(max_val * 0.3), int(max_val * 0.8)))

        yy, xx = np.ogrid[: shape[-2], : shape[-1]]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        for c in range(shape[0]):
            img[c][mask] = intensity

    return img


@pytest.fixture
def tmp_image_dir(tmp_path):
    """5 synthetic 256×256 uint16 2-channel TIFFs with nucleus-like circles."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        img = _make_nucleus_image(seed=i)
        tifffile.imwrite(img_dir / f"image_{i:03d}.tif", img)
    return img_dir


@pytest.fixture
def tmp_gt_dir(tmp_path, tmp_image_dir):
    """Matching ground-truth label masks (connected-component labels of channel 0)."""
    gt_dir = tmp_path / "ground_truth"
    gt_dir.mkdir()
    threshold = int(np.iinfo(np.uint16).max) * 0.1
    for i in range(5):
        img = tifffile.imread(tmp_image_dir / f"image_{i:03d}.tif")
        binary = img[0] > threshold
        labels = measure.label(binary).astype(np.uint16)
        tifffile.imwrite(gt_dir / f"image_{i:03d}_labels.tif", labels)
    return gt_dir


@pytest.fixture
def tmp_image_dir_bad(tmp_path):
    """Directory with dtype mismatch and an all-zero image for QC testing."""
    bad_dir = tmp_path / "bad_images"
    bad_dir.mkdir()

    # Normal uint16 image
    tifffile.imwrite(bad_dir / "image_000.tif", _make_nucleus_image(seed=100, dtype=np.uint16))

    # uint8 image — intentional dtype mismatch
    tifffile.imwrite(
        bad_dir / "image_001.tif",
        _make_nucleus_image(seed=101, shape=(2, 256, 256), dtype=np.uint8),
    )

    # All-zero image — triggers near-zero QC warning
    tifffile.imwrite(
        bad_dir / "image_002.tif",
        np.zeros((2, 256, 256), dtype=np.uint16),
    )

    return bad_dir
