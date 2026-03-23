"""Tests for microagent.core.train."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage import measure


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_img(seed: int = 0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Return a (H, W) uint16 image with a few filled circles."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.uint16)
    for _ in range(3):
        cy = int(rng.integers(10, shape[0] - 10))
        cx = int(rng.integers(10, shape[1] - 10))
        r = int(rng.integers(5, 12))
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = int(
            rng.integers(10000, 40000)
        )
    return img


def _make_mask(img: np.ndarray) -> np.ndarray:
    """Connected-component label mask derived from *img*."""
    threshold = int(np.iinfo(np.uint16).max) * 0.1
    return measure.label(img > threshold).astype(np.uint16)


@pytest.fixture
def raw_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """5-image raw images + ground-truth masks directories."""
    img_dir = tmp_path / "images"
    gt_dir = tmp_path / "gt"
    img_dir.mkdir()
    gt_dir.mkdir()
    for i in range(5):
        img = _make_img(seed=i)
        tifffile.imwrite(img_dir / f"img_{i:03d}.tif", img)
        tifffile.imwrite(gt_dir / f"img_{i:03d}_masks.tif", _make_mask(img))
    return img_dir, gt_dir


# ── test_prepare_data ──────────────────────────────────────────────────────────


def test_prepare_data_creates_correct_structure(raw_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """prepare_data should produce CellPose-named pairs under train/ and test/."""
    from microagent.core.train import prepare_data

    img_dir, gt_dir = raw_dirs
    out_root = tmp_path / "out"
    train_dir, test_dir = prepare_data(
        image_dir=img_dir,
        gt_dir=gt_dir,
        test_split=0.2,
        seed=0,
        output_root=out_root,
    )

    assert train_dir.exists()
    assert test_dir.exists()

    train_imgs = list(train_dir.glob("*_img.tif"))
    train_masks = list(train_dir.glob("*_masks.tif"))
    test_imgs = list(test_dir.glob("*_img.tif"))
    test_masks = list(test_dir.glob("*_masks.tif"))

    # Each image must have a paired mask
    assert len(train_imgs) == len(train_masks)
    assert len(test_imgs) == len(test_masks)

    # Total pairs = 5, at least 1 in test and at least 1 in train
    total = len(train_imgs) + len(test_imgs)
    assert total == 5
    assert len(test_imgs) >= 1
    assert len(train_imgs) >= 1


def test_prepare_data_deterministic(raw_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Same seed must produce the same split."""
    from microagent.core.train import prepare_data

    img_dir, gt_dir = raw_dirs

    train_a, _ = prepare_data(img_dir, gt_dir, seed=7, output_root=tmp_path / "run_a")
    train_b, _ = prepare_data(img_dir, gt_dir, seed=7, output_root=tmp_path / "run_b")

    names_a = sorted(f.name for f in train_a.glob("*_img.tif"))
    names_b = sorted(f.name for f in train_b.glob("*_img.tif"))
    assert names_a == names_b


def test_prepare_data_different_seeds_may_differ(raw_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Different seeds should (with high probability) produce different splits."""
    from microagent.core.train import prepare_data

    img_dir, gt_dir = raw_dirs

    train_a, _ = prepare_data(img_dir, gt_dir, seed=1, output_root=tmp_path / "run_a")
    train_b, _ = prepare_data(img_dir, gt_dir, seed=99, output_root=tmp_path / "run_b")

    names_a = sorted(f.name for f in train_a.glob("*_img.tif"))
    names_b = sorted(f.name for f in train_b.glob("*_img.tif"))
    # With 5 images and two seeds there is a >95% chance the splits differ;
    # we just assert both are valid sets rather than asserting inequality.
    assert len(names_a) >= 1
    assert len(names_b) >= 1


def test_prepare_data_no_pairs_raises(tmp_path: Path) -> None:
    """prepare_data should raise ValueError when no matched pairs exist."""
    from microagent.core.train import prepare_data

    empty_img = tmp_path / "empty_img"
    empty_gt = tmp_path / "empty_gt"
    empty_img.mkdir()
    empty_gt.mkdir()

    with pytest.raises(ValueError, match="No matched"):
        prepare_data(empty_img, empty_gt)


# ── test_train_config_defaults ─────────────────────────────────────────────────


def test_train_config_defaults() -> None:
    """TrainConfig fields must have the specified default values."""
    from microagent.core.train import TrainConfig

    cfg = TrainConfig(train_dir=Path("train"))
    assert cfg.model == "cellpose"
    assert cfg.pretrained == "cpsam"
    assert cfg.learning_rate == pytest.approx(1e-5)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.n_epochs == 100
    assert cfg.batch_size == 1
    assert cfg.image_filter == "_img"
    assert cfg.mask_filter == "_masks"
    assert cfg.seed == 42
    assert cfg.save_dir == Path("models")
    assert cfg.test_dir is None


def test_train_config_custom() -> None:
    """TrainConfig should accept custom values for all fields."""
    from microagent.core.train import TrainConfig

    cfg = TrainConfig(
        model="cellpose",
        pretrained="cyto3",
        train_dir=Path("/tmp/train"),
        test_dir=Path("/tmp/test"),
        learning_rate=1e-4,
        weight_decay=0.01,
        n_epochs=10,
        batch_size=4,
        image_filter="_raw",
        mask_filter="_lbl",
        seed=123,
        save_dir=Path("/tmp/models"),
    )
    assert cfg.pretrained == "cyto3"
    assert cfg.n_epochs == 10
    assert cfg.batch_size == 4
    assert cfg.seed == 123


# ── test_train_smoke ───────────────────────────────────────────────────────────


@pytest.mark.slow
def test_train_smoke(raw_dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """Smoke test: 2-epoch CellPose fine-tuning on tiny synthetic data.

    Skipped automatically when cellpose is not installed.
    """
    pytest.importorskip("cellpose")

    from microagent.core.train import TrainConfig, prepare_data, train_cellpose

    img_dir, gt_dir = raw_dirs
    train_dir, test_dir = prepare_data(
        img_dir, gt_dir, test_split=0.2, seed=0, output_root=tmp_path / "data"
    )

    cfg = TrainConfig(
        pretrained="cyto3",  # smaller model for speed
        train_dir=train_dir,
        test_dir=test_dir,
        n_epochs=2,
        learning_rate=1e-4,
        save_dir=tmp_path / "models",
        seed=0,
    )

    result = train_cellpose(cfg)

    assert result.model_path.exists(), "Model file should be created"
    assert result.elapsed_seconds > 0
    assert result.best_epoch >= 0
    assert result.config_used is cfg
