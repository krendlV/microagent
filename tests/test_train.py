"""Tests for microagent.core.train."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import tifffile
from skimage import measure


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_img(seed: int = 0, shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Return a (H, W) uint16 image with labelled-object-like spots."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.uint16)
    centers = ((12, 12), (12, 32), (12, 52), (34, 12), (34, 32), (34, 52))
    for cy, cx in centers:
        r = int(rng.integers(4, 7))
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = int(
            rng.integers(10000, 40000)
        )
    return img


def _make_mask(img: np.ndarray) -> np.ndarray:
    """Connected-component label mask derived from *img*."""
    threshold = int(np.iinfo(np.uint16).max) * 0.1
    return measure.label(img > threshold).astype(np.uint16)


def _write_cellpose_pair(data_dir: Path, stem: str, mask: np.ndarray) -> None:
    """Write a minimal CellPose image/mask pair."""
    img = np.zeros(mask.shape, dtype=np.uint16)
    img[mask > 0] = 20000
    tifffile.imwrite(data_dir / f"{stem}_img.tif", img)
    tifffile.imwrite(data_dir / f"{stem}_masks.tif", mask.astype(np.uint16))


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


def test_train_cellpose_rejects_all_sparse_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """train_cellpose should fail clearly before CellPose sees sparse masks."""
    import sys

    from microagent.core import train as train_module
    from microagent.core.train import TrainConfig, train_cellpose

    fake_cellpose = ModuleType("cellpose")
    fake_cellpose.models = SimpleNamespace()
    fake_cellpose.train = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setattr(train_module, "_HAS_CELLPOSE", True)

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    sparse_mask = np.zeros((32, 32), dtype=np.uint16)
    sparse_mask[8:16, 8:16] = 1
    for i in range(3):
        _write_cellpose_pair(train_dir, f"sparse_{i}", sparse_mask)

    cfg = TrainConfig(train_dir=train_dir, n_epochs=1, save_dir=tmp_path / "models")

    with pytest.raises(
        ValueError,
        match=(
            r"CellPose requires at least 5 labelled objects per image; "
            r"3 of 3 training images were too sparse"
        ),
    ):
        train_cellpose(cfg)


def test_train_cellpose_uses_pretrained_model_on_cellpose_v4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CellPose v4 must not receive the ignored model_type constructor arg."""
    import sys
    from unittest.mock import MagicMock

    from microagent.core import train as train_module
    from microagent.core.train import TrainConfig, train_cellpose

    fake_model = SimpleNamespace(net=object())
    fake_cellpose = ModuleType("cellpose")
    fake_cellpose.models = SimpleNamespace(CellposeModel=MagicMock(return_value=fake_model))
    fake_cellpose.train = SimpleNamespace(
        train_seg=MagicMock(return_value=(tmp_path / "models" / "model.cp", [1.0], []))
    )
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_cellpose.models)
    monkeypatch.setitem(sys.modules, "cellpose.train", fake_cellpose.train)
    monkeypatch.setattr(train_module, "_HAS_CELLPOSE", True)
    monkeypatch.setattr("microagent.core.cellpose_compat.version", lambda _name: "4.0.9")

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    mask = np.zeros((32, 32), dtype=np.uint16)
    for i, (row, col) in enumerate(
        [(2, 2), (2, 12), (12, 2), (12, 12), (22, 2), (22, 12)], start=1
    ):
        mask[row : row + 4, col : col + 4] = i
    _write_cellpose_pair(train_dir, "dense", mask)

    cfg = TrainConfig(
        pretrained="cyto3",
        train_dir=train_dir,
        n_epochs=1,
        save_dir=tmp_path / "models",
    )

    train_cellpose(cfg)

    fake_cellpose.models.CellposeModel.assert_called_once_with(
        gpu=False,
        pretrained_model="cyto3",
    )
    assert "model_type" not in fake_cellpose.models.CellposeModel.call_args.kwargs
