"""CellPose fine-tuning engine for microscopy images."""

from __future__ import annotations

import inspect
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tifffile as _tifffile

    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

try:
    from cellpose import models as _cp_models
    from cellpose import train as _cp_train

    _HAS_CELLPOSE = True
except ImportError:
    _HAS_CELLPOSE = False


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Configuration for fine-tuning a segmentation model.

    Parameters
    ----------
    model : str
        Model backend to train (currently only "cellpose" is supported).
    pretrained : str
        Pretrained model name to start from (e.g. "cpsam", "cyto3").
    train_dir : Path
        Directory containing training images and masks.
    test_dir : Path | None
        Directory containing test images and masks. If None, a split is
        created automatically by ``prepare_data``.
    learning_rate : float
        SGD/Adam learning rate.
    weight_decay : float
        Weight decay regularisation coefficient.
    n_epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size (use 1 for memory-constrained environments).
    image_filter : str
        Suffix that identifies raw image files in the training directory.
    mask_filter : str
        Suffix that identifies mask files in the training directory.
    seed : int
        Random seed for reproducibility.
    save_dir : Path
        Directory where the trained model will be written.
    """

    model: str = "cellpose"
    pretrained: str = "cpsam"
    train_dir: Path = field(default_factory=lambda: Path("train"))
    test_dir: Optional[Path] = None
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    n_epochs: int = 100
    batch_size: int = 1
    image_filter: str = "_img"
    mask_filter: str = "_masks"
    seed: int = 42
    save_dir: Path = field(default_factory=lambda: Path("models"))


@dataclass
class TrainResult:
    """Output of a completed training run.

    Parameters
    ----------
    model_path : Path
        Path to the saved model weights file.
    train_losses : list[float]
        Per-epoch training loss values.
    test_losses : list[float]
        Per-epoch test loss values (empty if no test set was evaluated).
    best_epoch : int
        Epoch index (0-based) with the lowest test loss.
    elapsed_seconds : float
        Wall-clock time for the full training run.
    config_used : TrainConfig
        The configuration that produced this result.
    """

    model_path: Path
    train_losses: list[float]
    test_losses: list[float]
    best_epoch: int
    elapsed_seconds: float
    config_used: TrainConfig


# ── Data preparation ───────────────────────────────────────────────────────────


def _save_tiff(array: np.ndarray, path: Path) -> None:
    """Write *array* to *path* as a TIFF, trying tifffile then imageio."""
    if _HAS_TIFFFILE:
        import tifffile

        tifffile.imwrite(str(path), array)
        return
    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), array)
        return
    except ImportError:
        pass
    raise RuntimeError("Cannot write TIFF: neither tifffile nor imageio is installed")


def _load_tiff(path: Path) -> np.ndarray:
    """Load a TIFF file, trying tifffile then imageio."""
    if _HAS_TIFFFILE:
        import tifffile

        return tifffile.imread(str(path))
    try:
        import imageio.v3 as iio

        return np.array(iio.imread(str(path)))
    except ImportError:
        pass
    raise RuntimeError("Cannot read TIFF: neither tifffile nor imageio is installed")


_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy"}
_MASK_SUFFIXES = ("_masks", "_mask", "_labels", "_label", "_seg")
MIN_TRAIN_MASKS = 5


def _find_image_mask_pairs(
    image_dir: Path, gt_dir: Path
) -> list[tuple[Path, Path]]:
    """Match image files in *image_dir* to mask files in *gt_dir* by stem."""
    img_files = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS
    )

    def _gt_stem(p: Path) -> str:
        s = p.stem
        for sfx in _MASK_SUFFIXES:
            if s.endswith(sfx):
                return s[: -len(sfx)]
        return s

    gt_map: dict[str, Path] = {}
    for gf in gt_dir.iterdir():
        if gf.suffix.lower() in _IMAGE_EXTS or gf.suffix.lower() == ".npy":
            gt_map[_gt_stem(gf).lower()] = gf

    pairs: list[tuple[Path, Path]] = []
    for img in img_files:
        key = img.stem.lower()
        if key in gt_map:
            pairs.append((img, gt_map[key]))

    return pairs


def prepare_data(
    image_dir: Path,
    gt_dir: Path,
    test_split: float = 0.2,
    seed: int = 42,
    output_root: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Organise images and masks into CellPose-compatible train/test directories.

    Each image is copied as ``{name}_img.tif`` and its paired mask as
    ``{name}_masks.tif`` under ``output_root/train/`` and
    ``output_root/test/`` respectively.

    Parameters
    ----------
    image_dir : Path
        Directory containing raw microscopy images.
    gt_dir : Path
        Directory containing ground-truth label masks.
    test_split : float
        Fraction of pairs to reserve for the test set.
    seed : int
        Random seed for the split.
    output_root : Path | None
        Where to create ``train/`` and ``test/`` subdirectories.
        Defaults to a sibling ``cellpose_data/`` directory next to *image_dir*.

    Returns
    -------
    tuple[Path, Path]
        (train_dir, test_dir) — absolute paths.
    """
    image_dir = Path(image_dir)
    gt_dir = Path(gt_dir)
    if output_root is None:
        output_root = image_dir.parent / "cellpose_data"

    train_dir = output_root / "train"
    test_dir = output_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_image_mask_pairs(image_dir, gt_dir)
    if not pairs:
        raise ValueError(
            f"No matched image/mask pairs found in {image_dir!r} and {gt_dir!r}"
        )

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_split)) if len(shuffled) > 1 else 0
    test_pairs = shuffled[:n_test]
    train_pairs = shuffled[n_test:]

    def _copy_pair(img_path: Path, mask_path: Path, dest_dir: Path) -> None:
        stem = img_path.stem
        # Images: copy or convert to uint16 TIFF
        dest_img = dest_dir / f"{stem}_img.tif"
        dest_mask = dest_dir / f"{stem}_masks.tif"
        if img_path.suffix.lower() in {".tif", ".tiff"}:
            shutil.copy2(img_path, dest_img)
        else:
            arr = _load_tiff(img_path)
            _save_tiff(arr, dest_img)

        if mask_path.suffix.lower() in {".tif", ".tiff"}:
            shutil.copy2(mask_path, dest_mask)
        elif mask_path.suffix.lower() == ".npy":
            arr = np.load(mask_path).astype(np.uint16)
            _save_tiff(arr, dest_mask)
        else:
            arr = _load_tiff(mask_path)
            _save_tiff(arr.astype(np.uint16), dest_mask)

    for img_p, mask_p in train_pairs:
        _copy_pair(img_p, mask_p, train_dir)
    for img_p, mask_p in test_pairs:
        _copy_pair(img_p, mask_p, test_dir)

    return train_dir, test_dir


# ── CellPose training ──────────────────────────────────────────────────────────


def _load_cellpose_dataset(
    data_dir: Path,
    image_filter: str,
    mask_filter: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load image/mask arrays from a CellPose-style directory.

    Parameters
    ----------
    data_dir : Path
        Directory with ``*{image_filter}.tif`` and ``*{mask_filter}.tif``.
    image_filter, mask_filter : str
        Filename suffixes (without extension) that identify images vs. masks.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        (images, masks) — parallel lists.
    """
    img_files = sorted(
        f for f in data_dir.iterdir() if image_filter in f.stem and f.suffix.lower() in {".tif", ".tiff"}
    )
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for img_path in img_files:
        stem_base = img_path.stem.replace(image_filter, "")
        mask_candidates = list(data_dir.glob(f"{stem_base}{mask_filter}.*"))
        if not mask_candidates:
            continue
        images.append(_load_tiff(img_path))
        masks.append(_load_tiff(mask_candidates[0]).astype(np.uint16))

    return images, masks


def _count_nonzero_labels(mask: np.ndarray) -> int:
    """Count distinct labelled objects in a mask, excluding background."""
    return int(np.count_nonzero(np.unique(mask)))


def train_cellpose(config: TrainConfig) -> TrainResult:
    """Fine-tune a CellPose model using the provided configuration.

    Parameters
    ----------
    config : TrainConfig
        Full training configuration.

    Returns
    -------
    TrainResult
        Training result including model path and loss history.

    Raises
    ------
    ImportError
        If cellpose is not installed.
    ValueError
        If the training directory is empty or no pairs are found.
    """
    if not _HAS_CELLPOSE:
        raise ImportError(
            "cellpose is required for training. Install with: pip install cellpose"
        )

    from cellpose import models as cp_models
    from cellpose import train as cp_train

    train_dir = Path(config.train_dir)
    test_dir = Path(config.test_dir) if config.test_dir is not None else None

    # Load training data
    train_images, train_masks = _load_cellpose_dataset(
        train_dir, config.image_filter, config.mask_filter
    )
    if not train_images:
        raise ValueError(
            f"No image/mask pairs found in {train_dir!r} "
            f"(image_filter={config.image_filter!r}, mask_filter={config.mask_filter!r})"
        )
    sparse_train_images = sum(
        _count_nonzero_labels(mask) < MIN_TRAIN_MASKS for mask in train_masks
    )
    if sparse_train_images == len(train_masks):
        raise ValueError(
            "CellPose requires at least "
            f"{MIN_TRAIN_MASKS} labelled objects per image; "
            f"{sparse_train_images} of {len(train_masks)} training images were "
            f"too sparse (<{MIN_TRAIN_MASKS} objects)."
        )

    # Load test data if available
    test_images: list[np.ndarray] = []
    test_masks: list[np.ndarray] = []
    if test_dir is not None and test_dir.exists():
        test_images, test_masks = _load_cellpose_dataset(
            test_dir, config.image_filter, config.mask_filter
        )

    # Initialise model
    np.random.seed(config.seed)
    model = cp_models.CellposeModel(
        gpu=False,
        model_type=config.pretrained,
    )

    config.save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # Build kwargs for train_seg — the API changed between cellpose versions
    train_kwargs: dict = dict(
        train_data=train_images,
        train_labels=train_masks,
        test_data=test_images if test_images else None,
        test_labels=test_masks if test_masks else None,
        save_path=str(config.save_dir),
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        SGD=False,
        batch_size=config.batch_size,
    )
    if "min_train_masks" in inspect.signature(cp_train.train_seg).parameters:
        train_kwargs["min_train_masks"] = MIN_TRAIN_MASKS

    try:
        # cellpose >= 3: train_seg returns (path, train_losses, test_losses)
        result = cp_train.train_seg(model.net, **train_kwargs)
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            model_path_raw, train_losses, test_losses = result[0], list(result[1]), list(result[2])
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            model_path_raw, train_losses = result[0], list(result[1])
            test_losses = []
        else:
            model_path_raw = result
            train_losses, test_losses = [], []
    except TypeError:
        # Fallback for older cellpose API
        result = cp_train.train_seg(model.net, **{
            k: v for k, v in train_kwargs.items()
            if k not in ("SGD",)
        })
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            model_path_raw = result[0]
            train_losses = list(result[1]) if len(result) > 1 else []
            test_losses = list(result[2]) if len(result) > 2 else []
        else:
            model_path_raw = result
            train_losses, test_losses = [], []

    elapsed = time.monotonic() - t0

    model_path = Path(str(model_path_raw))

    # Determine best epoch (lowest test loss, or last training epoch)
    if test_losses:
        best_epoch = int(np.argmin(test_losses))
    elif train_losses:
        best_epoch = int(np.argmin(train_losses))
    else:
        best_epoch = config.n_epochs - 1

    return TrainResult(
        model_path=model_path,
        train_losses=train_losses,
        test_losses=test_losses,
        best_epoch=best_epoch,
        elapsed_seconds=elapsed,
        config_used=config,
    )


# ── Post-training validation ───────────────────────────────────────────────────


def validate_model(model_path: Path, test_dir: Path) -> "EvaluationResult":  # noqa: F821
    """Run inference with a fine-tuned model on *test_dir* and evaluate.

    Parameters
    ----------
    model_path : Path
        Path to the saved CellPose model weights.
    test_dir : Path
        Directory containing ``*_img.tif`` and ``*_masks.tif`` pairs.

    Returns
    -------
    EvaluationResult
        Evaluation metrics from :mod:`microagent.core.evaluate`.

    Raises
    ------
    ImportError
        If cellpose is not installed.
    """
    if not _HAS_CELLPOSE:
        raise ImportError(
            "cellpose is required for validation. Install with: pip install cellpose"
        )

    import tempfile

    from cellpose import models as cp_models

    from microagent.core.evaluate import evaluate_masks

    model_path = Path(model_path)
    test_dir = Path(test_dir)

    # Load the fine-tuned model
    model = cp_models.CellposeModel(gpu=False, pretrained_model=str(model_path))

    # Find images
    img_files = sorted(
        f for f in test_dir.iterdir() if "_img" in f.stem and f.suffix.lower() in {".tif", ".tiff"}
    )
    if not img_files:
        raise ValueError(f"No *_img.tif files found in {test_dir!r}")

    with tempfile.TemporaryDirectory() as tmp_pred:
        pred_dir = Path(tmp_pred)
        for img_path in img_files:
            img = _load_tiff(img_path)
            # Ensure 2-D for CellPose
            if img.ndim == 3:
                img_2d = img[0] if img.shape[0] <= 4 else img[:, :, 0]
            else:
                img_2d = img
            masks, _, _ = model.eval(img_2d, diameter=None, channels=[0, 0])
            stem_base = img_path.stem.replace("_img", "")
            _save_tiff(masks.astype(np.uint16), pred_dir / f"{stem_base}_masks.tif")

        return evaluate_masks(pred_dir, test_dir)
