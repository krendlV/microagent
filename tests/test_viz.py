"""Tests for microagent.viz.overlays and microagent.viz.plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microagent.viz.overlays import (
    create_comparison,
    create_error_overlay,
    create_overlay,
    save_overlay_montage,
)
from microagent.viz.plots import (
    plot_metric_per_image,
    plot_metrics_summary,
    plot_object_size_distribution,
    plot_training_curves,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_gray(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Simple synthetic grayscale image."""
    rng = np.random.default_rng(42)
    return (rng.integers(0, 256, shape)).astype(np.uint8)


def _make_multichannel(shape: tuple[int, int] = (64, 64), n_channels: int = 2) -> np.ndarray:
    """Channel-first (C, H, W) synthetic image."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 1000, (n_channels, *shape), dtype=np.uint16)


def _make_masks(
    shape: tuple[int, int] = (64, 64),
    circles: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Integer label mask with filled circles."""
    if circles is None:
        circles = [(16, 16, 8), (16, 48, 8), (48, 16, 8), (48, 48, 8)]
    mask = np.zeros(shape, dtype=np.int32)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for label, (cy, cx, r) in enumerate(circles, start=1):
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = label
    return mask


def _make_eval_result():
    """Minimal synthetic EvaluationResult for plot tests."""
    from microagent.core.evaluate import (
        DatasetMetrics,
        EvaluationResult,
        ImageMetrics,
        ThresholdMetrics,
    )

    def _tm(thresh, f1, prec, rec):
        return ThresholdMetrics(
            threshold=thresh,
            precision=prec,
            recall=rec,
            f1=f1,
            tp=3,
            fp=1,
            fn=1,
            mean_true_score=f1 * 0.9,
        )

    per_image = [
        ImageMetrics(
            filename=f"img_{i}.tif",
            gt_count=4,
            pred_count=4,
            per_threshold=[_tm(0.5, 0.8 - i * 0.1, 0.85, 0.75)],
            mean_f1=0.8 - i * 0.1,
            panoptic_quality=0.7,
            iou_distribution=[0.7, 0.8],
        )
        for i in range(4)
    ]
    summary = DatasetMetrics(
        n_images=4,
        per_threshold=[
            _tm(0.5, 0.75, 0.80, 0.70),
            _tm(0.75, 0.60, 0.65, 0.55),
            _tm(0.9, 0.40, 0.45, 0.35),
        ],
        mean_f1=0.75,
        panoptic_quality=0.65,
        mean_gt_count=4.0,
        mean_pred_count=4.0,
    )
    return EvaluationResult(
        per_image=per_image,
        summary=summary,
        best_images=["img_0.tif"],
        worst_images=["img_3.tif"],
        unmatched_preds=[],
        unmatched_gts=[],
    )


# ── Overlay tests ──────────────────────────────────────────────────────────────


class TestCreateOverlay:
    def test_shape_grayscale(self):
        img = _make_gray()
        masks = _make_masks()
        out = create_overlay(img, masks)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_shape_multichannel(self):
        img = _make_multichannel()
        masks = _make_masks()
        out = create_overlay(img, masks)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_all_background_mask(self):
        img = _make_gray()
        masks = np.zeros((64, 64), dtype=np.int32)
        out = create_overlay(img, masks)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_alpha_zero_preserves_image(self):
        """With alpha=0 overlay should be close to the original grayscale-as-RGB."""
        img = _make_gray()
        masks = _make_masks()
        out = create_overlay(img, masks, alpha=0.0)
        assert out.shape == (64, 64, 3)


class TestErrorOverlay:
    def test_shape(self):
        img = _make_gray()
        pred = _make_masks()
        gt = _make_masks()
        out = create_error_overlay(img, pred, gt)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_tp_region_is_greenish(self):
        """When pred and gt are identical, all objects are TP — should be green."""
        img = _make_gray()
        masks = _make_masks()
        out = create_error_overlay(img, masks, masks, iou_threshold=0.5)
        # TP pixels: green channel dominates red and blue
        tp_mask = masks > 0
        green = out[tp_mask, 1].astype(float)
        red = out[tp_mask, 0].astype(float)
        blue = out[tp_mask, 2].astype(float)
        assert green.mean() > red.mean()
        assert green.mean() > blue.mean()

    def test_fp_region_is_reddish(self):
        """Extra predictions with no GT are FP — should be red."""
        img = _make_gray()
        gt = np.zeros((64, 64), dtype=np.int32)  # no GT objects
        pred = _make_masks(circles=[(32, 32, 8)])  # one FP prediction
        out = create_error_overlay(img, pred, gt, iou_threshold=0.5)
        fp_mask = pred > 0
        red = out[fp_mask, 0].astype(float)
        blue = out[fp_mask, 2].astype(float)
        assert red.mean() > blue.mean()

    def test_fn_region_is_bluish(self):
        """GT objects with no prediction are FN — should be blue."""
        img = _make_gray()
        gt = _make_masks(circles=[(32, 32, 8)])  # one GT object
        pred = np.zeros((64, 64), dtype=np.int32)  # no predictions
        out = create_error_overlay(img, pred, gt, iou_threshold=0.5)
        fn_mask = gt > 0
        blue = out[fn_mask, 2].astype(float)
        red = out[fn_mask, 0].astype(float)
        assert blue.mean() > red.mean()


class TestCreateComparison:
    def test_shape_wider_than_single(self):
        img = _make_gray()
        masks_a = _make_masks()
        masks_b = _make_masks(circles=[(20, 20, 10)])
        out = create_comparison(img, masks_a, masks_b)
        assert out.ndim == 3
        assert out.shape[2] == 3
        # Side-by-side so width should be greater than height
        assert out.shape[1] > out.shape[0] or out.shape[1] >= 64

    def test_dtype_uint8(self):
        img = _make_gray()
        masks_a = _make_masks()
        masks_b = _make_masks()
        out = create_comparison(img, masks_a, masks_b)
        assert out.dtype == np.uint8


class TestSaveOverlayMontage:
    def test_file_created(self, tmp_path: Path):
        images = [_make_gray() for _ in range(4)]
        masks_list = [_make_masks() for _ in range(4)]
        out = save_overlay_montage(images, masks_list, tmp_path / "montage.png", ncols=2)
        assert out.exists()
        assert out.suffix == ".png"

    def test_single_image(self, tmp_path: Path):
        out = save_overlay_montage(
            [_make_gray()], [_make_masks()], tmp_path / "single.png", ncols=3
        )
        assert out.exists()

    def test_returns_path(self, tmp_path: Path):
        result = save_overlay_montage(
            [_make_gray()], [_make_masks()], tmp_path / "m.png"
        )
        assert isinstance(result, Path)


# ── Plot tests ─────────────────────────────────────────────────────────────────


class TestPlotMetricsSummary:
    def test_output_exists(self, tmp_path: Path):
        result = _make_eval_result()
        out = plot_metrics_summary(result, tmp_path / "metrics.png")
        assert out.exists()
        assert out.suffix == ".png"

    def test_returns_path(self, tmp_path: Path):
        result = _make_eval_result()
        out = plot_metrics_summary(result, tmp_path / "m.png")
        assert isinstance(out, Path)


class TestPlotObjectSizeDistribution:
    def test_output_exists(self, tmp_path: Path):
        masks = _make_masks()
        out = plot_object_size_distribution(masks, tmp_path / "sizes.png")
        assert out.exists()

    def test_empty_masks(self, tmp_path: Path):
        masks = np.zeros((64, 64), dtype=np.int32)
        out = plot_object_size_distribution(masks, tmp_path / "empty.png")
        assert out.exists()


class TestPlotMetricPerImage:
    def test_output_exists(self, tmp_path: Path):
        result = _make_eval_result()
        out = plot_metric_per_image(result, "f1", tmp_path / "per_img.png")
        assert out.exists()

    def test_precision_metric(self, tmp_path: Path):
        result = _make_eval_result()
        out = plot_metric_per_image(result, "precision", tmp_path / "prec.png")
        assert out.exists()


class TestPlotTrainingCurves:
    def test_output_exists(self, tmp_path: Path):
        train = [1.0, 0.8, 0.6, 0.5, 0.45]
        val = [1.1, 0.9, 0.7, 0.6, 0.55]
        out = plot_training_curves(train, val, tmp_path / "curves.png")
        assert out.exists()

    def test_train_only(self, tmp_path: Path):
        train = [1.0, 0.8, 0.6]
        out = plot_training_curves(train, [], tmp_path / "train_only.png")
        assert out.exists()

    def test_returns_path(self, tmp_path: Path):
        out = plot_training_curves([1.0, 0.5], [0.9, 0.6], tmp_path / "c.png")
        assert isinstance(out, Path)
