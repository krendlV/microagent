"""Tests for microagent.core.evaluate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from microagent.cli import app
from microagent.core.evaluate import (
    EvaluationResult,
    compare_runs,
    evaluate_masks,
    _compute_iou_matrix,
    _match_at_threshold,
    _metrics_from_fallback,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_label_mask(shape: tuple[int, int], circles: list[tuple[int, int, int]]) -> np.ndarray:
    """Create an integer label mask with filled circles.

    Parameters
    ----------
    shape : (H, W)
    circles : list of (cy, cx, radius)
    """
    mask = np.zeros(shape, dtype=np.int32)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for label, (cy, cx, r) in enumerate(circles, start=1):
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = label
    return mask


def _write_mask(path: Path, mask: np.ndarray) -> None:
    tifffile.imwrite(str(path), mask.astype(np.int32))


SHAPE = (128, 128)
CIRCLES = [(30, 30, 15), (30, 90, 15), (90, 30, 15), (90, 90, 15)]


@pytest.fixture
def pred_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pred"
    d.mkdir()
    return d


@pytest.fixture
def gt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "gt"
    d.mkdir()
    return d


@pytest.fixture
def perfect_dirs(pred_dir: Path, gt_dir: Path) -> tuple[Path, Path]:
    """Identical masks in both dirs."""
    mask = _make_label_mask(SHAPE, CIRCLES)
    _write_mask(pred_dir / "img_000.tif", mask)
    _write_mask(gt_dir / "img_000.tif", mask)
    return pred_dir, gt_dir


@pytest.fixture
def no_overlap_dirs(pred_dir: Path, gt_dir: Path) -> tuple[Path, Path]:
    """Pred in top-left 30x30, GT in bottom-right – guaranteed zero overlap."""
    # GT: two small circles in the bottom-right corner
    gt_circles = [(110, 110, 5), (110, 120, 5)]
    gt_mask = _make_label_mask((128, 128), gt_circles)
    # Pred: two small circles in the top-left corner
    pred_circles = [(10, 10, 5), (10, 20, 5)]
    pred_mask = _make_label_mask((128, 128), pred_circles)
    _write_mask(gt_dir / "img_000.tif", gt_mask)
    _write_mask(pred_dir / "img_000.tif", pred_mask)
    return pred_dir, gt_dir


# ── Unit: IoU matrix & matching ───────────────────────────────────────────────


class TestIoUMatrix:
    def test_identical_masks(self) -> None:
        mask = _make_label_mask(SHAPE, CIRCLES)
        iou_mat = _compute_iou_matrix(mask, mask)
        assert iou_mat.shape == (4, 4)
        np.testing.assert_allclose(np.diag(iou_mat), 1.0, atol=1e-6)

    def test_no_overlap(self) -> None:
        gt = _make_label_mask(SHAPE, [(30, 10, 5)])
        pred = _make_label_mask(SHAPE, [(90, 100, 5)])
        iou_mat = _compute_iou_matrix(gt, pred)
        assert iou_mat[0, 0] == 0.0

    def test_empty_pred(self) -> None:
        gt = _make_label_mask(SHAPE, CIRCLES)
        pred = np.zeros(SHAPE, dtype=np.int32)
        iou_mat = _compute_iou_matrix(gt, pred)
        assert iou_mat.shape == (4, 0)

    def test_empty_both(self) -> None:
        gt = np.zeros(SHAPE, dtype=np.int32)
        pred = np.zeros(SHAPE, dtype=np.int32)
        iou_mat = _compute_iou_matrix(gt, pred)
        assert iou_mat.shape == (0, 0)


class TestMatchAtThreshold:
    def test_perfect_match(self) -> None:
        mask = _make_label_mask(SHAPE, CIRCLES)
        iou_mat = _compute_iou_matrix(mask, mask)
        tp, fp, fn, ious = _match_at_threshold(iou_mat, 0.5)
        assert tp == 4
        assert fp == 0
        assert fn == 0

    def test_no_overlap(self) -> None:
        gt = _make_label_mask(SHAPE, [(30, 10, 5)])
        pred = _make_label_mask(SHAPE, [(90, 100, 5)])
        iou_mat = _compute_iou_matrix(gt, pred)
        tp, fp, fn, _ = _match_at_threshold(iou_mat, 0.5)
        assert tp == 0
        assert fp == 1
        assert fn == 1

    def test_empty_pred(self) -> None:
        iou_mat = np.zeros((3, 0))
        tp, fp, fn, ious = _match_at_threshold(iou_mat, 0.5)
        assert tp == 0
        assert fp == 0
        assert fn == 3

    def test_empty_gt(self) -> None:
        iou_mat = np.zeros((0, 2))
        tp, fp, fn, ious = _match_at_threshold(iou_mat, 0.5)
        assert tp == 0
        assert fp == 2
        assert fn == 0


# ── evaluate_masks: perfect match ─────────────────────────────────────────────


class TestPerfectMatch:
    def test_f1_one_at_all_thresholds(self, perfect_dirs: tuple[Path, Path]) -> None:
        pred_dir, gt_dir = perfect_dirs
        result = evaluate_masks(pred_dir, gt_dir, thresholds=[0.5, 0.75, 0.9])
        assert len(result.per_image) == 1
        im = result.per_image[0]
        for tm in im.per_threshold:
            assert abs(tm.f1 - 1.0) < 1e-6, f"Expected F1=1 at {tm.threshold}, got {tm.f1}"
        assert abs(result.summary.map - 1.0) < 1e-6
        assert abs(result.summary.panoptic_quality - 1.0) < 1e-6

    def test_no_unmatched(self, perfect_dirs: tuple[Path, Path]) -> None:
        result = evaluate_masks(*perfect_dirs)
        assert result.unmatched_preds == []
        assert result.unmatched_gts == []


# ── evaluate_masks: no overlap ────────────────────────────────────────────────


class TestNoMatch:
    def test_f1_zero_at_all_thresholds(self, no_overlap_dirs: tuple[Path, Path]) -> None:
        pred_dir, gt_dir = no_overlap_dirs
        result = evaluate_masks(pred_dir, gt_dir, thresholds=[0.5, 0.75, 0.9])
        im = result.per_image[0]
        for tm in im.per_threshold:
            assert tm.f1 == 0.0, f"Expected F1=0 at {tm.threshold}, got {tm.f1}"
        assert result.summary.map == 0.0


# ── evaluate_masks: partial match (known TP/FP/FN) ────────────────────────────


class TestPartialMatch:
    def test_known_counts(self, pred_dir: Path, gt_dir: Path) -> None:
        """GT has 4 objects, pred has 3 matching + 1 false positive."""
        # GT: 4 circles
        gt_circles = [(30, 30, 15), (30, 90, 15), (90, 30, 15), (90, 90, 15)]
        gt_mask = _make_label_mask(SHAPE, gt_circles)

        # Pred: first 3 GT circles (perfect match) + 1 far-away FP
        pred_circles = [(30, 30, 15), (30, 90, 15), (90, 30, 15), (10, 60, 5)]
        pred_mask = _make_label_mask(SHAPE, pred_circles)

        _write_mask(gt_dir / "partial.tif", gt_mask)
        _write_mask(pred_dir / "partial.tif", pred_mask)

        result = evaluate_masks(pred_dir, gt_dir, thresholds=[0.5])
        im = result.per_image[0]
        m05 = im.per_threshold[0]

        assert m05.tp == 3
        assert m05.fp == 1
        assert m05.fn == 1
        # precision = 3/4 = 0.75, recall = 3/4 = 0.75
        assert abs(m05.precision - 0.75) < 1e-6
        assert abs(m05.recall - 0.75) < 1e-6


# ── Fallback matching produces same results as direct call ────────────────────


class TestFallbackMatching:
    def test_fallback_same_as_direct(self, perfect_dirs: tuple[Path, Path]) -> None:
        """Force fallback and verify metrics match non-fallback for perfect masks."""
        pred_dir, gt_dir = perfect_dirs

        result_fallback = evaluate_masks(pred_dir, gt_dir, thresholds=[0.5], force_fallback=True)
        result_normal = evaluate_masks(pred_dir, gt_dir, thresholds=[0.5], force_fallback=False)

        for fb_im, nm_im in zip(result_fallback.per_image, result_normal.per_image):
            for fb_tm, nm_tm in zip(fb_im.per_threshold, nm_im.per_threshold):
                assert abs(fb_tm.f1 - nm_tm.f1) < 1e-6
                assert fb_tm.tp == nm_tm.tp
                assert fb_tm.fp == nm_tm.fp
                assert fb_tm.fn == nm_tm.fn

    def test_fallback_metrics_from_fallback_function(self) -> None:
        """Test _metrics_from_fallback directly with a known mask."""
        mask = _make_label_mask(SHAPE, CIRCLES)
        per_thresh, iou_dist = _metrics_from_fallback(mask, mask, [0.5])
        assert abs(per_thresh[0].f1 - 1.0) < 1e-6
        assert abs(per_thresh[0].mean_true_score - 1.0) < 1e-6
        assert len(iou_dist) == 4

    def test_fallback_no_stardist_env(self, pred_dir: Path, gt_dir: Path) -> None:
        """Simulate stardist not installed: patch _HAS_STARDIST to False."""
        mask = _make_label_mask(SHAPE, CIRCLES)
        _write_mask(pred_dir / "img.tif", mask)
        _write_mask(gt_dir / "img.tif", mask)

        with patch("microagent.core.evaluate._HAS_STARDIST", False):
            result = evaluate_masks(pred_dir, gt_dir, thresholds=[0.5])

        im = result.per_image[0]
        assert abs(im.per_threshold[0].f1 - 1.0) < 1e-6


# ── compare_runs ──────────────────────────────────────────────────────────────


class TestCompareRuns:
    def _make_result(self, f1: float, filename: str = "img.tif") -> EvaluationResult:
        """Build a minimal EvaluationResult with a given F1@0.5."""
        from microagent.core.evaluate import (
            DatasetMetrics,
            ImageMetrics,
            ThresholdMetrics,
        )

        tm = ThresholdMetrics(
            threshold=0.5,
            precision=f1,
            recall=f1,
            f1=f1,
            tp=int(f1 * 4),
            fp=int((1 - f1) * 4),
            fn=int((1 - f1) * 4),
            mean_true_score=f1,
        )
        im = ImageMetrics(
            filename=filename,
            gt_count=4,
            pred_count=4,
            per_threshold=[tm],
            map=f1,
            panoptic_quality=f1 * f1,
            iou_distribution=[],
        )
        summary = DatasetMetrics(
            n_images=1,
            per_threshold=[tm],
            map=f1,
            panoptic_quality=f1 * f1,
            mean_gt_count=4.0,
            mean_pred_count=4.0,
        )
        return EvaluationResult(
            per_image=[im],
            summary=summary,
            best_images=[filename],
            worst_images=[filename],
            unmatched_preds=[],
            unmatched_gts=[],
        )

    def test_improvement_detected(self) -> None:
        baseline = self._make_result(0.5)
        improved = self._make_result(0.8)
        comp = compare_runs(baseline, improved)
        assert comp.metric_deltas["f1@0.50"] > 0
        assert "img.tif" in comp.improved_images
        assert comp.regressed_images == []

    def test_regression_detected(self) -> None:
        baseline = self._make_result(0.9)
        worse = self._make_result(0.3)
        comp = compare_runs(baseline, worse)
        assert comp.metric_deltas["f1@0.50"] < 0
        assert "img.tif" in comp.regressed_images
        assert comp.improved_images == []

    def test_no_change(self) -> None:
        r = self._make_result(0.7)
        comp = compare_runs(r, r)
        assert comp.improved_images == []
        assert comp.regressed_images == []
        for delta in comp.metric_deltas.values():
            assert abs(delta) < 1e-9


# ── JSON round-trip ───────────────────────────────────────────────────────────


class TestJsonRoundTrip:
    def test_save_load(self, perfect_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        result = evaluate_masks(*perfect_dirs)
        out = tmp_path / "metrics.json"
        result.save_json(out)
        loaded = EvaluationResult.load_json(out)
        assert loaded.summary.n_images == result.summary.n_images
        assert abs(loaded.summary.map - result.summary.map) < 1e-9


# ── CLI integration ───────────────────────────────────────────────────────────


class TestEvaluateCli:
    def test_cli_perfect_match(self, perfect_dirs: tuple[Path, Path]) -> None:
        runner = CliRunner()
        pred_dir, gt_dir = perfect_dirs
        result = runner.invoke(app, ["evaluate", str(pred_dir), str(gt_dir)])
        assert result.exit_code == 0, result.output
        assert "SUMMARY" in result.output

    def test_cli_with_output(self, perfect_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        runner = CliRunner()
        pred_dir, gt_dir = perfect_dirs
        out = tmp_path / "out.json"
        result = runner.invoke(
            app,
            ["evaluate", str(pred_dir), str(gt_dir), "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "summary" in data

    def test_cli_custom_thresholds(self, perfect_dirs: tuple[Path, Path]) -> None:
        runner = CliRunner()
        pred_dir, gt_dir = perfect_dirs
        result = runner.invoke(
            app,
            ["evaluate", str(pred_dir), str(gt_dir), "--thresholds", "0.5,0.75"],
        )
        assert result.exit_code == 0
        assert "F1@0.5" in result.output

    def test_cli_compare(self, perfect_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        runner = CliRunner()
        pred_dir, gt_dir = perfect_dirs
        baseline_json = tmp_path / "baseline.json"
        # First run to save baseline
        runner.invoke(
            app,
            ["evaluate", str(pred_dir), str(gt_dir), "--output", str(baseline_json)],
        )
        # Second run comparing against baseline
        result = runner.invoke(
            app,
            ["evaluate", str(pred_dir), str(gt_dir), "--compare", str(baseline_json)],
        )
        assert result.exit_code == 0
        assert "Comparison" in result.output

    def test_cli_unmatched_warning(
        self, pred_dir: Path, gt_dir: Path, tmp_path: Path
    ) -> None:
        """Extra pred file without GT counterpart triggers unmatched warning."""
        mask = _make_label_mask(SHAPE, CIRCLES)
        _write_mask(pred_dir / "img_000.tif", mask)
        _write_mask(pred_dir / "extra.tif", mask)
        _write_mask(gt_dir / "img_000.tif", mask)

        runner = CliRunner()
        result = runner.invoke(app, ["evaluate", str(pred_dir), str(gt_dir)])
        assert result.exit_code == 0
        assert "Unmatched" in result.output
