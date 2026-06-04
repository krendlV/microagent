"""Tests for microagent.core.optimize."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile
from skimage import measure
from typer.testing import CliRunner

from microagent.cli import app
from microagent.core.optimize import (
    OptimizationResult,
    OptimizeConfig,
    TrialRecord,
    _get_metric_value,
    create_objective,
    run_optimization,
    select_search_space,
)

optuna = pytest.importorskip("optuna")


# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_circle_mask(
    shape: tuple[int, int] = (128, 128),
    n_circles: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Return an integer label mask with non-overlapping circles."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.uint16)
    for _ in range(n_circles):
        cy = int(rng.integers(20, shape[0] - 20))
        cx = int(rng.integers(20, shape[1] - 20))
        r = int(rng.integers(8, 20))
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = int(
            rng.integers(10000, 40000)
        )
    return img


@pytest.fixture
def opt_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Matching image + GT directories with 3 synthetic TIFFs each."""
    img_dir = tmp_path / "images"
    gt_dir = tmp_path / "gt"
    img_dir.mkdir()
    gt_dir.mkdir()

    for i in range(3):
        img = _make_circle_mask(seed=i)
        tifffile.imwrite(img_dir / f"img_{i:03d}.tif", img)
        binary = img > (img.max() * 0.1)
        labels = measure.label(binary).astype(np.uint16)
        tifffile.imwrite(gt_dir / f"img_{i:03d}_mask.tif", labels)

    return img_dir, gt_dir


# ── select_search_space ────────────────────────────────────────────────────────


class TestSelectSearchSpace:
    def test_returns_empty_for_none(self) -> None:
        assert select_search_space(None) == {}

    def test_returns_empty_for_missing_diameter(self) -> None:
        project = {"imaging": {"staining": "fluorescence"}}
        assert select_search_space(project) == {}

    def test_narrows_diameter_range(self) -> None:
        project = {"imaging": {"cell_diameter_pixels": 50}}
        space = select_search_space(project)
        assert space["diameter_low"] == pytest.approx(25.0)
        assert space["diameter_high"] == pytest.approx(75.0)

    def test_clamps_diameter_low_at_ten(self) -> None:
        project = {"imaging": {"cell_diameter_pixels": 5}}
        space = select_search_space(project)
        assert space["diameter_low"] == pytest.approx(10.0)

    def test_large_diameter(self) -> None:
        project = {"imaging": {"cell_diameter_pixels": 200}}
        space = select_search_space(project)
        assert space["diameter_low"] == pytest.approx(100.0)
        assert space["diameter_high"] == pytest.approx(300.0)


# ── _get_metric_value ──────────────────────────────────────────────────────────


class TestGetMetricValue:
    def _make_im_metrics(self, f1: float = 0.7, iou: float = 0.5) -> MagicMock:
        tm = MagicMock()
        tm.threshold = iou
        tm.f1 = f1
        tm.precision = 0.8
        tm.recall = 0.6
        im = MagicMock()
        im.per_threshold = [tm]
        im.map = 0.65
        im.panoptic_quality = 0.55
        return im

    def test_f1(self) -> None:
        im = self._make_im_metrics(f1=0.72)
        assert _get_metric_value(im, "f1", 0.5) == pytest.approx(0.72)

    def test_map(self) -> None:
        im = self._make_im_metrics()
        assert _get_metric_value(im, "map", 0.5) == pytest.approx(0.65)

    def test_pq(self) -> None:
        im = self._make_im_metrics()
        assert _get_metric_value(im, "pq", 0.5) == pytest.approx(0.55)

    def test_precision(self) -> None:
        im = self._make_im_metrics()
        assert _get_metric_value(im, "precision", 0.5) == pytest.approx(0.8)


# ── create_objective ───────────────────────────────────────────────────────────


class TestCreateObjective:
    def test_objective_returns_float(self, opt_dirs: tuple[Path, Path]) -> None:
        """Objective called with a mock trial should return a float in [0, 1]."""
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=1,
            metric="f1",
            iou_threshold=0.5,
        )

        # Mock the segmenter so we don't need real CellPose
        fake_mask = np.zeros((128, 128), dtype=np.int32)
        fake_mask[20:50, 20:50] = 1

        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            objective = create_objective(config)

        # Build a minimal mock trial
        trial = MagicMock()
        trial.suggest_float.side_effect = [30.0, 0.4, 0.0]  # diameter, flow, cellprob
        trial.should_prune.return_value = False
        trial.report.return_value = None

        value = objective(trial)

        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0

    def test_objective_calls_predict_for_each_image(
        self, opt_dirs: tuple[Path, Path]
    ) -> None:
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=1,
        )

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        trial = MagicMock()
        trial.suggest_float.side_effect = [30.0, 0.4, 0.0]
        trial.should_prune.return_value = False
        trial.report.return_value = None

        # Keep patch active for both create_objective and objective(trial)
        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            objective = create_objective(config)
            objective(trial)

        # 3 images in opt_dirs fixture → predict called 3 times
        assert mock_segmenter.predict.call_count == 3

    def test_objective_raises_on_empty_gt_dir(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        gt_dir = tmp_path / "gt"
        img_dir.mkdir()
        gt_dir.mkdir()
        tifffile.imwrite(img_dir / "img.tif", np.zeros((64, 64), dtype=np.uint16))
        # gt_dir is empty → no matches

        config = OptimizeConfig(image_dir=img_dir, gt_dir=gt_dir, model="cellpose")
        with pytest.raises(RuntimeError, match="No matched"):
            create_objective(config)


# ── run_optimization ───────────────────────────────────────────────────────────


class TestRunOptimization:
    def test_save_json_round_trip(self, tmp_path: Path) -> None:
        result = OptimizationResult(
            best_params={"diameter": 42.5, "model": "cyto3"},
            best_value=0.82,
            baseline_value=0.71,
            improvement=0.11,
            trials=[
                TrialRecord(
                    number=0,
                    params={"diameter": 30.0, "flow_threshold": 0.4},
                    value=0.75,
                ),
                TrialRecord(
                    number=1,
                    params={"diameter": 42.5, "flow_threshold": 0.3},
                    value=0.82,
                ),
            ],
            study_path=tmp_path / "optuna_study.pkl",
        )

        out = tmp_path / "optimization.json"
        result.save_json(out)

        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == {
            "best_params": {"diameter": 42.5, "model": "cyto3"},
            "best_value": 0.82,
            "baseline_value": 0.71,
            "improvement": 0.11,
            "trials": [
                {
                    "number": 0,
                    "params": {"diameter": 30.0, "flow_threshold": 0.4},
                    "value": 0.75,
                },
                {
                    "number": 1,
                    "params": {"diameter": 42.5, "flow_threshold": 0.3},
                    "value": 0.82,
                },
            ],
        }

    def test_smoke_returns_result_structure(
        self, opt_dirs: tuple[Path, Path]
    ) -> None:
        """3 trials on synthetic data; verify OptimizationResult fields."""
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=3,
            metric="f1",
            seed=0,
        )

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        fake_mask[10:40, 10:40] = 1
        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            result = run_optimization(config)

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_value, float)
        assert isinstance(result.baseline_value, float)
        assert isinstance(result.improvement, float)
        assert isinstance(result.trials, list)
        assert result.improvement == pytest.approx(result.best_value - result.baseline_value)

    def test_trials_logged(self, opt_dirs: tuple[Path, Path]) -> None:
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=3,
            seed=1,
        )

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            result = run_optimization(config)

        assert len(result.trials) == 3
        for rec in result.trials:
            assert isinstance(rec, TrialRecord)
            assert isinstance(rec.number, int)
            assert isinstance(rec.params, dict)
            assert isinstance(rec.value, float)

    def test_callback_invoked(self, opt_dirs: tuple[Path, Path]) -> None:
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=2,
            seed=2,
        )

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        callback_records: list[TrialRecord] = []

        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            run_optimization(config, on_trial_complete=callback_records.append)

        assert len(callback_records) == 2

    def test_study_pickle_saved(
        self, opt_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        import pickle

        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=2,
            seed=3,
        )

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            result = run_optimization(config)

        assert result.study_path is not None
        assert result.study_path.exists()
        # Verify it's a valid pickle
        loaded = pickle.loads(result.study_path.read_bytes())
        assert hasattr(loaded, "best_params")

    def test_best_params_in_cellpose_range(self, opt_dirs: tuple[Path, Path]) -> None:
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=3,
            seed=4,
        )

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        mock_segmenter = MagicMock()
        mock_segmenter.predict.return_value = fake_mask

        with patch(
            "microagent.core.optimize._build_segmenter", return_value=mock_segmenter
        ):
            result = run_optimization(config)

        assert "diameter" in result.best_params
        assert "flow_threshold" in result.best_params
        assert "cellprob_threshold" in result.best_params
        assert 10.0 <= result.best_params["diameter"] <= 200.0
        assert 0.1 <= result.best_params["flow_threshold"] <= 1.0
        assert -3.0 <= result.best_params["cellprob_threshold"] <= 3.0


# ── search-space selection with project.yaml ───────────────────────────────────


class TestSearchSpaceWithProject:
    def test_diameter_range_narrowed_by_project(
        self, opt_dirs: tuple[Path, Path]
    ) -> None:
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=2,
            seed=5,
        )

        # Project says cell diameter = 30 → diameter search = [15, 45]
        project = {"imaging": {"cell_diameter_pixels": 30}}

        with patch(
            "microagent.core.optimize._load_project", return_value=project
        ), patch(
            "microagent.core.optimize._build_segmenter",
            return_value=MagicMock(
                predict=MagicMock(return_value=np.zeros((128, 128), dtype=np.int32))
            ),
        ):
            result = run_optimization(config)

        assert "diameter" in result.best_params
        assert 15.0 <= result.best_params["diameter"] <= 45.0

    def test_no_project_uses_wide_range(self, opt_dirs: tuple[Path, Path]) -> None:
        img_dir, gt_dir = opt_dirs
        config = OptimizeConfig(
            image_dir=img_dir,
            gt_dir=gt_dir,
            model="cellpose",
            n_trials=2,
            seed=6,
            project_path=None,
        )

        with patch(
            "microagent.core.optimize._build_segmenter",
            return_value=MagicMock(
                predict=MagicMock(return_value=np.zeros((128, 128), dtype=np.int32))
            ),
        ):
            result = run_optimization(config)

        # With wide range [10, 200] any value is valid
        assert 10.0 <= result.best_params["diameter"] <= 200.0


# ── CLI integration ────────────────────────────────────────────────────────────


class TestOptimizeCLI:
    def test_optimize_cli_smoke(self, opt_dirs: tuple[Path, Path], tmp_path: Path) -> None:
        """CLI optimize subcommand completes and prints best params."""
        img_dir, gt_dir = opt_dirs
        runner = CliRunner()
        output_json = tmp_path / "optimization.json"

        fake_mask = np.zeros((128, 128), dtype=np.int32)
        fake_mask[10:40, 10:40] = 1
        mock_seg = MagicMock()
        mock_seg.predict.return_value = fake_mask

        with patch("microagent.core.optimize._build_segmenter", return_value=mock_seg):
            result = runner.invoke(
                app,
                [
                    "optimize",
                    str(img_dir),
                    str(gt_dir),
                    "--trials", "3",
                    "--metric", "f1",
                    "--model", "cellpose",
                    "--output-json", str(output_json),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Best Hyperparameters" in result.output
        assert "Optimization JSON saved" in result.output
        assert output_json.exists()

    def test_optimize_cli_missing_image_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "optimize",
                str(tmp_path / "nonexistent"),
                str(tmp_path),
                "--trials", "1",
            ],
        )
        assert result.exit_code != 0
