"""Tests for microagent.core.segment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from microagent.cli import app


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_circle_image(shape: tuple[int, int] = (256, 256), n_circles: int = 4) -> np.ndarray:
    """Return a (H, W) uint16 image with filled circles."""
    rng = np.random.default_rng(42)
    img = np.zeros(shape, dtype=np.uint16)
    for _ in range(n_circles):
        cy = int(rng.integers(30, shape[0] - 30))
        cx = int(rng.integers(30, shape[1] - 30))
        r = int(rng.integers(15, 35))
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        img[mask] = int(rng.integers(20000, 50000))
    return img


@pytest.fixture
def single_image_dir(tmp_path: Path) -> Path:
    """Directory with a single synthetic nucleus TIFF."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = _make_circle_image()
    tifffile.imwrite(img_dir / "nucleus_000.tif", img)
    return img_dir


@pytest.fixture
def multi_image_dir(tmp_path: Path) -> Path:
    """Directory with 3 synthetic nucleus TIFFs."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(99)
    for i in range(3):
        img = _make_circle_image(n_circles=int(rng.integers(2, 6)))
        tifffile.imwrite(img_dir / f"img_{i:03d}.tif", img)
    return img_dir


# ── Unit tests for CellPoseSegmenter ──────────────────────────────────────────


class TestCellPoseSegmenter:
    @pytest.mark.slow
    def test_predict_returns_mask(self, single_image_dir: Path) -> None:
        """CellPoseSegmenter.predict() returns an int32 2-D label mask with >0 labels."""
        from microagent.core.segment import CellPoseSegmenter

        seg = CellPoseSegmenter(diameter=30)
        img = _make_circle_image()
        mask = seg.predict(img)
        assert mask.ndim == 2
        assert mask.dtype == np.int32
        assert mask.max() > 0, "Expected at least one labelled cell"

    @pytest.mark.slow
    def test_predict_multichannel(self) -> None:
        """predict() accepts (C, H, W) arrays and uses channel 0."""
        from microagent.core.segment import CellPoseSegmenter

        seg = CellPoseSegmenter(diameter=30)
        img_3d = np.stack([_make_circle_image(), np.zeros((256, 256), dtype=np.uint16)])
        mask = seg.predict(img_3d)
        assert mask.ndim == 2

    def test_get_info_keys(self) -> None:
        """get_info() returns required keys without model download."""
        from unittest.mock import MagicMock, patch

        with patch("microagent.core.segment._HAS_CELLPOSE", True), patch(
            "microagent.core.segment._cp_models"
        ) as mock_models:
            mock_models.CellposeModel.return_value = MagicMock()
            from importlib import reload
            import microagent.core.segment as seg_mod

            seg = seg_mod.CellPoseSegmenter.__new__(seg_mod.CellPoseSegmenter)
            seg._diameter = 30
            seg._flow_threshold = 0.4
            seg._cellprob_threshold = 0.0
            seg._channels = [0, 0]
            info = seg.get_info()

        assert "model_name" in info
        assert "backend" in info
        assert "parameters" in info
        assert info["backend"] == "cellpose"
        assert info["model_name"] == "cpsam"

    def test_get_default_params_no_project(self) -> None:
        """get_default_params(None) returns sensible defaults."""
        from unittest.mock import MagicMock, patch

        with patch("microagent.core.segment._HAS_CELLPOSE", True), patch(
            "microagent.core.segment._cp_models"
        ) as mock_models:
            mock_models.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            seg = seg_mod.CellPoseSegmenter.__new__(seg_mod.CellPoseSegmenter)
            seg._diameter = None
            seg._flow_threshold = 0.4
            seg._cellprob_threshold = 0.0
            seg._channels = [0, 0]
            params = seg.get_default_params(None)

        assert "diameter" in params
        assert "flow_threshold" in params
        assert "cellprob_threshold" in params
        assert "channels" in params

    def test_get_default_params_from_project(self) -> None:
        """get_default_params reads diameter and channels from project dict."""
        from unittest.mock import MagicMock, patch

        project = {
            "imaging": {
                "cell_diameter_pixels": 25,
                "channels": {"nucleus": 1, "cytoplasm": 0},
            }
        }
        with patch("microagent.core.segment._HAS_CELLPOSE", True), patch(
            "microagent.core.segment._cp_models"
        ) as mock_models:
            mock_models.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            seg = seg_mod.CellPoseSegmenter.__new__(seg_mod.CellPoseSegmenter)
            seg._diameter = None
            seg._flow_threshold = 0.4
            seg._cellprob_threshold = 0.0
            seg._channels = [0, 0]
            params = seg.get_default_params(project)

        assert params["diameter"] == 25
        assert params["channels"] == [0, 1]


# ── Model selection ────────────────────────────────────────────────────────────


class TestSelectSegmenter:
    def test_no_project_returns_cellpose(self) -> None:
        """select_segmenter(None) returns CellPoseSegmenter when cellpose is available."""
        from unittest.mock import MagicMock, patch

        with patch("microagent.core.segment._HAS_CELLPOSE", True), patch(
            "microagent.core.segment._cp_models"
        ) as mock_cp:
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod
            from importlib import reload

            reload(seg_mod)
            seg_mod._HAS_CELLPOSE = True
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(None)
        assert result.__class__.__name__ == "CellPoseSegmenter"

    def test_nuclei_fluorescence_prefers_cellpose(self) -> None:
        """Nuclei + fluorescence → CellPose when cellpose is available."""
        from unittest.mock import MagicMock, patch

        project = {"imaging": {"segmentation_target": "nuclei", "staining": "fluorescence"}}
        with patch("microagent.core.segment._HAS_CELLPOSE", True), patch(
            "microagent.core.segment._cp_models"
        ) as mock_cp:
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            seg_mod._HAS_CELLPOSE = True
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(project)
        assert result.__class__.__name__ == "CellPoseSegmenter"

    def test_nuclei_he_prefers_stardist(self) -> None:
        """Nuclei + H&E → StarDist when stardist is available."""
        from unittest.mock import MagicMock, patch

        project = {"imaging": {"segmentation_target": "nuclei", "staining": "h&e"}}
        import microagent.core.segment as seg_mod

        mock_sd_cls = MagicMock()
        mock_sd_cls.from_pretrained.return_value = MagicMock()
        mock_cp = MagicMock()
        mock_cp.CellposeModel.return_value = MagicMock()

        orig_has_sd = seg_mod._HAS_STARDIST
        orig_has_cp = seg_mod._HAS_CELLPOSE
        orig_sd = getattr(seg_mod, "_StarDist2D", None)
        orig_cp = seg_mod._cp_models
        try:
            seg_mod._HAS_STARDIST = True
            seg_mod._HAS_CELLPOSE = True
            seg_mod._StarDist2D = mock_sd_cls
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(project)
        finally:
            seg_mod._HAS_STARDIST = orig_has_sd
            seg_mod._HAS_CELLPOSE = orig_has_cp
            if orig_sd is None:
                # Remove the attribute we injected
                try:
                    delattr(seg_mod, "_StarDist2D")
                except AttributeError:
                    pass
            else:
                seg_mod._StarDist2D = orig_sd
            seg_mod._cp_models = orig_cp

        assert result.__class__.__name__ == "StarDistSegmenter"
        assert result._model_name == "2D_versatile_he"

    def test_whole_cells_returns_cellpose(self) -> None:
        """whole_cells target → CellPose."""
        from unittest.mock import MagicMock, patch

        project = {"imaging": {"segmentation_target": "whole_cells", "staining": "fluorescence"}}
        with patch("microagent.core.segment._HAS_CELLPOSE", True), patch(
            "microagent.core.segment._cp_models"
        ) as mock_cp:
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            seg_mod._HAS_CELLPOSE = True
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(project)
        assert result.__class__.__name__ == "CellPoseSegmenter"

    def test_he_falls_back_to_cellpose_when_no_stardist(self) -> None:
        """Nuclei + H&E falls back to CellPose when StarDist is not installed."""
        from unittest.mock import MagicMock, patch

        project = {"imaging": {"segmentation_target": "nuclei", "staining": "h&e"}}
        with (
            patch("microagent.core.segment._HAS_STARDIST", False),
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_cp,
        ):
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            seg_mod._HAS_STARDIST = False
            seg_mod._HAS_CELLPOSE = True
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(project)
        assert result.__class__.__name__ == "CellPoseSegmenter"


# ── StarDist unavailability ────────────────────────────────────────────────────


class TestStarDistUnavailable:
    def test_raises_import_error_when_not_installed(self) -> None:
        """StarDistSegmenter raises ImportError with clear message if stardist is absent."""
        from unittest.mock import patch

        with patch("microagent.core.segment._HAS_STARDIST", False):
            import microagent.core.segment as seg_mod

            seg_mod._HAS_STARDIST = False
            with pytest.raises(ImportError, match="stardist"):
                seg_mod.StarDistSegmenter()


# ── run_segmentation integration tests ────────────────────────────────────────


class TestRunSegmentation:
    @pytest.mark.slow
    def test_produces_tiff_masks(self, multi_image_dir: Path, tmp_path: Path) -> None:
        """run_segmentation() writes a .tif mask per input image."""
        from microagent.core.segment import run_segmentation

        out_dir = tmp_path / "masks"
        result = run_segmentation(multi_image_dir, out_dir, model="cellpose")

        assert len(result.mask_paths) == 3
        for p in result.mask_paths:
            assert Path(p).exists(), f"Mask not written: {p}"
            assert Path(p).suffix.lower() == ".tif"

    @pytest.mark.slow
    def test_masks_are_int32_tiffs(self, single_image_dir: Path, tmp_path: Path) -> None:
        """Output masks are 32-bit labeled TIFFs."""
        from microagent.core.segment import run_segmentation

        out_dir = tmp_path / "masks"
        result = run_segmentation(single_image_dir, out_dir, model="cellpose")

        mask = tifffile.imread(result.mask_paths[0])
        assert mask.dtype == np.int32
        assert mask.ndim == 2

    @pytest.mark.slow
    def test_metadata_json_written(self, single_image_dir: Path, tmp_path: Path) -> None:
        """segmentation_metadata.json is written alongside masks."""
        from microagent.core.segment import run_segmentation

        out_dir = tmp_path / "masks"
        run_segmentation(single_image_dir, out_dir, model="cellpose")

        meta = out_dir / "segmentation_metadata.json"
        assert meta.exists()
        data = json.loads(meta.read_text())
        assert "mask_paths" in data
        assert "model_info" in data
        assert "per_image_stats" in data

    def test_missing_image_dir_raises(self, tmp_path: Path) -> None:
        """run_segmentation raises FileNotFoundError for non-existent directory."""
        from microagent.core.segment import run_segmentation

        with pytest.raises(FileNotFoundError):
            run_segmentation(tmp_path / "nonexistent", tmp_path / "out")

    def test_empty_image_dir_raises(self, tmp_path: Path) -> None:
        """run_segmentation raises RuntimeError when no images are found."""
        from microagent.core.segment import run_segmentation

        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(RuntimeError, match="No images found"):
            run_segmentation(empty, tmp_path / "out")

    @pytest.mark.slow
    def test_per_image_stats_populated(self, multi_image_dir: Path, tmp_path: Path) -> None:
        """per_image_stats has one entry per image with expected fields."""
        from microagent.core.segment import run_segmentation

        result = run_segmentation(multi_image_dir, tmp_path / "masks", model="cellpose")
        assert len(result.per_image_stats) == 3
        for stat in result.per_image_stats:
            assert stat.filename.endswith(".tif")
            assert stat.elapsed_seconds >= 0
            assert stat.n_labels >= 0


# ── CLI tests ──────────────────────────────────────────────────────────────────


runner = CliRunner()


class TestSegmentCLI:
    @pytest.mark.slow
    def test_segment_cli_cellpose(self, multi_image_dir: Path, tmp_path: Path) -> None:
        """CLI segment subcommand exits 0 and writes masks when using cellpose."""
        out_dir = tmp_path / "masks"
        result = runner.invoke(
            app,
            ["segment", str(multi_image_dir), "--output", str(out_dir), "--model", "cellpose"],
        )
        assert result.exit_code == 0, result.output
        assert "masks saved" in result.output.lower()

    def test_segment_cli_missing_dir(self, tmp_path: Path) -> None:
        """CLI returns exit code 1 when image directory does not exist."""
        result = runner.invoke(
            app,
            ["segment", str(tmp_path / "does_not_exist"), "--output", str(tmp_path / "out")],
        )
        assert result.exit_code == 1

    def test_segment_help(self) -> None:
        """segment --help exits 0 and mentions key options."""
        import re

        result = runner.invoke(app, ["segment", "--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes before checking for option names
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--model" in plain
        assert "--output" in plain

    @pytest.mark.slow
    def test_segment_cli_with_diameter(self, single_image_dir: Path, tmp_path: Path) -> None:
        """--diameter flag is accepted and forwarded to the model."""
        out_dir = tmp_path / "masks"
        result = runner.invoke(
            app,
            [
                "segment",
                str(single_image_dir),
                "--output",
                str(out_dir),
                "--model",
                "cellpose",
                "--diameter",
                "30",
            ],
        )
        assert result.exit_code == 0, result.output

    @pytest.mark.slow
    def test_segment_cli_shows_table(self, single_image_dir: Path, tmp_path: Path) -> None:
        """CLI output includes the per-image results table."""
        out_dir = tmp_path / "masks"
        result = runner.invoke(
            app,
            ["segment", str(single_image_dir), "--output", str(out_dir), "--model", "cellpose"],
        )
        assert result.exit_code == 0, result.output
        assert "Segmentation Results" in result.output
