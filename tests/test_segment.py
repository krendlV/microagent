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

        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_models,
        ):
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

        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_models,
        ):
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
        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_models,
        ):
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

    def test_predict_warns_and_drops_channels_on_cellpose_v4(self) -> None:
        """CellPose v4 channels selection is surfaced instead of silently forwarded."""
        from unittest.mock import MagicMock, patch

        import microagent.core.segment as seg_mod

        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (
            np.ones((16, 16), dtype=np.int32),
            None,
            None,
        )

        seg = seg_mod.CellPoseSegmenter.__new__(seg_mod.CellPoseSegmenter)
        seg._model = cellpose_model
        seg._diameter = 30
        seg._flow_threshold = 0.4
        seg._cellprob_threshold = 0.0
        seg._channels = [0, 1]

        with (
            patch("microagent.core.cellpose_compat.version", return_value="4.0.9"),
            patch("microagent.core.cellpose_compat.console.print") as warn,
        ):
            mask = seg.predict(np.zeros((16, 16), dtype=np.uint16))

        assert mask.dtype == np.int32
        warn.assert_called_once()
        assert "channels argument" in warn.call_args.args[0]
        assert "channels" not in cellpose_model.eval.call_args.kwargs


# ── Unit tests for MicroSamSegmenter ──────────────────────────────────────────


class TestMicroSamSegmenter:
    def test_raises_import_error_when_not_installed(self) -> None:
        """MicroSamSegmenter raises ImportError with conda-forge install guidance."""
        from unittest.mock import patch

        with patch("microagent.core.segment._HAS_MICROSAM", False):
            import microagent.core.segment as seg_mod

            with pytest.raises(ImportError, match="conda-forge"):
                seg_mod.MicroSamSegmenter()

    def test_get_info_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_info() returns required keys without downloading a model."""
        import microagent.core.segment as seg_mod

        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", True)
        monkeypatch.setattr(
            seg_mod,
            "_microsam_get_predictor_and_segmenter",
            lambda **_kwargs: (object(), object()),
        )

        seg = seg_mod.MicroSamSegmenter(
            model_type="vit_b_em_organelles",
            segmentation_mode="auto",
            batch_size=2,
        )
        info = seg.get_info()

        assert info["backend"] == "micro_sam"
        assert info["model_name"] == "vit_b_em_organelles"
        assert info["parameters"]["segmentation_mode"] == "auto"
        assert info["parameters"]["batch_size"] == 2

    def test_predict_returns_int32_mask(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """predict() calls micro-SAM automatic segmentation and returns int32 labels."""
        from unittest.mock import MagicMock

        import microagent.core.segment as seg_mod

        predictor = object()
        segmenter = object()
        auto_seg = MagicMock(return_value=np.ones((16, 16), dtype=np.uint16))

        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", True)
        monkeypatch.setattr(
            seg_mod,
            "_microsam_get_predictor_and_segmenter",
            lambda **_kwargs: (predictor, segmenter),
        )
        monkeypatch.setattr(seg_mod, "_microsam_automatic_instance_segmentation", auto_seg)

        seg = seg_mod.MicroSamSegmenter(tile_shape=(8, 8), halo=(2, 2), min_size=5)
        mask = seg.predict(np.zeros((16, 16), dtype=np.uint16), batch_size=3)

        assert mask.dtype == np.int32
        assert mask.shape == (16, 16)
        assert auto_seg.call_args.kwargs["predictor"] is predictor
        assert auto_seg.call_args.kwargs["segmenter"] is segmenter
        assert auto_seg.call_args.kwargs["tile_shape"] == (8, 8)
        assert auto_seg.call_args.kwargs["halo"] == (2, 2)
        assert auto_seg.call_args.kwargs["batch_size"] == 3
        assert auto_seg.call_args.kwargs["min_size"] == 5

    def test_installed_micro_sam_import_smoke(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skip unless micro_sam is installed, then smoke-test construction with fakes."""
        pytest.importorskip("micro_sam")

        import microagent.core.segment as seg_mod

        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", True)
        monkeypatch.setattr(
            seg_mod,
            "_microsam_get_predictor_and_segmenter",
            lambda **_kwargs: (object(), object()),
        )

        seg = seg_mod.MicroSamSegmenter(model_type="vit_b_em_organelles")
        assert seg.get_info()["backend"] == "micro_sam"


# ── Model selection ────────────────────────────────────────────────────────────


class TestSelectSegmenter:
    def test_no_project_returns_cellpose(self) -> None:
        """select_segmenter(None) returns CellPoseSegmenter when cellpose is available."""
        from unittest.mock import MagicMock, patch

        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_cp,
        ):
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod
            from importlib import reload

            reload(seg_mod)
            seg_mod._HAS_CELLPOSE = True
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(None)
        assert result.__class__.__name__ == "CellPoseSegmenter"

    def test_nuclei_fluorescence_prefers_stardist(self) -> None:
        """Nuclei + fluorescence uses the shared project recommendation matrix."""
        from unittest.mock import MagicMock, patch

        project = {"imaging": {"segmentation_target": "nuclei", "staining": "fluorescence"}}
        import microagent.core.segment as seg_mod

        mock_sd_cls = MagicMock()
        mock_sd_cls.from_pretrained.return_value = MagicMock()
        with (
            patch("microagent.core.segment._HAS_STARDIST", True),
            patch("microagent.core.segment._StarDist2D", mock_sd_cls),
        ):
            result = seg_mod.select_segmenter(project)
        assert result.__class__.__name__ == "StarDistSegmenter"
        assert result._model_name == "2D_versatile_fluo"

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
        had_sd_attr = hasattr(seg_mod, "_StarDist2D")
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
            if not had_sd_attr:
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
        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_cp,
        ):
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            seg_mod._HAS_CELLPOSE = True
            seg_mod._cp_models = mock_cp
            result = seg_mod.select_segmenter(project)
        assert result.__class__.__name__ == "CellPoseSegmenter"
        assert result._model_name == "cyto3"

    def test_em_prefers_microsam_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EM projects select micro-SAM when the optional backend is installed."""
        import microagent.core.segment as seg_mod

        project = {"modality": "EM", "structures": ["whole_cells"]}
        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", True)
        monkeypatch.setattr(
            seg_mod,
            "_microsam_get_predictor_and_segmenter",
            lambda **_kwargs: (object(), object()),
        )

        result = seg_mod.select_segmenter(project)

        assert result.__class__.__name__ == "MicroSamSegmenter"
        assert result._model_type == "vit_b_em_organelles"

    def test_organelles_prefer_microsam_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Organelle targets select micro-SAM when the optional backend is installed."""
        import microagent.core.segment as seg_mod

        project = {"modality": "fluorescence", "structures": ["organelles"]}
        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", True)
        monkeypatch.setattr(
            seg_mod,
            "_microsam_get_predictor_and_segmenter",
            lambda **_kwargs: (object(), object()),
        )

        result = seg_mod.select_segmenter(project)

        assert result.__class__.__name__ == "MicroSamSegmenter"

    def test_em_falls_back_to_cellpose_when_no_microsam(self) -> None:
        """EM/organelle micro-SAM recommendations fall back to CellPose cpsam if absent."""
        from unittest.mock import MagicMock, patch

        project = {"modality": "EM", "structures": ["whole_cells"]}
        with (
            patch("microagent.core.segment._HAS_MICROSAM", False),
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models") as mock_cp,
        ):
            mock_cp.CellposeModel.return_value = MagicMock()
            import microagent.core.segment as seg_mod

            result = seg_mod.select_segmenter(project)

        assert result.__class__.__name__ == "CellPoseSegmenter"
        assert result._model_name == "cpsam"

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


# ── Poor-fit fallback warnings ────────────────────────────────────────────────


class TestPoorFitFallbackWarning:
    def test_warns_when_stardist_recommended_but_unavailable(
        self,
        single_image_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Warning is emitted when stardist is recommended but unavailable for H&E."""
        from unittest.mock import MagicMock

        import microagent.core.segment as seg_mod

        project = {
            "recommended_model": "stardist",
            "recommended_params": {"model_name": "2D_versatile_he"},
            "imaging": {"staining": "h&e", "segmentation_target": "nuclei"},
        }
        import yaml

        project_path = tmp_path / "project.yaml"
        project_path.write_text(yaml.safe_dump(project))

        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (np.zeros((256, 256), dtype=np.int32), None, None)
        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = cellpose_model

        warned: list[str] = []
        monkeypatch.setattr(seg_mod, "_HAS_STARDIST", False)
        monkeypatch.setattr(seg_mod, "_HAS_CELLPOSE", True)
        monkeypatch.setattr(seg_mod, "_cp_models", cellpose_mod)
        monkeypatch.setattr(
            "microagent.core.cellpose_compat.console.print",
            lambda msg, **_: warned.append(str(msg)),
        )

        result = seg_mod.run_segmentation(
            single_image_dir, tmp_path / "masks", project_path=project_path
        )

        assert result.model_info["backend"] == "cellpose", "Should have fallen back to cellpose"
        assert any("stardist" in w.lower() for w in warned), f"No stardist warning found: {warned}"
        assert any("poor fit" in w.lower() for w in warned), f"No 'poor fit' in warnings: {warned}"

    def test_no_warning_when_recommended_backend_is_used(
        self,
        single_image_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No fallback warning when the recommended backend is available."""
        from unittest.mock import MagicMock

        import microagent.core.segment as seg_mod

        project = {
            "recommended_model": "cellpose",
            "recommended_params": {"model_name": "cyto3"},
        }
        import yaml

        project_path = tmp_path / "project.yaml"
        project_path.write_text(yaml.safe_dump(project))

        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (np.zeros((256, 256), dtype=np.int32), None, None)
        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = cellpose_model

        warned: list[str] = []
        monkeypatch.setattr(seg_mod, "_HAS_CELLPOSE", True)
        monkeypatch.setattr(seg_mod, "_cp_models", cellpose_mod)
        monkeypatch.setattr(
            "microagent.core.cellpose_compat.console.print",
            lambda msg, **_: warned.append(str(msg)),
        )

        seg_mod.run_segmentation(
            single_image_dir, tmp_path / "masks", project_path=project_path
        )

        fallback_warns = [w for w in warned if "unavailable" in w.lower()]
        assert not fallback_warns, f"Unexpected fallback warning: {fallback_warns}"

    def test_warns_when_microsam_recommended_but_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Warning is emitted when micro-SAM is recommended for EM but unavailable."""
        import microagent.core.segment as seg_mod

        project = {"modality": "EM", "structures": ["organelles"]}
        from unittest.mock import MagicMock

        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = MagicMock()

        warned: list[str] = []
        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", False)
        monkeypatch.setattr(seg_mod, "_HAS_CELLPOSE", True)
        monkeypatch.setattr(seg_mod, "_cp_models", cellpose_mod)
        monkeypatch.setattr(
            "microagent.core.cellpose_compat.console.print",
            lambda msg, **_: warned.append(str(msg)),
        )

        result = seg_mod.select_segmenter(project)

        assert result.__class__.__name__ == "CellPoseSegmenter"
        assert any("micro" in w.lower() and "unavailable" in w.lower() for w in warned)


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
    def test_project_recommended_stardist_model_is_used(
        self,
        single_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """project.yaml recommended_model/recommended_params drive auto segmentation."""
        from unittest.mock import MagicMock, patch

        import yaml

        from microagent.core.segment import run_segmentation

        project_path = tmp_path / "project.yaml"
        project_path.write_text(
            yaml.safe_dump(
                {
                    "recommended_model": "stardist",
                    "recommended_params": {
                        "model_name": "2D_versatile_he",
                        "prob_thresh": 0.5,
                        "nms_thresh": 0.4,
                    },
                }
            )
        )
        stardist_model = MagicMock()
        stardist_model.predict_instances.return_value = (
            np.ones((256, 256), dtype=np.int32),
            {},
        )
        stardist_cls = MagicMock()
        stardist_cls.from_pretrained.return_value = stardist_model

        with (
            patch("microagent.core.segment._HAS_STARDIST", True),
            patch("microagent.core.segment._StarDist2D", stardist_cls),
        ):
            result = run_segmentation(
                single_image_dir,
                tmp_path / "masks",
                project_path=project_path,
            )

        stardist_cls.from_pretrained.assert_called_once_with("2D_versatile_he")
        assert result.model_info["backend"] == "stardist"
        assert result.model_info["model_name"] == "2D_versatile_he"
        assert result.parameters["prob_thresh"] == 0.5
        assert result.parameters["nms_thresh"] == 0.4

    def test_explicit_model_overrides_project_recommendation(
        self,
        single_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """An explicit model backend ignores project.yaml recommended_model."""
        from unittest.mock import MagicMock, patch

        import yaml

        from microagent.core.segment import run_segmentation

        project_path = tmp_path / "project.yaml"
        project_path.write_text(
            yaml.safe_dump(
                {
                    "recommended_model": "stardist",
                    "recommended_params": {"model_name": "2D_versatile_he"},
                }
            )
        )
        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (
            np.ones((256, 256), dtype=np.int32),
            None,
            None,
        )
        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = cellpose_model

        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models", cellpose_mod),
        ):
            result = run_segmentation(
                single_image_dir,
                tmp_path / "masks",
                model="cellpose",
                project_path=project_path,
            )

        cellpose_mod.CellposeModel.assert_called_once()
        assert result.model_info["backend"] == "cellpose"
        assert result.model_info["model_name"] == "cpsam"

    def test_explicit_diameter_overrides_project_recommendation_params(
        self,
        single_image_dir: Path,
        tmp_path: Path,
    ) -> None:
        """An explicit diameter overrides recommended CellPose diameter."""
        from unittest.mock import MagicMock, patch

        import yaml

        from microagent.core.segment import run_segmentation

        project_path = tmp_path / "project.yaml"
        project_path.write_text(
            yaml.safe_dump(
                {
                    "recommended_model": "cellpose",
                    "recommended_params": {
                        "model_name": "cyto2",
                        "diameter": 21,
                        "flow_threshold": 0.4,
                    },
                }
            )
        )
        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (
            np.ones((256, 256), dtype=np.int32),
            None,
            None,
        )
        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = cellpose_model

        with (
            patch("microagent.core.segment._HAS_CELLPOSE", True),
            patch("microagent.core.segment._cp_models", cellpose_mod),
        ):
            result = run_segmentation(
                single_image_dir,
                tmp_path / "masks",
                project_path=project_path,
                diameter=42,
            )

        assert result.model_info["model_name"] == "cyto2"
        assert result.parameters["diameter"] == 42
        assert cellpose_model.eval.call_args.kwargs["diameter"] == 42

    def test_explicit_microsam_backend_runs_with_mock(
        self,
        single_image_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_segmentation supports explicit micro_sam backend selection."""
        import microagent.core.segment as seg_mod

        monkeypatch.setattr(seg_mod, "_HAS_MICROSAM", True)
        monkeypatch.setattr(
            seg_mod,
            "_microsam_get_predictor_and_segmenter",
            lambda **_kwargs: (object(), object()),
        )
        monkeypatch.setattr(
            seg_mod,
            "_microsam_automatic_instance_segmentation",
            lambda **_kwargs: np.ones((256, 256), dtype=np.int32),
        )

        result = seg_mod.run_segmentation(
            single_image_dir,
            tmp_path / "masks",
            model="micro_sam",
            min_size=7,
        )

        assert result.model_info["backend"] == "micro_sam"
        assert result.parameters["min_size"] == 7

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
        """segmentation.json is written alongside masks."""
        from microagent.core.segment import run_segmentation

        out_dir = tmp_path / "masks"
        run_segmentation(single_image_dir, out_dir, model="cellpose")

        meta = out_dir / "segmentation.json"
        assert meta.exists()
        data = json.loads(meta.read_text())
        assert "mask_paths" in data
        assert "model_info" in data
        assert "per_image_stats" in data

    def test_n_labels_counts_objects_not_max_value(
        self,
        single_image_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """n_labels counts distinct non-background labels, not the max label value."""
        from unittest.mock import MagicMock

        import microagent.core.segment as seg_mod

        # Labels {0, 5, 9}: 2 objects. mask.max() would wrongly return 9.
        sparse_mask = np.zeros((256, 256), dtype=np.int32)
        sparse_mask[10:50, 10:50] = 5
        sparse_mask[100:150, 100:150] = 9

        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (sparse_mask, None, None)
        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = cellpose_model

        monkeypatch.setattr(seg_mod, "_HAS_CELLPOSE", True)
        monkeypatch.setattr(seg_mod, "_cp_models", cellpose_mod)

        result = seg_mod.run_segmentation(
            single_image_dir, tmp_path / "masks", model="cellpose"
        )

        assert len(result.per_image_stats) == 1
        assert result.per_image_stats[0].n_labels == 2

    def test_n_labels_zero_for_empty_mask(
        self,
        single_image_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An all-zero mask yields n_labels == 0, not -1."""
        from unittest.mock import MagicMock

        import microagent.core.segment as seg_mod

        zero_mask = np.zeros((256, 256), dtype=np.int32)

        cellpose_model = MagicMock()
        cellpose_model.eval.return_value = (zero_mask, None, None)
        cellpose_mod = MagicMock()
        cellpose_mod.CellposeModel.return_value = cellpose_model

        monkeypatch.setattr(seg_mod, "_HAS_CELLPOSE", True)
        monkeypatch.setattr(seg_mod, "_cp_models", cellpose_mod)

        result = seg_mod.run_segmentation(
            single_image_dir, tmp_path / "masks", model="cellpose"
        )

        assert result.per_image_stats[0].n_labels == 0

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
    def test_segment_cli_forwards_explicit_model_and_diameter(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLI --model and --diameter are forwarded as explicit overrides."""
        from microagent.core.segment import SegmentationResult
        import microagent.core.segment as seg_mod

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        project_path = tmp_path / "project.yaml"
        project_path.write_text("recommended_model: stardist\nrecommended_params: {}\n")
        captured: dict[str, object] = {}

        def fake_run_segmentation(**kwargs: object) -> SegmentationResult:
            captured.update(kwargs)
            return SegmentationResult(
                mask_paths=[],
                model_info={
                    "backend": "cellpose",
                    "model_name": "cpsam",
                    "parameters": {"diameter": kwargs.get("diameter")},
                },
                parameters={"diameter": kwargs.get("diameter")},
                elapsed_seconds=0.0,
            )

        monkeypatch.setattr(seg_mod, "run_segmentation", fake_run_segmentation)

        result = runner.invoke(
            app,
            [
                "--no-track",
                "segment",
                str(img_dir),
                "--output",
                str(tmp_path / "masks"),
                "--project",
                str(project_path),
                "--model",
                "cellpose",
                "--diameter",
                "42",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["model"] == "cellpose"
        assert captured["diameter"] == 42
        assert captured["project_path"] == project_path

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
        result = runner.invoke(app, ["segment", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--output" in result.output

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
