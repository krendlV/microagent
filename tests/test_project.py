"""Tests for project schema, knowledge module, and CLI init command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

from microagent.project.knowledge import (
    auto_detect_from_directory,
    extract_from_text,
    load_document,
    load_project,
    recommend_model,
    save_project,
)
from microagent.project.schema import ChannelConfig, ComputeConfig, ProjectConfig


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_project(**overrides) -> ProjectConfig:
    defaults = dict(
        name="test",
        organism="mouse",
        sample_type="cell_culture",
        modality="fluorescence",
        structures=["whole_cells"],
        channels=[ChannelConfig(name="GFP", index=0, target="whole_cells")],
        image_format="tiff",
        bit_depth=16,
        has_ground_truth=False,
        analysis_goal="segment",
        data_dir=Path("."),
    )
    defaults.update(overrides)
    return ProjectConfig(**defaults)


# ── test_project_roundtrip ────────────────────────────────────────────────────


def test_project_roundtrip(tmp_path: Path) -> None:
    """Save and reload a ProjectConfig and verify all fields are identical."""
    project = _make_project(
        name="roundtrip",
        organism="zebrafish",
        modality="confocal",
        structures=["nuclei", "membrane"],
        channels=[
            ChannelConfig(name="DAPI", index=0, target="nuclei"),
            ChannelConfig(name="GFP", index=1, target="membrane"),
        ],
        typical_dimensions=(512, 512),
        bit_depth=16,
        has_ground_truth=True,
        ground_truth_format="masks",
        gt_dir=tmp_path / "gt",
        compute=ComputeConfig(gpu_model="RTX 3090", vram_gb=24.0, ram_gb=64.0),
        recommended_model="stardist",
        recommended_params={"model_name": "2D_versatile_fluo"},
        data_dir=tmp_path / "images",
    )

    yaml_path = tmp_path / "project.yaml"
    save_project(project, yaml_path)
    loaded = load_project(yaml_path)

    assert loaded.name == project.name
    assert loaded.organism == project.organism
    assert loaded.modality == project.modality
    assert loaded.structures == project.structures
    assert len(loaded.channels) == 2
    assert loaded.channels[0].name == "DAPI"
    assert loaded.channels[1].target == "membrane"
    assert loaded.typical_dimensions == (512, 512)
    assert loaded.bit_depth == 16
    assert loaded.has_ground_truth is True
    assert loaded.ground_truth_format == "masks"
    assert loaded.gt_dir == project.gt_dir
    assert loaded.compute.gpu_model == "RTX 3090"
    assert loaded.compute.vram_gb == 24.0
    assert loaded.recommended_model == "stardist"
    assert loaded.recommended_params == {"model_name": "2D_versatile_fluo"}


# ── test_project_defaults ─────────────────────────────────────────────────────


def test_project_defaults(tmp_path: Path) -> None:
    """Missing optional fields should receive sensible defaults when loaded."""
    minimal_yaml = tmp_path / "project.yaml"
    minimal_yaml.write_text("name: minimal\n")

    project = load_project(minimal_yaml)

    assert project.name == "minimal"
    assert project.modality == "fluorescence"
    assert project.bit_depth == 16
    assert project.analysis_goal == "segment"
    assert project.has_ground_truth is False
    assert project.gt_dir is None
    assert project.typical_dimensions is None
    assert project.recommended_model == ""
    assert project.recommended_params == {}
    assert project.compute.gpu_model is None


# ── test_model_recommendation ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides,expected_model,expected_param_key,expected_param_value",
    [
        # EM → micro-SAM organelle/EM model
        (
            {"modality": "EM", "structures": ["whole_cells"]},
            "micro_sam",
            "model_type",
            "vit_b_em_organelles",
        ),
        # organelles → micro-SAM organelle/EM model
        (
            {"modality": "fluorescence", "structures": ["organelles"]},
            "micro_sam",
            "model_type",
            "vit_b_em_organelles",
        ),
        # H&E → stardist versatile_he
        (
            {"modality": "H&E", "structures": ["nuclei"]},
            "stardist",
            "model_name",
            "2D_versatile_he",
        ),
        # nuclei only → stardist versatile_fluo
        (
            {"modality": "fluorescence", "structures": ["nuclei"]},
            "stardist",
            "model_name",
            "2D_versatile_fluo",
        ),
        # phase_contrast → cellpose cyto2
        (
            {"modality": "phase_contrast", "structures": ["whole_cells"]},
            "cellpose",
            "model_name",
            "cyto2",
        ),
        # brightfield → cellpose cyto2
        (
            {"modality": "brightfield", "structures": ["whole_cells"]},
            "cellpose",
            "model_name",
            "cyto2",
        ),
        # confocal + whole_cells → cellpose cpsam
        (
            {"modality": "confocal", "structures": ["whole_cells"]},
            "cellpose",
            "model_name",
            "cpsam",
        ),
        # fluorescence + whole_cells → cellpose cyto3
        (
            {"modality": "fluorescence", "structures": ["whole_cells"]},
            "cellpose",
            "model_name",
            "cyto3",
        ),
        # low VRAM → cellpose cyto2
        (
            {
                "modality": "fluorescence",
                "structures": ["whole_cells"],
                "compute": ComputeConfig(vram_gb=2.0),
            },
            "cellpose",
            "model_name",
            "cyto2",
        ),
    ],
)
def test_model_recommendation(
    overrides: dict,
    expected_model: str,
    expected_param_key: str,
    expected_param_value: str,
) -> None:
    project = _make_project(**overrides)
    model, params = recommend_model(project)
    assert model == expected_model
    assert params.get(expected_param_key) == expected_param_value


def test_recommend_model_updates_project() -> None:
    """recommend_model return values can be assigned back to project fields."""
    project = _make_project(modality="EM", structures=["whole_cells"])
    model, params = recommend_model(project)
    project.recommended_model = model
    project.recommended_params = params
    assert project.recommended_model == "micro_sam"
    assert project.recommended_params["model_type"] == "vit_b_em_organelles"


# ── test_auto_detect ──────────────────────────────────────────────────────────


def test_auto_detect_tiff(tmp_path: Path) -> None:
    """auto_detect_from_directory should detect tiff format and dimensions."""
    img = np.zeros((64, 64), dtype=np.uint16)
    tifffile.imwrite(str(tmp_path / "image.tif"), img)

    result = auto_detect_from_directory(tmp_path)

    assert result["format"] in ("tiff", "ome-tiff")
    assert result["dimensions"] == (64, 64)
    assert result["dtype"] == "uint16"
    assert result["channel_count"] == 1


def test_auto_detect_multichannel(tmp_path: Path) -> None:
    """auto_detect_from_directory should detect multi-channel images."""
    img = np.zeros((3, 64, 64), dtype=np.uint8)
    tifffile.imwrite(str(tmp_path / "multi.tif"), img)

    result = auto_detect_from_directory(tmp_path)

    assert result["format"] in ("tiff", "ome-tiff")
    assert result["channel_count"] == 3


def test_auto_detect_empty_dir(tmp_path: Path) -> None:
    """auto_detect_from_directory should return all-None dict for empty dir."""
    result = auto_detect_from_directory(tmp_path)
    assert all(v is None for v in result.values())


def test_auto_detect_nonexistent(tmp_path: Path) -> None:
    """auto_detect_from_directory on a non-image dir returns all-None."""
    result = auto_detect_from_directory(tmp_path / "nonexistent")
    assert all(v is None for v in result.values())


# ── test_load_document ────────────────────────────────────────────────────────


def test_load_document_text(tmp_path: Path) -> None:
    """load_document reads plain-text files correctly."""
    doc = tmp_path / "brief.txt"
    doc.write_text("Mouse confocal nuclei project.")
    assert load_document(doc) == "Mouse confocal nuclei project."


def test_load_document_markdown(tmp_path: Path) -> None:
    """load_document reads markdown files correctly."""
    doc = tmp_path / "readme.md"
    doc.write_text("# Project\nFluorescence imaging of human cells.")
    text = load_document(doc)
    assert "human" in text
    assert "Fluorescence" in text


# ── test_extract_from_text ────────────────────────────────────────────────────


def test_keyword_extract_organism(monkeypatch) -> None:
    """Keyword fallback detects organism from plain text."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("We imaged zebrafish embryos with confocal microscopy.")
    assert result.get("organism") == "zebrafish"


def test_keyword_extract_modality(monkeypatch) -> None:
    """Keyword fallback detects modality from plain text."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("Brightfield imaging of HeLa cells.")
    assert result.get("modality") == "brightfield"


def test_keyword_extract_structures(monkeypatch) -> None:
    """Keyword fallback detects structures."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("We segment nuclei and whole cells from the images.")
    assert "nuclei" in result.get("structures", [])
    assert "whole_cells" in result.get("structures", [])


def test_keyword_extract_goal(monkeypatch) -> None:
    """Keyword fallback detects analysis goal."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("The goal is to count cells in each frame.")
    assert result.get("analysis_goal") == "count"


def test_keyword_extract_ground_truth(monkeypatch) -> None:
    """Keyword fallback sets has_ground_truth when annotations are mentioned."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("We have ground truth masks for each image.")
    assert result.get("has_ground_truth") is True
    assert result.get("ground_truth_format") == "masks"


def test_keyword_extract_bit_depth(monkeypatch) -> None:
    """Keyword fallback detects bit depth."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("Images are 16-bit TIFF files.")
    assert result.get("bit_depth") == 16


def test_keyword_extract_vram(monkeypatch) -> None:
    """Keyword fallback extracts VRAM from text."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_from_text("The server has 24GB VRAM and 128GB RAM.")
    assert result.get("compute", {}).get("vram_gb") == 24.0


def test_llm_extract_uses_api_key(monkeypatch, tmp_path) -> None:
    """extract_from_text calls anthropic when ANTHROPIC_API_KEY is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='{"organism": "mouse", "modality": "confocal"}')]

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = fake_client

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        result = extract_from_text("Mouse confocal project.")

    assert result.get("organism") == "mouse"
    assert result.get("modality") == "confocal"
    fake_client.messages.create.assert_called_once()


def test_llm_extract_falls_back_on_error(monkeypatch) -> None:
    """extract_from_text falls back to keywords when API raises."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.side_effect = Exception("network error")

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        result = extract_from_text("We image mouse nuclei with confocal.")

    # Fallback keyword extraction should still work
    assert result.get("organism") == "mouse"
