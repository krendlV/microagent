"""Tests for the HTML report generator (viz/report.py) and CLI report command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from microagent.cli import app
from microagent.fair.provenance import RunMetadata
from microagent.viz.report import ReportData, generate_report

runner = CliRunner()


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_provenance() -> RunMetadata:
    return RunMetadata(
        microagent_version="0.1.0",
        python_version="3.10.0",
        platform="Linux-test",
        cellpose_version=None,
        stardist_version=None,
        torch_version="2.0.0",
        numpy_version=np.__version__,
        cuda_version=None,
        gpu_name=None,
        gpu_vram_mb=None,
        cpu_model="x86_64-test",
        ram_total_gb=16.0,
        data_hash="abc123",
        parameters={"model": "cyto3"},
        random_seed=42,
        timestamp_utc="2026-03-23T10:00:00+00:00",
        wall_clock_seconds=1.5,
        git_commit="deadbeef",
        git_dirty=False,
        command="test",
    )


def _make_inspection_dict() -> dict:
    return {
        "file_count": 3,
        "file_paths": ["a.tif", "b.tif", "c.tif"],
        "dimensions": [[2, 256, 256]],
        "dtypes": ["uint16"],
        "channel_count": 2,
        "intensity_stats": [
            {"channel": 0, "min": 0.0, "max": 65535.0, "mean": 1200.5, "std": 3400.2},
            {"channel": 1, "min": 0.0, "max": 65535.0, "mean": 900.1, "std": 2100.7},
        ],
        "issues": ["Near-zero image: c.tif"],
        "thumbnail_paths": [],
    }


def _make_segmentation_dict() -> dict:
    return {
        "mask_paths": ["masks/a.tif", "masks/b.tif", "masks/c.tif"],
        "model_info": {"backend": "cellpose", "model_name": "cyto3", "parameters": {}},
        "parameters": {"diameter": 30},
        "elapsed_seconds": 12.34,
        "per_image_stats": [
            {"filename": "a.tif", "n_labels": 10, "elapsed_seconds": 4.1},
            {"filename": "b.tif", "n_labels": 8, "elapsed_seconds": 3.9},
            {"filename": "c.tif", "n_labels": 12, "elapsed_seconds": 4.3},
        ],
    }


def _make_evaluation_dict() -> dict:
    return {
        "per_image": [
            {
                "filename": "a.tif",
                "gt_count": 10,
                "pred_count": 10,
                "per_threshold": [
                    {"threshold": 0.5, "precision": 0.9, "recall": 0.85, "f1": 0.874,
                     "tp": 8, "fp": 1, "fn": 2, "mean_true_score": 0.88},
                ],
                "mean_f1": 0.82,
                "panoptic_quality": 0.78,
                "iou_distribution": [],
            },
        ],
        "summary": {
            "n_images": 1,
            "per_threshold": [
                {"threshold": 0.5, "precision": 0.9, "recall": 0.85, "f1": 0.874,
                 "tp": 8, "fp": 1, "fn": 2, "mean_true_score": 0.88},
            ],
            "mean_f1": 0.82,
            "panoptic_quality": 0.78,
            "mean_gt_count": 10.0,
            "mean_pred_count": 10.0,
        },
        "best_images": ["a.tif"],
        "worst_images": [],
        "unmatched_preds": [],
        "unmatched_gts": [],
        "comparison": None,
    }


def _make_optimization_dict() -> dict:
    return {
        "best_params": {"diameter": 28, "flow_threshold": 0.4},
        "best_value": 0.875,
        "baseline_value": 0.820,
        "improvement": 0.055,
        "trials": [
            {"number": 0, "params": {"diameter": 25, "flow_threshold": 0.3}, "value": 0.82},
            {"number": 1, "params": {"diameter": 28, "flow_threshold": 0.4}, "value": 0.875},
        ],
        "study_path": None,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_report_generates_html(tmp_path: Path) -> None:
    """Output is valid HTML containing expected section headings."""
    output = tmp_path / "report.html"
    data = ReportData(
        project={"name": "TestProject", "description": "Unit test"},
        inspection=_make_inspection_dict(),
        segmentation=_make_segmentation_dict(),
        evaluation=_make_evaluation_dict(),
        optimization=_make_optimization_dict(),
        provenance=_make_provenance(),
    )
    generate_report(data, output)

    assert output.exists()
    html = output.read_text(encoding="utf-8")

    # Basic HTML structure
    assert "<!DOCTYPE html>" in html
    assert "<html" in html
    assert "</html>" in html

    # Expected section headings
    assert "Data Summary" in html
    assert "Segmentation Results" in html
    assert "Metrics Dashboard" in html
    assert "Optimization Summary" in html
    assert "Reproducibility" in html

    # Project info
    assert "TestProject" in html

    # Provenance values
    assert "0.1.0" in html
    assert "x86_64-test" in html


def test_report_images_embedded(tmp_path: Path) -> None:
    """PNG files are embedded as base64 data URIs (no external file references)."""
    # Create a tiny PNG-like file (1x1 white pixel PNG)
    import struct
    import zlib

    def _tiny_png() -> bytes:
        """Generate a minimal valid 1x1 white PNG."""
        sig = b"\x89PNG\r\n\x1a\n"

        def chunk(name: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + name + data
            return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        raw = b"\x00\xFF\xFF\xFF"
        idat = chunk(b"IDAT", zlib.compress(raw))
        iend = chunk(b"IEND", b"")
        return sig + ihdr + idat + iend

    overlay_dir = tmp_path / "overlays"
    overlay_dir.mkdir()
    (overlay_dir / "img_001_overlay.png").write_bytes(_tiny_png())

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    (plots_dir / "metrics.png").write_bytes(_tiny_png())

    output = tmp_path / "report.html"
    data = ReportData(
        project={},
        inspection=_make_inspection_dict(),
        segmentation=_make_segmentation_dict(),
        provenance=_make_provenance(),
        overlay_images=list(overlay_dir.glob("*.png")),
        plots=list(plots_dir.glob("*.png")),
    )
    generate_report(data, output)

    html = output.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "Overlay Gallery" in html
    assert "Metric Plots" in html


def test_report_no_evaluation(tmp_path: Path) -> None:
    """Report renders successfully without evaluation or optimization data."""
    output = tmp_path / "report.html"
    data = ReportData(
        project={"name": "Partial"},
        inspection=_make_inspection_dict(),
        segmentation=_make_segmentation_dict(),
        evaluation=None,
        optimization=None,
        provenance=_make_provenance(),
    )
    generate_report(data, output)

    assert output.exists()
    html = output.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Data Summary" in html
    assert "Segmentation Results" in html
    # Optional sections should be absent
    assert "Metrics Dashboard" not in html
    assert "Optimization Summary" not in html


def test_report_provenance_collect() -> None:
    """RunMetadata.collect() returns a populated instance."""
    meta = RunMetadata.collect(command="pytest", seed=0)
    assert meta.microagent_version == "0.1.0"
    assert meta.python_version.startswith("3.")
    assert meta.numpy_version
    assert meta.command == "pytest"
    assert meta.random_seed == 0
    assert meta.timestamp_utc  # non-empty


def test_report_cli(tmp_path: Path) -> None:
    """CLI 'report' command produces an HTML file."""
    # Write minimal JSON result files
    insp = tmp_path / "inspection.json"
    insp.write_text(json.dumps(_make_inspection_dict()), encoding="utf-8")

    seg = tmp_path / "segmentation.json"
    seg.write_text(json.dumps(_make_segmentation_dict()), encoding="utf-8")

    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            "--inspection", str(insp),
            "--segmentation", str(seg),
            "--output", str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Data Summary" in html
