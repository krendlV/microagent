"""Tests for microagent.core.inspect."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from microagent.cli import app
from microagent.core.inspect import InspectionReport, inspect_directory

runner = CliRunner()


class TestInspectBasic:
    def test_file_count(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert report.file_count == 5

    def test_consistent_dimensions(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        # All 5 images are the same shape → only one unique dimension
        assert len(report.dimensions) == 1
        assert report.dimensions[0] == [2, 256, 256]

    def test_consistent_dtype(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert report.dtypes == ["uint16"]

    def test_channel_count(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert report.channel_count == 2

    def test_intensity_stats_populated(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        # One ChannelStats entry per channel
        assert len(report.intensity_stats) == 2
        for cs in report.intensity_stats:
            assert cs.min >= 0
            assert cs.max > cs.min
            assert cs.std >= 0

    def test_no_issues_on_clean_data(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert report.issues == []

    def test_thumbnail_created(self, tmp_image_dir, tmp_path):
        thumb_dir = tmp_path / "thumbs"
        report = inspect_directory(tmp_image_dir, thumbnail_dir=thumb_dir)
        assert len(report.thumbnail_paths) == 1
        assert Path(report.thumbnail_paths[0]).exists()
        assert not (tmp_image_dir / "microagent_inspection").exists()

    def test_no_thumbnail_without_dir(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert report.thumbnail_paths == []
        assert not (tmp_image_dir / "microagent_inspection").exists()

    def test_channel_filter(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir, channels=[0])
        assert len(report.intensity_stats) == 1
        assert report.intensity_stats[0].channel == 0

    def test_return_type(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert isinstance(report, InspectionReport)

    def test_file_paths_absolute(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        assert len(report.file_paths) == 5
        for fp in report.file_paths:
            assert Path(fp).is_absolute()


class TestInspectQCWarnings:
    def test_issues_non_empty(self, tmp_image_dir_bad):
        report = inspect_directory(tmp_image_dir_bad)
        assert len(report.issues) > 0

    def test_dtype_mismatch_detected(self, tmp_image_dir_bad):
        report = inspect_directory(tmp_image_dir_bad)
        dtype_issues = [i for i in report.issues if "dtype mismatch" in i.lower()]
        assert dtype_issues, f"Expected dtype mismatch warning, got: {report.issues}"

    def test_near_zero_detected(self, tmp_image_dir_bad):
        report = inspect_directory(tmp_image_dir_bad)
        zero_issues = [i for i in report.issues if "near-zero" in i.lower()]
        assert zero_issues, f"Expected near-zero warning, got: {report.issues}"

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        report = inspect_directory(empty)
        assert report.file_count == 0
        assert len(report.issues) > 0

    def test_single_file_warning(self, tmp_path):
        import tifffile
        import numpy as np

        one_dir = tmp_path / "one"
        one_dir.mkdir()
        tifffile.imwrite(one_dir / "only.tif", np.zeros((2, 64, 64), dtype=np.uint16) + 1000)
        report = inspect_directory(one_dir)
        single_issues = [i for i in report.issues if "1 image" in i.lower() or "only 1" in i.lower()]
        assert single_issues


class TestInspectJsonOutput:
    def test_to_dict_serialisable(self, tmp_image_dir):
        report = inspect_directory(tmp_image_dir)
        d = report.to_dict()
        # Must be JSON-serialisable (no sets, numpy types, etc.)
        serialised = json.dumps(d)
        assert isinstance(serialised, str)

    def test_json_round_trip(self, tmp_image_dir, tmp_path):
        report = inspect_directory(tmp_image_dir)
        out = tmp_path / "report.json"
        report.save_json(out)

        loaded = json.loads(out.read_text())
        assert loaded["file_count"] == report.file_count
        assert loaded["channel_count"] == report.channel_count
        assert loaded["dtypes"] == report.dtypes
        assert loaded["dimensions"] == report.dimensions

    def test_json_has_required_keys(self, tmp_image_dir, tmp_path):
        report = inspect_directory(tmp_image_dir)
        out = tmp_path / "report.json"
        report.save_json(out)

        data = json.loads(out.read_text())
        required = {
            "file_count",
            "file_paths",
            "dimensions",
            "dtypes",
            "channel_count",
            "intensity_stats",
            "issues",
            "thumbnail_paths",
        }
        assert required.issubset(data.keys())

    def test_intensity_stats_serialised(self, tmp_image_dir, tmp_path):
        report = inspect_directory(tmp_image_dir)
        out = tmp_path / "report.json"
        report.save_json(out)

        data = json.loads(out.read_text())
        assert isinstance(data["intensity_stats"], list)
        assert len(data["intensity_stats"]) == 2
        for cs in data["intensity_stats"]:
            assert {"channel", "min", "max", "mean", "std"}.issubset(cs.keys())


class TestInspectCLI:
    def test_exit_code_zero(self, tmp_image_dir):
        result = runner.invoke(app, ["inspect", str(tmp_image_dir)])
        assert result.exit_code == 0, result.output

    def test_output_contains_filenames(self, tmp_image_dir):
        result = runner.invoke(app, ["inspect", str(tmp_image_dir)])
        assert "image_000.tif" in result.output

    def test_json_output_flag(self, tmp_image_dir, tmp_path):
        out = tmp_path / "report.json"
        result = runner.invoke(app, ["inspect", str(tmp_image_dir), "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["file_count"] == 5

    def test_channels_flag(self, tmp_image_dir):
        result = runner.invoke(app, ["inspect", str(tmp_image_dir), "--channels", "0"])
        assert result.exit_code == 0, result.output

    def test_invalid_channels_flag(self, tmp_image_dir):
        result = runner.invoke(app, ["inspect", str(tmp_image_dir), "--channels", "abc"])
        assert result.exit_code != 0

    def test_qc_warnings_shown(self, tmp_image_dir_bad):
        result = runner.invoke(app, ["inspect", str(tmp_image_dir_bad)])
        assert result.exit_code == 0
        assert "QC" in result.output or "warn" in result.output.lower() or "mismatch" in result.output.lower()
