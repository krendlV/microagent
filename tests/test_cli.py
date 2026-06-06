"""Smoke tests for the microagent CLI."""
import json
from pathlib import Path

import numpy as np
import tifffile
from typer.testing import CliRunner

from microagent.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "microagent" in result.output.lower()


def test_run_command_no_open(tmp_path, tmp_image_dir, monkeypatch):
    """run command: end-to-end on synthetic data, verifies output structure."""
    from microagent.core.segment import PerImageStats, SegmentationResult

    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "run_output"

    def fake_run_segmentation(image_dir, output_dir, model, project_path=None, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_paths = []
        for img_file in sorted(Path(image_dir).glob("*.tif")):
            mask_path = output_dir / f"{img_file.stem}_mask.tif"
            tifffile.imwrite(str(mask_path), np.zeros((256, 256), dtype=np.uint16))
            mask_paths.append(str(mask_path))
        return SegmentationResult(
            mask_paths=mask_paths,
            model_info={"backend": "cellpose", "model_name": "stub", "parameters": {}},
            parameters={},
            elapsed_seconds=0.1,
            per_image_stats=[
                PerImageStats(filename=f.name, n_labels=3, elapsed_seconds=0.1)
                for f in sorted(Path(image_dir).glob("*.tif"))
            ],
        )

    def fake_generate_report(data, path, **kwargs):
        Path(path).write_text("<html>stub</html>", encoding="utf-8")

    monkeypatch.setattr("microagent.core.segment.run_segmentation", fake_run_segmentation)
    monkeypatch.setattr("microagent.viz.report.generate_report", fake_generate_report)

    result = runner.invoke(
        app,
        [
            "--no-track",
            "run",
            str(tmp_image_dir),
            "--output",
            str(output_dir),
            "--no-open",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "masks").is_dir()
    assert (output_dir / "overlays").is_dir()
    assert (output_dir / "inspection.json").exists()
    assert (output_dir / "segmentation.json").exists()
    assert (output_dir / "report.html").exists()
    assert (output_dir / "report.html").read_text() == "<html>stub</html>"
    # No evaluation JSON without --ground-truth
    assert not (output_dir / "metrics.json").exists()
    # Summary message present
    assert "Done." in result.output


def test_segment_tracking_record_exports_bundle(tmp_path, monkeypatch):
    """segment appends experiments.jsonl and export resolves the logged run."""
    from microagent.core.segment import PerImageStats, SegmentationResult

    monkeypatch.chdir(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "img_000.tif").write_bytes(b"synthetic")

    def fake_run_segmentation(
        image_dir: Path,
        output_dir: Path,
        model: str,
        project_path: Path | None = None,
        **kwargs,
    ) -> SegmentationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / "img_000_mask.tif"
        mask_path.write_bytes(b"mask")
        return SegmentationResult(
            mask_paths=[str(mask_path)],
            model_info={
                "backend": model,
                "model_name": "stub-model",
                "parameters": kwargs,
            },
            parameters=kwargs,
            elapsed_seconds=1.25,
            per_image_stats=[
                PerImageStats(
                    filename="img_000.tif",
                    n_labels=3,
                    elapsed_seconds=1.25,
                )
            ],
        )

    def fake_environment_lock(output_path: Path) -> Path:
        output_path.write_text("stub==1.0\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "microagent.core.segment.run_segmentation",
        fake_run_segmentation,
    )
    monkeypatch.setattr(
        "microagent.fair.container.generate_environment_lock",
        fake_environment_lock,
    )

    segment_result = runner.invoke(
        app,
        [
            "segment",
            str(image_dir),
            "--output",
            str(tmp_path / "masks"),
            "--model",
            "cellpose",
        ],
    )

    assert segment_result.exit_code == 0, segment_result.output
    experiments = tmp_path / "experiments.jsonl"
    assert experiments.exists()
    lines = experiments.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    run_id = record["run_id"]
    assert len(run_id) == 8
    assert record["results"]["n_masks"] == 1
    assert record["results"]["n_objects"] == 3
    assert f"run {run_id} logged" in segment_result.output

    export_result = runner.invoke(
        app,
        [
            "export",
            "--run",
            run_id,
            "--format",
            "bundle",
            "--output",
            str(tmp_path / "export"),
        ],
    )

    assert export_result.exit_code == 0, export_result.output
    assert list((tmp_path / "export").glob("*.zip"))
