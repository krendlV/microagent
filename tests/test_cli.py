"""Smoke tests for the microagent CLI."""
import json
from pathlib import Path

from typer.testing import CliRunner

from microagent.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "microagent" in result.output.lower()


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
