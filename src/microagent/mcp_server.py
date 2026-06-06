"""MCP server exposing MicroAgent functions to any MCP-compatible LLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP

    _HAS_MCP = True
except ImportError:  # pragma: no cover
    _HAS_MCP = False
    FastMCP = None  # type: ignore[assignment,misc]

if not _HAS_MCP:
    raise ImportError(
        "mcp is required for the MCP server. Install with: pip install 'microagent[mcp]'"
    )

mcp = FastMCP("MicroAgent", json_response=True)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def inspect_data(path: str) -> dict[str, Any]:
    """Inspect a directory of microscopy images. Returns QC report with
    file count, dimensions, intensity stats, and quality warnings."""
    try:
        from microagent.core.inspect import inspect_directory

        report = inspect_directory(Path(path))
        return {"status": "success", **report.to_dict()}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def segment(
    image_dir: str,
    output_dir: str = "masks",
    model: str = "auto",
    diameter: int | None = None,
    project: str | None = None,
    track: bool = True,
) -> dict[str, Any]:
    """Run segmentation on microscopy images. Models: auto, cellpose, stardist, micro_sam.
    Pass project= to apply project.yaml model selection. Returns mask paths, cell counts,
    model info, and run_id for reproducibility tracking."""
    try:
        from microagent.core.segment import run_segmentation

        kwargs: dict[str, Any] = {}
        if diameter is not None:
            kwargs["diameter"] = diameter

        project_path = Path(project) if project else None
        run_id: str | None = None

        if track:
            from microagent.fair.tracking import ExperimentTracker, tracked_run

            params = {
                "image_dir": image_dir,
                "output_dir": output_dir,
                "model": model,
                "diameter": diameter,
                "project": project,
            }
            with tracked_run(
                ExperimentTracker(),
                f"mcp:segment {image_dir}",
                params,
                data_path=Path(image_dir),
                log_on_exception=False,
            ) as tracked_results:
                result = run_segmentation(
                    image_dir=Path(image_dir),
                    output_dir=Path(output_dir),
                    model=model,
                    project_path=project_path,
                    **kwargs,
                )
                tracked_results.update({
                    "n_masks": len(result.mask_paths),
                    "n_objects": sum(s.n_labels for s in result.per_image_stats),
                    "elapsed_seconds": result.elapsed_seconds,
                    "output_dir": output_dir,
                    "backend": result.model_info.get("backend"),
                    "model_name": result.model_info.get("model_name"),
                })
            run_id = tracked_results.get("run_id")
        else:
            result = run_segmentation(
                image_dir=Path(image_dir),
                output_dir=Path(output_dir),
                model=model,
                project_path=project_path,
                **kwargs,
            )

        response: dict[str, Any] = {"status": "success", **result.to_dict()}
        if run_id is not None:
            response["run_id"] = run_id
        return response
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def evaluate(
    pred_dir: str,
    gt_dir: str,
    thresholds: str = "0.5,0.75,0.9",
    track: bool = True,
) -> dict[str, Any]:
    """Evaluate segmentation quality against ground truth masks.
    Returns precision, recall, F1, mean F1 across thresholds, panoptic quality
    per image and overall, and run_id for reproducibility tracking.
    """
    try:
        from dataclasses import asdict

        from microagent.core.evaluate import evaluate_masks

        thresh_list = [float(t.strip()) for t in thresholds.split(",")]
        run_id: str | None = None

        if track:
            from microagent.fair.tracking import ExperimentTracker, tracked_run

            params = {"pred_dir": pred_dir, "gt_dir": gt_dir, "thresholds": thresholds}
            with tracked_run(
                ExperimentTracker(),
                f"mcp:evaluate {pred_dir}",
                params,
                data_path=Path(pred_dir),
                log_on_exception=False,
            ) as tracked_results:
                result = evaluate_masks(Path(pred_dir), Path(gt_dir), thresholds=thresh_list)
                f1_at_05 = next(
                    (
                        tm.f1
                        for tm in result.summary.per_threshold
                        if abs(tm.threshold - 0.5) < 1e-9
                    ),
                    result.summary.per_threshold[0].f1 if result.summary.per_threshold else 0.0,
                )
                tracked_results.update({
                    "f1": f1_at_05,
                    "mean_f1": result.summary.mean_f1,
                    "panoptic_quality": result.summary.panoptic_quality,
                    "n_images": result.summary.n_images,
                    "mean_gt_count": result.summary.mean_gt_count,
                    "mean_pred_count": result.summary.mean_pred_count,
                })
            run_id = tracked_results.get("run_id")
        else:
            result = evaluate_masks(Path(pred_dir), Path(gt_dir), thresholds=thresh_list)

        response: dict[str, Any] = {"status": "success", **asdict(result)}
        if run_id is not None:
            response["run_id"] = run_id
        return response
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def train(
    image_dir: str,
    gt_dir: str,
    epochs: int = 100,
    output_dir: str = "models",
    track: bool = True,
) -> dict[str, Any]:
    """Fine-tune a CellPose model on your data. Returns model path, metrics,
    and run_id for reproducibility tracking."""
    try:
        from dataclasses import asdict

        from microagent.core.train import TrainConfig, prepare_data, train_cellpose

        train_data_dir, test_data_dir = prepare_data(
            image_dir=Path(image_dir),
            gt_dir=Path(gt_dir),
        )
        config = TrainConfig(
            train_dir=train_data_dir,
            test_dir=test_data_dir,
            n_epochs=epochs,
            save_dir=Path(output_dir),
        )
        run_id: str | None = None

        if track:
            from microagent.fair.tracking import ExperimentTracker, tracked_run

            params = {
                "image_dir": image_dir,
                "gt_dir": gt_dir,
                "epochs": epochs,
                "output_dir": output_dir,
            }
            with tracked_run(
                ExperimentTracker(),
                f"mcp:train {image_dir}",
                params,
                data_path=Path(image_dir),
                log_on_exception=False,
            ) as tracked_results:
                result = train_cellpose(config)
                losses = result.test_losses or result.train_losses
                tracked_results.update({
                    "model_path": str(result.model_path),
                    "best_epoch": result.best_epoch + 1,
                    "best_loss": min(losses) if losses else None,
                    "elapsed_seconds": result.elapsed_seconds,
                })
            run_id = tracked_results.get("run_id")
        else:
            result = train_cellpose(config)

        d = asdict(result)
        # Path objects are not JSON-serializable; convert them
        d["model_path"] = str(result.model_path)
        d["config_used"]["train_dir"] = str(result.config_used.train_dir)
        d["config_used"]["test_dir"] = (
            str(result.config_used.test_dir) if result.config_used.test_dir else None
        )
        d["config_used"]["save_dir"] = str(result.config_used.save_dir)
        response: dict[str, Any] = {"status": "success", **d}
        if run_id is not None:
            response["run_id"] = run_id
        return response
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def optimize(
    image_dir: str,
    gt_dir: str,
    n_trials: int = 20,
    metric: str = "f1",
    track: bool = True,
) -> dict[str, Any]:
    """Run hyperparameter optimization with Optuna. Returns best parameters,
    improvement over baseline, and run_id for reproducibility tracking."""
    try:
        from microagent.core.optimize import OptimizeConfig, run_optimization

        config = OptimizeConfig(
            image_dir=Path(image_dir),
            gt_dir=Path(gt_dir),
            n_trials=n_trials,
            metric=metric,
        )
        run_id: str | None = None

        if track:
            from microagent.fair.tracking import ExperimentTracker, tracked_run

            params = {
                "image_dir": image_dir,
                "gt_dir": gt_dir,
                "n_trials": n_trials,
                "metric": metric,
            }
            with tracked_run(
                ExperimentTracker(),
                f"mcp:optimize {image_dir}",
                params,
                data_path=Path(image_dir),
                log_on_exception=False,
            ) as tracked_results:
                result = run_optimization(config)
                tracked_results.update({
                    "best_params": result.best_params,
                    "best_value": result.best_value,
                    "baseline_value": result.baseline_value,
                    "improvement": result.improvement,
                    "n_trials": len(result.trials),
                    "study_path": str(result.study_path) if result.study_path else None,
                })
            run_id = tracked_results.get("run_id")
        else:
            result = run_optimization(config)

        response: dict[str, Any] = {
            "status": "success",
            "best_params": result.best_params,
            "best_value": result.best_value,
            "baseline_value": result.baseline_value,
            "improvement": result.improvement,
            "n_trials": len(result.trials),
            "study_path": str(result.study_path) if result.study_path else None,
        }
        if run_id is not None:
            response["run_id"] = run_id
        return response
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def generate_report(output: str = "report.html") -> dict[str, Any]:
    """Generate a self-contained HTML report of all analysis results."""
    try:
        from microagent.viz.report import generate_report as _gen_report
        from microagent.viz.report import load_report_data

        cwd = Path.cwd()
        data = load_report_data(
            inspection_json=_find_file(
                cwd,
                "inspection.json",
                "microagent_inspection/inspection.json",
            ),
            segmentation_json=_find_file(
                cwd,
                "segmentation.json",
                "masks/segmentation.json",
                "masks/segmentation_metadata.json",
            ),
            evaluation_json=_find_file(cwd, "metrics.json", "evaluation.json"),
            optimization_json=_find_file(cwd, "optimization.json"),
            project_yaml=_find_file(cwd, "project.yaml"),
            overlay_dir=_find_dir(cwd, "overlays"),
            plots_dir=_find_dir(cwd, "plots"),
            command="mcp:generate_report",
        )
        out_path = _gen_report(data, Path(output))
        return {"status": "success", "report_path": str(out_path)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def get_project_info() -> dict[str, Any]:
    """Read the current project.yaml and return project configuration."""
    try:
        import yaml  # type: ignore[import-untyped]

        project_path = Path.cwd() / "project.yaml"
        if not project_path.exists():
            return {
                "status": "error",
                "error": "project.yaml not found in current directory",
            }
        data = yaml.safe_load(project_path.read_text(encoding="utf-8")) or {}
        return {"status": "success", **data}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def create_project(
    organism: str,
    modality: str,
    structures: str,
    channels: str = "0,0",
    image_format: str = "tiff",
) -> dict[str, Any]:
    """Create a new project.yaml with the specified parameters."""
    try:
        import yaml  # type: ignore[import-untyped]

        ch_list = [int(c.strip()) for c in channels.split(",")]
        project = {
            "organism": organism,
            "modality": modality,
            "structures": [s.strip() for s in structures.split(",")],
            "imaging": {
                "format": image_format,
                "channels": {
                    "cytoplasm": ch_list[0],
                    "nucleus": ch_list[1] if len(ch_list) > 1 else 0,
                },
            },
        }
        out_path = Path.cwd() / "project.yaml"
        out_path.write_text(yaml.dump(project, default_flow_style=False), encoding="utf-8")
        return {"status": "success", "project_path": str(out_path), "project": project}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_file(base: Path, *candidates: str) -> Path | None:
    """Return the first existing candidate path, or None."""
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    return None


def _find_dir(base: Path, *candidates: str) -> Path | None:
    """Return the first existing candidate directory, or None."""
    for c in candidates:
        p = base / c
        if p.is_dir():
            return p
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# FastMCP instance for use as a library
app = mcp


def main() -> None:
    """Run the MCP server."""
    import sys

    from rich.console import Console

    console = Console(file=sys.stderr)
    console.print("[bold green]Starting MicroAgent MCP server[/bold green]")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
