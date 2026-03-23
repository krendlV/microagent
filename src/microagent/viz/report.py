"""Self-contained HTML report generation for MicroAgent pipeline results."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReportData:
    """All data needed to render a self-contained HTML report.

    Parameters
    ----------
    project : dict
        Parsed contents of project.yaml (or empty dict if not available).
    inspection : any
        InspectionReport dataclass or equivalent dict, or None.
    segmentation : any
        SegmentationResult dataclass or equivalent dict, or None.
    evaluation : any
        EvaluationResult dataclass or equivalent dict, or None.
    optimization : any
        OptimizationResult dataclass or equivalent dict, or None.
    provenance : any
        RunMetadata dataclass or equivalent dict.
    overlay_images : list[Path]
        Paths to overlay PNG files to embed in the gallery section.
    plots : list[Path]
        Paths to metric plot PNG files to embed in the charts section.
    """

    project: dict[str, Any]
    inspection: Any  # InspectionReport or dict or None
    segmentation: Any  # SegmentationResult or dict or None
    evaluation: Any | None = None  # EvaluationResult or dict or None
    optimization: Any | None = None  # OptimizationResult or dict or None
    provenance: Any = None  # RunMetadata or dict
    overlay_images: list[Path] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_dict(obj: Any) -> Any:
    """Convert a dataclass (or plain dict) to a plain dict, recursively."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    try:
        return asdict(obj)
    except TypeError:
        return obj


def _embed_image(path: Path) -> str:
    """Read an image file and return a base64 data URI string."""
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(data: ReportData, output_path: Path) -> Path:
    """Render a self-contained HTML report and write it to *output_path*.

    All images are embedded as base64 data URIs so the resulting file has
    no external dependencies and can be shared freely.

    Parameters
    ----------
    data:
        Populated ReportData containing pipeline outputs and metadata.
    output_path:
        Destination path for the HTML report file.

    Returns
    -------
    Path
        The output path (same as *output_path*).
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise ImportError(
            "jinja2 is required for HTML report generation. "
            "Install it with: pip install jinja2"
        ) from exc

    template_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )

    # Convert all dataclass instances to plain dicts for the template
    ctx: dict[str, Any] = {
        "project": data.project or {},
        "inspection": _to_dict(data.inspection),
        "segmentation": _to_dict(data.segmentation),
        "evaluation": _to_dict(data.evaluation),
        "optimization": _to_dict(data.optimization),
        "provenance": _to_dict(data.provenance) or {},
        "overlay_uris": [],
        "plot_uris": [],
    }

    for p in data.overlay_images:
        if p.exists():
            ctx["overlay_uris"].append({"name": p.stem, "uri": _embed_image(p)})

    for p in data.plots:
        if p.exists():
            ctx["plot_uris"].append({"name": p.stem, "uri": _embed_image(p)})

    template = env.get_template("report.html.j2")
    html = template.render(**ctx)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Loaders — reconstruct data from JSON files written by CLI commands
# ---------------------------------------------------------------------------


def load_report_data(
    *,
    inspection_json: Path | None = None,
    segmentation_json: Path | None = None,
    evaluation_json: Path | None = None,
    optimization_json: Path | None = None,
    project_yaml: Path | None = None,
    overlay_dir: Path | None = None,
    plots_dir: Path | None = None,
    command: str = "report",
) -> ReportData:
    """Build a ReportData by loading JSON result files from disk.

    Missing files are silently skipped (fields become None).

    Parameters
    ----------
    inspection_json:
        Path to inspection JSON produced by ``microagent inspect``.
    segmentation_json:
        Path to segmentation JSON produced by ``microagent segment``.
    evaluation_json:
        Path to evaluation/metrics JSON produced by ``microagent evaluate``.
    optimization_json:
        Path to optimization JSON produced by ``microagent optimize``.
    project_yaml:
        Path to project.yaml.
    overlay_dir:
        Directory containing overlay PNG files.
    plots_dir:
        Directory containing metric plot PNG files.
    command:
        Command description stored in the provenance metadata.

    Returns
    -------
    ReportData
    """
    from microagent.fair.provenance import collect_metadata

    def _load(path: Path | None) -> dict | None:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    project: dict[str, Any] = {}
    if project_yaml and project_yaml.exists():
        try:
            import yaml  # type: ignore[import-untyped]

            project = yaml.safe_load(project_yaml.read_text(encoding="utf-8")) or {}
        except ImportError:
            # Fall back to empty dict if PyYAML is unavailable
            project = {"_note": "PyYAML not installed — project.yaml not parsed"}

    overlay_images: list[Path] = []
    if overlay_dir and overlay_dir.is_dir():
        overlay_images = sorted(
            p for p in overlay_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    plot_paths: list[Path] = []
    if plots_dir and plots_dir.is_dir():
        plot_paths = sorted(
            p for p in plots_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    # Extract seed from segmentation or optimization JSON if present
    seg_dict = _load(segmentation_json)
    opt_dict = _load(optimization_json)
    seed: int | None = None
    if opt_dict and isinstance(opt_dict.get("study_path"), str):
        pass  # seed not stored directly in OptimizationResult

    return ReportData(
        project=project,
        inspection=_load(inspection_json),
        segmentation=seg_dict,
        evaluation=_load(evaluation_json),
        optimization=opt_dict,
        provenance=collect_metadata(command=command, random_seed=seed or 0),
        overlay_images=overlay_images,
        plots=plot_paths,
    )
