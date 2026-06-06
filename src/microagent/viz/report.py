"""Self-contained HTML report generation for MicroAgent pipeline results."""

from __future__ import annotations

import base64
import io
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Long edge (px) embedded overlays/plots are downscaled to. Screen-resolution
# galleries never need full 2048 px assets, and this keeps self-contained
# reports small enough to email.
_MAX_EMBED_EDGE = 1600

# JPEG quality for re-encoded photographic images (overlays/composites).
_JPEG_QUALITY = 85

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


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


def _encode_image(path: Path, *, is_plot: bool = False) -> tuple[str, bytes]:
    """Load, downscale, and re-encode an image for compact embedding.

    Photographic images (overlays, composites) are re-encoded as JPEG to
    collapse their size dramatically; charts and plots stay PNG so text and
    thin lines remain crisp. Either way the long edge is capped at
    ``_MAX_EMBED_EDGE`` px. Falls back to the raw file bytes (with the
    suffix-derived MIME type) if Pillow is unavailable or cannot read it.

    Parameters
    ----------
    path:
        Source image on disk.
    is_plot:
        When True, keep the image lossless (PNG); otherwise emit JPEG.

    Returns
    -------
    tuple[str, bytes]
        The MIME type and the encoded image bytes.
    """
    fallback_mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    try:
        from PIL import Image
    except ImportError:
        return fallback_mime, path.read_bytes()

    try:
        with Image.open(path) as im:
            im.load()
            long_edge = max(im.size)
            if long_edge > _MAX_EMBED_EDGE:
                scale = _MAX_EMBED_EDGE / long_edge
                new_size = (
                    max(1, round(im.width * scale)),
                    max(1, round(im.height * scale)),
                )
                im = im.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            if is_plot:
                im.save(buf, format="PNG", optimize=True)
                mime = "image/png"
            else:
                if im.mode not in ("RGB", "L"):
                    im = im.convert("RGB")
                im.save(buf, format="JPEG", quality=_JPEG_QUALITY)
                mime = "image/jpeg"
            return mime, buf.getvalue()
    except (OSError, ValueError):
        # Unreadable or unsupported image — embed it verbatim.
        return fallback_mime, path.read_bytes()


def _embed_image(path: Path, *, is_plot: bool = False) -> str:
    """Return a base64 data URI for *path*, downscaled and re-encoded."""
    mime, data = _encode_image(path, is_plot=is_plot)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _copy_asset(path: Path, assets_dir: Path) -> str:
    """Copy *path* (full resolution) into *assets_dir*, return a relative URI.

    Used by the ``--no-embed`` mode so the HTML references sidecar files
    instead of inlining megabytes of base64.
    """
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / path.name
    shutil.copyfile(path, dest)
    return f"{assets_dir.name}/{dest.name}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(data: ReportData, output_path: Path, *, embed: bool = True) -> Path:
    """Render an HTML report and write it to *output_path*.

    By default (``embed=True``) every image is downscaled, re-encoded, and
    inlined as a base64 data URI, so the resulting file is self-contained and
    can be shared freely while staying small. With ``embed=False`` the
    full-resolution images are copied into an ``<output_stem>_assets/`` sidecar
    directory and referenced by relative path instead.

    Parameters
    ----------
    data:
        Populated ReportData containing pipeline outputs and metadata.
    output_path:
        Destination path for the HTML report file.
    embed:
        When True, inline images as compact data URIs; when False, write
        full-resolution sidecar assets and reference them by relative path.

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if embed:
        # ``link`` is left unset: with an inlined data URI there is nothing
        # bigger to link to, and duplicating the URI in an <a href> would
        # double the embedded payload.
        for p in data.overlay_images:
            if p.exists():
                ctx["overlay_uris"].append(
                    {"name": p.stem, "uri": _embed_image(p, is_plot=False), "link": None}
                )
        for p in data.plots:
            if p.exists():
                ctx["plot_uris"].append(
                    {"name": p.stem, "uri": _embed_image(p, is_plot=True), "link": None}
                )
    else:
        # Sidecar assets carry the full resolution, so the gallery thumbnail
        # links out to the same file it displays.
        assets_dir = output_path.parent / f"{output_path.stem}_assets"
        for p in data.overlay_images:
            if p.exists():
                rel = _copy_asset(p, assets_dir)
                ctx["overlay_uris"].append({"name": p.stem, "uri": rel, "link": rel})
        for p in data.plots:
            if p.exists():
                rel = _copy_asset(p, assets_dir)
                ctx["plot_uris"].append({"name": p.stem, "uri": rel, "link": rel})

    template = env.get_template("report.html.j2")
    html = template.render(**ctx)

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

    def _auto(path: Path | None, *candidates: str) -> Path | None:
        if path is not None:
            return path
        for candidate in candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        return None

    def _load(path: Path | None) -> dict | None:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    inspection_json = _auto(
        inspection_json,
        "inspection.json",
        "microagent_inspection/inspection.json",
    )
    segmentation_json = _auto(
        segmentation_json,
        "segmentation.json",
        "masks/segmentation.json",
        "masks/segmentation_metadata.json",
    )
    evaluation_json = _auto(evaluation_json, "metrics.json", "evaluation.json")
    optimization_json = _auto(optimization_json, "optimization.json")
    project_yaml = _auto(project_yaml, "project.yaml")

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
        # Embed one representation per image: the per-image ``*_overlay`` files.
        # Redundant side-by-side composites and the combined montage are
        # skipped to keep the gallery (and report size) lean.
        def _is_primary_overlay(p: Path) -> bool:
            stem = p.stem.lower()
            return "sidebyside" not in stem and not stem.startswith("montage")

        candidates = [
            p for p in overlay_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        filtered = [p for p in candidates if _is_primary_overlay(p)]
        # If nothing matches the naming convention, fall back to all images so
        # arbitrary overlay directories still render something.
        overlay_images = sorted(filtered or candidates)

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
