"""Segmentation backends: CellPose, StarDist, and auto-selection logic."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Optional dependency flags ──────────────────────────────────────────────────

try:
    from cellpose import models as _cp_models

    _HAS_CELLPOSE = True
except ImportError:
    _HAS_CELLPOSE = False

try:
    from stardist.models import StarDist2D as _StarDist2D

    _HAS_STARDIST = True
except ImportError:
    _HAS_STARDIST = False

try:
    import tifffile  # noqa: F401

    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class PerImageStats:
    filename: str
    n_labels: int
    elapsed_seconds: float


@dataclass
class SegmentationResult:
    mask_paths: list[str]
    model_info: dict[str, Any]
    parameters: dict[str, Any]
    elapsed_seconds: float
    per_image_stats: list[PerImageStats] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# ── Abstract base ──────────────────────────────────────────────────────────────


class Segmenter(ABC):
    """Abstract base class for segmentation backends."""

    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Run inference on a single (C, H, W) or (H, W) image.

        Parameters
        ----------
        image:
            Input image array. Shape (H, W) or (C, H, W), any dtype.
        **kwargs:
            Backend-specific overrides (e.g. diameter, prob_thresh).

        Returns
        -------
        np.ndarray
            Integer label mask of shape (H, W), dtype int32. 0 = background.
        """

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return model name, version, and parameters used.

        Returns
        -------
        dict
            Keys: model_name (str), backend (str), parameters (dict).
        """

    @abstractmethod
    def get_default_params(self, project: dict[str, Any] | None) -> dict[str, Any]:
        """Derive default parameters from a project.yaml dict (or None).

        Parameters
        ----------
        project:
            Parsed project.yaml as a dict, or None.

        Returns
        -------
        dict
            Parameters that will be forwarded to predict().
        """


# ── CellPose backend ───────────────────────────────────────────────────────────


class CellPoseSegmenter(Segmenter):
    """CellPose (cpsam) segmentation backend.

    Parameters
    ----------
    diameter:
        Expected cell diameter in pixels. None → CellPose auto-detects.
    flow_threshold:
        Flow error threshold. Higher values include more cell predictions.
    cellprob_threshold:
        Cell probability threshold. Lower → more cells detected.
    channels:
        [cytoplasm, nucleus] channel indices. [0, 0] = grayscale.
    gpu:
        Use GPU if available.
    """

    def __init__(
        self,
        diameter: int | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        channels: list[int] | None = None,
        gpu: bool = True,
    ) -> None:
        if not _HAS_CELLPOSE:
            raise ImportError(
                "cellpose is not installed. Install it with: pip install cellpose"
            )
        self._diameter = diameter
        self._flow_threshold = flow_threshold
        self._cellprob_threshold = cellprob_threshold
        self._channels = channels if channels is not None else [0, 0]
        self._model = _cp_models.CellposeModel(gpu=gpu, pretrained_model="cpsam")

    def predict(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Run CellPose inference.

        Parameters
        ----------
        image:
            (H, W) or (C, H, W) array.
        **kwargs:
            Override: diameter, flow_threshold, cellprob_threshold, channels.

        Returns
        -------
        np.ndarray
            Label mask (H, W), int32.
        """
        # Flatten multi-channel to 2-D for CellPose (use first channel)
        img2d = image[0].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)

        diameter = kwargs.get("diameter", self._diameter)
        flow_threshold = kwargs.get("flow_threshold", self._flow_threshold)
        cellprob_threshold = kwargs.get("cellprob_threshold", self._cellprob_threshold)
        channels = kwargs.get("channels", self._channels)

        masks, _flows, _styles = self._model.eval(
            img2d,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=channels,
        )
        return masks.astype(np.int32)

    def get_info(self) -> dict[str, Any]:
        return {
            "model_name": "cpsam",
            "backend": "cellpose",
            "parameters": {
                "diameter": self._diameter,
                "flow_threshold": self._flow_threshold,
                "cellprob_threshold": self._cellprob_threshold,
                "channels": self._channels,
            },
        }

    def get_default_params(self, project: dict[str, Any] | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "diameter": self._diameter,
            "flow_threshold": self._flow_threshold,
            "cellprob_threshold": self._cellprob_threshold,
            "channels": self._channels,
        }
        if project is None:
            return params

        imaging = project.get("imaging", {})
        # Honour per-project diameter if present
        if "cell_diameter_pixels" in imaging:
            params["diameter"] = int(imaging["cell_diameter_pixels"])
        # Map channel config: {nucleus: 0, cytoplasm: 1} → [cyto, nucleus]
        ch_map = imaging.get("channels", {})
        nucleus_ch = ch_map.get("nucleus", None)
        cyto_ch = ch_map.get("cytoplasm", None)
        if nucleus_ch is not None and cyto_ch is not None:
            params["channels"] = [int(cyto_ch), int(nucleus_ch)]
        elif nucleus_ch is not None:
            params["channels"] = [0, int(nucleus_ch)]
        return params


# ── StarDist backend ───────────────────────────────────────────────────────────


class StarDistSegmenter(Segmenter):
    """StarDist segmentation backend (optional dependency).

    Parameters
    ----------
    model_name:
        Pre-trained model, e.g. "2D_versatile_fluo" or "2D_versatile_he".
    prob_thresh:
        Object probability threshold. Higher → stricter detection.
    nms_thresh:
        Non-maximum suppression threshold.
    scale:
        Isotropic scale factor applied before prediction.
    """

    def __init__(
        self,
        model_name: str = "2D_versatile_fluo",
        prob_thresh: float | None = None,
        nms_thresh: float | None = None,
        scale: float | None = None,
    ) -> None:
        if not _HAS_STARDIST:
            raise ImportError(
                "stardist is not installed. Install it with: pip install stardist"
            )
        self._model_name = model_name
        self._prob_thresh = prob_thresh
        self._nms_thresh = nms_thresh
        self._scale = scale
        self._model = _StarDist2D.from_pretrained(model_name)

    def predict(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Run StarDist inference.

        Parameters
        ----------
        image:
            (H, W) or (C, H, W) array.
        **kwargs:
            Override: prob_thresh, nms_thresh, scale.

        Returns
        -------
        np.ndarray
            Label mask (H, W), int32.
        """
        img2d = image[0].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)

        # Normalize to [0, 1]
        img_max = img2d.max()
        if img_max > 0:
            img2d = img2d / img_max

        prob_thresh = kwargs.get("prob_thresh", self._prob_thresh)
        nms_thresh = kwargs.get("nms_thresh", self._nms_thresh)
        scale = kwargs.get("scale", self._scale)

        predict_kwargs: dict[str, Any] = {}
        if prob_thresh is not None:
            predict_kwargs["prob_thresh"] = prob_thresh
        if nms_thresh is not None:
            predict_kwargs["nms_thresh"] = nms_thresh
        if scale is not None:
            predict_kwargs["scale"] = scale

        labels, _details = self._model.predict_instances(img2d, **predict_kwargs)
        return labels.astype(np.int32)

    def get_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "backend": "stardist",
            "parameters": {
                "prob_thresh": self._prob_thresh,
                "nms_thresh": self._nms_thresh,
                "scale": self._scale,
            },
        }

    def get_default_params(self, project: dict[str, Any] | None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "prob_thresh": self._prob_thresh,
            "nms_thresh": self._nms_thresh,
            "scale": self._scale,
        }
        return params


# ── Model selection ────────────────────────────────────────────────────────────

_STAINING_TO_STARDIST_MODEL = {
    "fluorescence": "2D_versatile_fluo",
    "he": "2D_versatile_he",
    "h&e": "2D_versatile_he",
}


def select_segmenter(project: dict[str, Any] | None = None) -> Segmenter:
    """Choose and instantiate the best segmenter given a project config.

    Selection logic (Section 3.3 of project plan):

    - nuclei + fluorescence  → CellPose cpsam (preferred) or StarDist fluo
    - nuclei + H&E           → StarDist 2D_versatile_he
    - whole_cells            → CellPose cpsam
    - organelles / EM        → CellPose cpsam (μSAM placeholder)
    - anything else          → CellPose cpsam (default)

    Parameters
    ----------
    project:
        Parsed project.yaml as a dict, or None.

    Returns
    -------
    Segmenter
        An instantiated, pre-configured segmenter.
    """
    if project is None:
        return _make_cellpose(project)

    imaging = project.get("imaging", {})
    target = imaging.get("segmentation_target", "").lower()
    staining = imaging.get("staining", "").lower()

    # nuclei + H&E → StarDist he
    if "nuclei" in target and staining in ("he", "h&e"):
        if _HAS_STARDIST:
            sd = StarDistSegmenter(model_name="2D_versatile_he")
            _apply_project_params(sd, project)
            return sd
        # Fallback to CellPose if StarDist not installed
        return _make_cellpose(project)

    # nuclei + fluorescence → CellPose (preferred) or StarDist
    if "nuclei" in target and staining in ("fluorescence", "fluo", ""):
        if _HAS_CELLPOSE:
            return _make_cellpose(project)
        if _HAS_STARDIST:
            sd = StarDistSegmenter(model_name="2D_versatile_fluo")
            _apply_project_params(sd, project)
            return sd
        raise RuntimeError(
            "Neither cellpose nor stardist is installed. "
            "Install at least one: pip install cellpose"
        )

    # whole_cells, organelles/EM, custom → CellPose cpsam
    return _make_cellpose(project)


def _make_cellpose(project: dict[str, Any] | None) -> CellPoseSegmenter:
    seg = CellPoseSegmenter()
    params = seg.get_default_params(project)
    seg._diameter = params.get("diameter")
    seg._flow_threshold = params.get("flow_threshold", 0.4)
    seg._cellprob_threshold = params.get("cellprob_threshold", 0.0)
    seg._channels = params.get("channels", [0, 0])
    return seg


def _apply_project_params(seg: Segmenter, project: dict[str, Any] | None) -> None:
    """Apply project-derived params back onto a segmenter instance (in-place)."""
    # StarDist has no mutable params to apply from project currently
    pass


# ── I/O helpers ────────────────────────────────────────────────────────────────

_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _discover_images(image_dir: Path) -> list[Path]:
    """Return sorted list of image paths in image_dir."""
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


def _load_image(path: Path) -> np.ndarray:
    """Load image from path; returns numpy array."""
    if _HAS_TIFFFILE and path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(str(path))
    try:
        import imageio

        return np.asarray(imageio.imread(str(path)))
    except ImportError:
        pass
    raise RuntimeError(f"Cannot load {path}: install tifffile or imageio")


def _save_mask(mask: np.ndarray, path: Path) -> None:
    """Save 32-bit labeled mask as TIFF."""
    if not _HAS_TIFFFILE:
        raise RuntimeError("tifffile is required to save masks: pip install tifffile")
    import tifffile

    tifffile.imwrite(str(path), mask.astype(np.int32))


# ── Public pipeline entry point ────────────────────────────────────────────────


def run_segmentation(
    image_dir: Path,
    output_dir: Path,
    model: str = "auto",
    project_path: Path | None = None,
    **kwargs: Any,
) -> SegmentationResult:
    """Segment all images in image_dir and save labeled TIFF masks to output_dir.

    Parameters
    ----------
    image_dir:
        Directory containing input images.
    output_dir:
        Directory where mask TIFFs and metadata JSON will be written.
    model:
        ``"auto"`` (select from project.yaml), ``"cellpose"``, or ``"stardist"``.
    project_path:
        Path to project.yaml. Parsed and passed to model selection / params.
    **kwargs:
        Extra parameters forwarded to ``segmenter.predict()``.

    Returns
    -------
    SegmentationResult
        Paths to saved masks, model info, parameters, and per-image statistics.

    Raises
    ------
    FileNotFoundError
        If image_dir does not exist.
    RuntimeError
        If no images are found in image_dir.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load project config ────────────────────────────────────────────────────
    project: dict[str, Any] | None = None
    if project_path is not None:
        project_path = Path(project_path)
        if project_path.exists():
            import yaml  # type: ignore[import-untyped]

            raw = yaml.safe_load(project_path.read_text())
            if isinstance(raw, dict):
                project = raw
            else:
                import logging as _log

                _log.getLogger(__name__).warning(
                    "project.yaml does not contain a dict, ignoring: %s", project_path
                )

    # ── Instantiate segmenter ──────────────────────────────────────────────────
    segmenter: Segmenter
    if model == "cellpose":
        segmenter = CellPoseSegmenter()
    elif model == "stardist":
        segmenter = StarDistSegmenter()
    else:
        segmenter = select_segmenter(project)

    # Apply project-derived defaults as base params, then overlay kwargs
    default_params = segmenter.get_default_params(project)
    predict_kwargs = {**default_params, **kwargs}

    # ── Discover images ────────────────────────────────────────────────────────
    image_paths = _discover_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    # ── Run inference ──────────────────────────────────────────────────────────
    wall_start = time.perf_counter()
    mask_paths: list[str] = []
    per_image_stats: list[PerImageStats] = []

    for img_path in image_paths:
        t0 = time.perf_counter()
        image = _load_image(img_path)
        mask = segmenter.predict(image, **predict_kwargs)
        elapsed = time.perf_counter() - t0

        mask_name = img_path.stem + "_mask.tif"
        mask_path = output_dir / mask_name
        _save_mask(mask, mask_path)
        mask_paths.append(str(mask_path))

        n_labels = int(mask.max())
        per_image_stats.append(
            PerImageStats(filename=img_path.name, n_labels=n_labels, elapsed_seconds=elapsed)
        )

    total_elapsed = time.perf_counter() - wall_start

    result = SegmentationResult(
        mask_paths=mask_paths,
        model_info=segmenter.get_info(),
        parameters=predict_kwargs,
        elapsed_seconds=total_elapsed,
        per_image_stats=per_image_stats,
    )

    # Save metadata JSON alongside masks
    meta_path = output_dir / "segmentation_metadata.json"
    result.save_json(meta_path)

    return result
