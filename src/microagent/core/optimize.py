"""Hyperparameter optimisation via Optuna for microscopy segmentation."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class OptimizeConfig:
    """Configuration for hyperparameter optimisation.

    Parameters
    ----------
    image_dir:
        Directory containing raw images.
    gt_dir:
        Directory containing ground-truth label masks.
    model:
        Backend to optimise: ``"auto"``, ``"cellpose"``, or ``"stardist"``.
    n_trials:
        Number of Optuna trials to run.
    metric:
        Optimisation target: ``"f1"``, ``"precision"``, ``"recall"``, ``"map"``, ``"pq"``.
    iou_threshold:
        IoU threshold used when computing F1 / precision / recall.
    seed:
        Random seed for TPESampler reproducibility.
    project_path:
        Optional path to project.yaml for search-space narrowing.
    """

    image_dir: Path
    gt_dir: Path
    model: str = "auto"
    n_trials: int = 20
    metric: str = "f1"
    iou_threshold: float = 0.5
    seed: int = 42
    project_path: Path | None = None


@dataclass
class TrialRecord:
    """Result of a single completed Optuna trial."""

    number: int
    params: dict[str, Any]
    value: float


@dataclass
class OptimizationResult:
    """Output of :func:`run_optimization`.

    Parameters
    ----------
    best_params:
        Parameter dict from the best Optuna trial.
    best_value:
        Metric value achieved by best_params.
    baseline_value:
        Metric value with default (un-tuned) parameters.
    improvement:
        ``best_value - baseline_value``.
    trials:
        All completed trial records.
    study_path:
        Path where the Optuna study pickle was saved (None on failure).
    """

    best_params: dict[str, Any]
    best_value: float
    baseline_value: float
    improvement: float
    trials: list[TrialRecord] = field(default_factory=list)
    study_path: Path | None = None


# ── Internal helpers ───────────────────────────────────────────────────────────


def _load_project(project_path: Path | None) -> dict[str, Any] | None:
    """Load and validate a project YAML file.

    Parameters
    ----------
    project_path : Path | None
        Path to a project.yaml file, or None.

    Returns
    -------
    dict[str, Any] | None
        Parsed YAML contents as a dict, or ``None`` if *project_path* is
        ``None``, the file does not exist, the YAML content is not a dict,
        or parsing fails due to ``yaml.YAMLError`` or ``OSError``.
    """
    if project_path is None:
        return None
    project_path = Path(project_path)
    if not project_path.exists():
        return None
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(project_path.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except (yaml.YAMLError, OSError):
        return None


def _resolve_backend(model: str) -> str:
    """Return the concrete backend name: 'cellpose' or 'stardist'."""
    from microagent.core.segment import _HAS_CELLPOSE, _HAS_STARDIST

    if model == "cellpose":
        if not _HAS_CELLPOSE:
            raise ImportError("cellpose is not installed. pip install cellpose")
        return "cellpose"
    if model == "stardist":
        if not _HAS_STARDIST:
            raise ImportError("stardist is not installed. pip install stardist")
        return "stardist"
    # auto: prefer cellpose
    if _HAS_CELLPOSE:
        return "cellpose"
    if _HAS_STARDIST:
        return "stardist"
    raise ImportError("Neither cellpose nor stardist is installed.")


_GT_SUFFIXES = ("_masks", "_mask", "_seg", "_labels", "_label")
_GT_EXTS = {".tif", ".tiff", ".npy"}


def _gt_stem(path: Path) -> str:
    """Strip known mask suffixes and lowercase the stem."""
    s = path.stem
    for sfx in _GT_SUFFIXES:
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break
    return s.lower()


def _match_gt_paths(image_paths: list[Path], gt_dir: Path) -> list[Path | None]:
    """Return GT mask paths aligned to *image_paths* (None if no match)."""
    gt_files = sorted(f for f in Path(gt_dir).iterdir() if f.suffix.lower() in _GT_EXTS)
    gt_map: dict[str, Path] = {_gt_stem(f): f for f in gt_files}
    return [gt_map.get(p.stem.lower()) for p in image_paths]


def _build_segmenter(backend: str, params: dict[str, Any]) -> Any:
    """Instantiate a segmenter with the given parameter dict.

    Parameters
    ----------
    backend : str
        Either ``'cellpose'`` or ``'stardist'``.
    params : dict[str, Any]
        Parameter dictionary with backend-specific keys.

    Returns
    -------
    Segmenter
        Instantiated segmenter subclass.
    """
    from microagent.core.segment import CellPoseSegmenter, StarDistSegmenter

    if backend == "cellpose":
        return CellPoseSegmenter(
            diameter=params.get("diameter"),
            flow_threshold=params.get("flow_threshold", 0.4),
            cellprob_threshold=params.get("cellprob_threshold", 0.0),
        )
    return StarDistSegmenter(
        prob_thresh=params.get("prob_thresh"),
        nms_thresh=params.get("nms_thresh"),
    )


def _suggest_params(trial: Any, backend: str, search_space: dict[str, Any]) -> dict[str, Any]:
    """Ask Optuna to suggest parameters for one trial.

    Parameters
    ----------
    trial : optuna.Trial
        Current Optuna trial object.
    backend : str
        Either ``'cellpose'`` or ``'stardist'``.
    search_space : dict[str, Any]
        Bounds for the parameter search space.

    Returns
    -------
    dict[str, Any]
        Suggested parameter dict for this trial.
    """
    if backend == "cellpose":
        diam_low = float(search_space.get("diameter_low", 10.0))
        diam_high = float(search_space.get("diameter_high", 200.0))
        return {
            "diameter": trial.suggest_float("diameter", diam_low, diam_high),
            "flow_threshold": trial.suggest_float("flow_threshold", 0.1, 1.0),
            "cellprob_threshold": trial.suggest_float("cellprob_threshold", -3.0, 3.0),
        }
    # stardist
    return {
        "prob_thresh": trial.suggest_float("prob_thresh", 0.3, 0.9),
        "nms_thresh": trial.suggest_float("nms_thresh", 0.1, 0.7),
    }


def _get_metric_value(im_metrics: Any, metric: str, iou_threshold: float) -> float:
    """Extract a scalar metric from an ImageMetrics instance.

    Parameters
    ----------
    im_metrics : ImageMetrics
        Per-image evaluation metrics.
    metric : str
        Metric name: ``'f1'``, ``'precision'``, ``'recall'``, ``'map'``, or ``'pq'``.
    iou_threshold : float
        IoU threshold to look up for threshold-dependent metrics.

    Returns
    -------
    float
        Scalar metric value.
    """
    if metric == "map":
        return float(im_metrics.map)
    if metric in ("pq", "panoptic_quality"):
        return float(im_metrics.panoptic_quality)
    # f1 / precision / recall — look up the right IoU threshold bucket
    for tm in im_metrics.per_threshold:
        if abs(tm.threshold - iou_threshold) < 1e-9:
            return float(getattr(tm, metric, tm.f1))
    # fallback to first bucket
    return float(getattr(im_metrics.per_threshold[0], metric, im_metrics.per_threshold[0].f1))


def _load_paired_data(
    config: OptimizeConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Load matched (image, gt) array pairs from config directories."""
    from microagent.core.evaluate import _load_mask
    from microagent.core.segment import _discover_images, _load_image

    image_paths = _discover_images(Path(config.image_dir))
    if not image_paths:
        raise RuntimeError(f"No images found in {config.image_dir}")

    gt_paths = _match_gt_paths(image_paths, Path(config.gt_dir))

    images: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    names: list[str] = []
    for img_path, gt_path in zip(image_paths, gt_paths, strict=False):
        if gt_path is None:
            continue
        images.append(_load_image(img_path))
        gts.append(_load_mask(gt_path))
        names.append(img_path.name)

    if not images:
        raise RuntimeError(f"No matched image/GT pairs found in {config.gt_dir}")
    return images, gts, names


def _build_objective_fn(
    backend: str,
    search_space: dict[str, Any],
    images: list[np.ndarray],
    gts: list[np.ndarray],
    names: list[str],
    config: OptimizeConfig,
) -> Callable:
    """Return a closure that Optuna can call as its objective function."""
    from microagent.core.evaluate import _evaluate_pair

    def objective(trial: Any) -> float:
        params = _suggest_params(trial, backend, search_space)
        segmenter = _build_segmenter(backend, params)

        values: list[float] = []
        for step, (image, gt, name) in enumerate(zip(images, gts, names, strict=True)):
            pred = segmenter.predict(image, **params)
            im_metrics = _evaluate_pair(gt, pred, name, [config.iou_threshold])
            val = _get_metric_value(im_metrics, config.metric, config.iou_threshold)
            values.append(val)

            trial.report(float(np.mean(values)), step)
            if trial.should_prune():
                import optuna as _optuna

                raise _optuna.TrialPruned()

        return float(np.mean(values))

    return objective


def _compute_baseline(
    config: OptimizeConfig,
    images: list[np.ndarray],
    gts: list[np.ndarray],
    names: list[str],
    backend: str,
) -> float:
    """Return the target metric with default (un-tuned) parameters.

    Parameters
    ----------
    config : OptimizeConfig
        Optimisation configuration.
    images : list[np.ndarray]
        Input image arrays.
    gts : list[np.ndarray]
        Ground-truth label arrays aligned with *images*.
    names : list[str]
        Filenames aligned with *images*.
    backend : str
        Segmentation backend name.

    Returns
    -------
    float
        Mean metric value across all image/GT pairs.
    """
    from microagent.core.evaluate import _evaluate_pair

    segmenter = _build_segmenter(backend, {})
    values: list[float] = []
    for image, gt, name in zip(images, gts, names, strict=True):
        pred = segmenter.predict(image)
        im_metrics = _evaluate_pair(gt, pred, name, [config.iou_threshold])
        values.append(_get_metric_value(im_metrics, config.metric, config.iou_threshold))
    return float(np.mean(values)) if values else 0.0


# ── Public API ─────────────────────────────────────────────────────────────────


def select_search_space(project: dict[str, Any] | None) -> dict[str, Any]:
    """Derive parameter search-space bounds from a project.yaml dict.

    Implements the AIxCell principle of domain-aware search-space restriction:
    if the project specifies a known cell diameter, the ``diameter`` search
    range is narrowed to ±50 % of that value.

    Parameters
    ----------
    project:
        Parsed project.yaml as a dict, or None.

    Returns
    -------
    dict
        Keys: ``diameter_low``, ``diameter_high`` (and future per-param keys).
    """
    space: dict[str, Any] = {}
    if project is None:
        return space
    imaging = project.get("imaging", {})
    diameter = imaging.get("cell_diameter_pixels")
    if diameter is not None:
        d = float(diameter)
        space["diameter_low"] = max(10.0, d * 0.5)
        space["diameter_high"] = d * 1.5
    return space


def create_objective(config: OptimizeConfig) -> Callable:
    """Build an Optuna objective function for the given optimisation config.

    Images and ground-truth masks are loaded once when this function is called
    and shared across all trials through a closure.

    Parameters
    ----------
    config:
        Optimisation configuration.

    Returns
    -------
    Callable
        An Optuna-compatible objective: ``(trial) -> float``.

    Raises
    ------
    ImportError
        If optuna is not installed or the requested backend is unavailable.
    RuntimeError
        If no matched image/GT pairs are found.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required: pip install optuna")

    project = _load_project(config.project_path)
    search_space = select_search_space(project)
    backend = _resolve_backend(config.model)
    images, gts, names = _load_paired_data(config)
    return _build_objective_fn(backend, search_space, images, gts, names, config)


def run_optimization(
    config: OptimizeConfig,
    on_trial_complete: Callable[[TrialRecord], None] | None = None,
) -> OptimizationResult:
    """Run Optuna hyperparameter optimisation and return the best parameters.

    Parameters
    ----------
    config:
        Optimisation settings.
    on_trial_complete:
        Optional callback invoked after each successful trial with a
        :class:`TrialRecord`. Use this to update a progress display.

    Returns
    -------
    OptimizationResult

    Raises
    ------
    ImportError
        If optuna is not installed.
    RuntimeError
        If no matched image/GT pairs are found.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required: pip install optuna")

    import optuna

    project = _load_project(config.project_path)
    search_space = select_search_space(project)
    backend = _resolve_backend(config.model)
    images, gts, names = _load_paired_data(config)

    baseline_value = _compute_baseline(config, images, gts, names, backend)

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    objective = _build_objective_fn(backend, search_space, images, gts, names, config)

    trials_log: list[TrialRecord] = []

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        record = TrialRecord(number=trial.number, params=dict(trial.params), value=trial.value)
        trials_log.append(record)
        if on_trial_complete is not None:
            on_trial_complete(record)

    study.optimize(objective, n_trials=config.n_trials, callbacks=[_callback])

    # Persist study for later analysis
    study_path: Path | None = None
    try:
        study_path = Path(config.image_dir).parent / "optuna_study.pkl"
        study_path.write_bytes(pickle.dumps(study))
    except Exception:
        logger.warning("Failed to persist Optuna study to disk", exc_info=True)
        study_path = None

    best_params: dict[str, Any] = dict(study.best_params) if study.best_trials else {}
    best_value: float = float(study.best_value) if study.best_trials else 0.0

    return OptimizationResult(
        best_params=best_params,
        best_value=best_value,
        baseline_value=baseline_value,
        improvement=best_value - baseline_value,
        trials=trials_log,
        study_path=study_path,
    )
