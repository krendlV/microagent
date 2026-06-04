"""Instance segmentation evaluation metrics for microscopy masks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from stardist.matching import matching, matching_dataset

    _HAS_STARDIST = True
except ImportError:
    _HAS_STARDIST = False


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class ThresholdMetrics:
    """Metrics at a single IoU threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    mean_true_score: float  # mean IoU of matched pairs


@dataclass
class ImageMetrics:
    """All metrics for a single image pair."""

    filename: str
    gt_count: int
    pred_count: int
    per_threshold: list[ThresholdMetrics]
    mean_f1: float  # mean F1 across IoU thresholds
    panoptic_quality: float  # PQ = SQ * RQ at IoU 0.5
    iou_distribution: list[float]  # per-object IoU of matched pairs at IoU 0.5


@dataclass
class DatasetMetrics:
    """Aggregated metrics across all images."""

    n_images: int
    per_threshold: list[ThresholdMetrics]
    mean_f1: float
    panoptic_quality: float
    mean_gt_count: float
    mean_pred_count: float


@dataclass
class ComparisonResult:
    """Delta between two EvaluationResults."""

    metric_deltas: dict[str, float]  # metric_name -> delta (b - a)
    improved_images: list[str]  # filenames where F1@0.5 improved
    regressed_images: list[str]  # filenames where F1@0.5 worsened


@dataclass
class EvaluationResult:
    """Full evaluation output."""

    per_image: list[ImageMetrics]
    summary: DatasetMetrics
    best_images: list[str]  # top 3 by F1@0.5
    worst_images: list[str]  # bottom 3 by F1@0.5
    unmatched_preds: list[str]
    unmatched_gts: list[str]
    comparison: Optional[ComparisonResult] = None

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load_json(path: Path) -> "EvaluationResult":
        data = json.loads(Path(path).read_text())
        per_image = [
            ImageMetrics(
                filename=im["filename"],
                gt_count=im["gt_count"],
                pred_count=im["pred_count"],
                per_threshold=[ThresholdMetrics(**tm) for tm in im["per_threshold"]],
                mean_f1=im["mean_f1"],
                panoptic_quality=im["panoptic_quality"],
                iou_distribution=im["iou_distribution"],
            )
            for im in data["per_image"]
        ]
        summary = DatasetMetrics(
            n_images=data["summary"]["n_images"],
            per_threshold=[ThresholdMetrics(**tm) for tm in data["summary"]["per_threshold"]],
            mean_f1=data["summary"]["mean_f1"],
            panoptic_quality=data["summary"]["panoptic_quality"],
            mean_gt_count=data["summary"]["mean_gt_count"],
            mean_pred_count=data["summary"]["mean_pred_count"],
        )
        comp = None
        if data.get("comparison"):
            comp = ComparisonResult(**data["comparison"])
        return EvaluationResult(
            per_image=per_image,
            summary=summary,
            best_images=data["best_images"],
            worst_images=data["worst_images"],
            unmatched_preds=data["unmatched_preds"],
            unmatched_gts=data["unmatched_gts"],
            comparison=comp,
        )


# ── Fallback IoU matching ──────────────────────────────────────────────────────


def _compute_iou_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between all GT and pred label pairs.

    Parameters
    ----------
    y_true : np.ndarray
        Integer label array (0 = background).
    y_pred : np.ndarray
        Integer label array (0 = background).

    Returns
    -------
    np.ndarray
        Shape (n_gt, n_pred) IoU matrix.
    """
    gt_ids = np.unique(y_true)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(y_pred)
    pred_ids = pred_ids[pred_ids > 0]

    n_gt = len(gt_ids)
    n_pred = len(pred_ids)
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float64)

    for i, g in enumerate(gt_ids):
        gt_mask = y_true == g
        for j, p in enumerate(pred_ids):
            pred_mask = y_pred == p
            inter = np.logical_and(gt_mask, pred_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou_mat[i, j] = inter / union

    return iou_mat


def _match_at_threshold(
    iou_mat: np.ndarray, thresh: float
) -> tuple[int, int, int, list[float]]:
    """Hungarian matching at given IoU threshold.

    Returns
    -------
    tp, fp, fn, matched_ious
    """
    from scipy.optimize import linear_sum_assignment

    n_gt, n_pred = iou_mat.shape
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0, []
    if n_gt == 0:
        return 0, n_pred, 0, []
    if n_pred == 0:
        return 0, 0, n_gt, []

    # Maximise IoU via Hungarian algorithm
    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)

    tp = 0
    matched_ious: list[float] = []
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] >= thresh:
            tp += 1
            matched_ious.append(float(iou_mat[r, c]))

    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn, matched_ious


def _metrics_from_fallback(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: list[float]
) -> tuple[list[ThresholdMetrics], list[float]]:
    """Compute metrics using scipy fallback (no stardist)."""
    iou_mat = _compute_iou_matrix(y_true, y_pred)
    results: list[ThresholdMetrics] = []
    iou_dist_05: list[float] = []

    for thresh in thresholds:
        tp, fp, fn, matched_ious = _match_at_threshold(iou_mat, thresh)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_ts = float(np.mean(matched_ious)) if matched_ious else 0.0
        results.append(
            ThresholdMetrics(
                threshold=thresh,
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                mean_true_score=mean_ts,
            )
        )
        if abs(thresh - 0.5) < 1e-9:
            iou_dist_05 = matched_ious

    return results, iou_dist_05


def _metrics_from_stardist(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: list[float]
) -> tuple[list[ThresholdMetrics], list[float]]:
    """Compute metrics using stardist.matching."""
    results: list[ThresholdMetrics] = []
    iou_dist_05: list[float] = []

    for thresh in thresholds:
        m = matching(y_true, y_pred, thresh=thresh)
        results.append(
            ThresholdMetrics(
                threshold=thresh,
                precision=float(m.precision),
                recall=float(m.recall),
                f1=float(m.f1),
                tp=int(m.tp),
                fp=int(m.fp),
                fn=int(m.fn),
                mean_true_score=float(m.mean_true_score),
            )
        )
        if abs(thresh - 0.5) < 1e-9:
            iou_dist_05 = [float(v) for v in m.iou]

    return results, iou_dist_05


# ── Per-image evaluation ───────────────────────────────────────────────────────


def _load_mask(path: Path) -> np.ndarray:
    """Load a label mask from TIFF or NPY."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile

            return tifffile.imread(str(path)).astype(np.int32)
        except ImportError:
            pass
        try:
            import imageio.v3 as iio

            return np.array(iio.imread(str(path))).astype(np.int32)
        except ImportError:
            pass
        raise RuntimeError(f"Cannot load TIFF: neither tifffile nor imageio available ({path})")
    if suffix == ".npy":
        return np.load(path).astype(np.int32)
    raise ValueError(f"Unsupported mask format: {suffix}")


def _evaluate_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str,
    thresholds: list[float],
    force_fallback: bool = False,
) -> ImageMetrics:
    """Compute all metrics for one GT/pred pair."""
    gt_count = int(np.max(y_true))
    pred_count = int(np.max(y_pred))

    if _HAS_STARDIST and not force_fallback:
        per_thresh, iou_dist = _metrics_from_stardist(y_true, y_pred, thresholds)
    else:
        per_thresh, iou_dist = _metrics_from_fallback(y_true, y_pred, thresholds)

    mean_f1 = float(np.mean([m.f1 for m in per_thresh]))

    # PQ at IoU 0.5: SQ * RQ where SQ = mean_true_score, RQ = F1
    m05 = next((m for m in per_thresh if abs(m.threshold - 0.5) < 1e-9), per_thresh[0])
    pq = m05.mean_true_score * m05.f1

    return ImageMetrics(
        filename=filename,
        gt_count=gt_count,
        pred_count=pred_count,
        per_threshold=per_thresh,
        mean_f1=mean_f1,
        panoptic_quality=pq,
        iou_distribution=iou_dist,
    )


# ── File matching ──────────────────────────────────────────────────────────────

_MASK_SUFFIXES = ("_masks", "_mask", "_seg", "_labels", "_label")
_MASK_EXTS = {".tif", ".tiff", ".npy"}


def _stem(path: Path) -> str:
    """Return the base stem stripping known mask suffixes."""
    s = path.stem
    for sfx in _MASK_SUFFIXES:
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break
    return s.lower()


def _match_files(
    pred_dir: Path, gt_dir: Path
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    """Match prediction files to ground-truth files by normalised stem."""
    pred_files = sorted(f for f in pred_dir.iterdir() if f.suffix.lower() in _MASK_EXTS)
    gt_files = sorted(f for f in gt_dir.iterdir() if f.suffix.lower() in _MASK_EXTS)

    gt_map: dict[str, Path] = {_stem(f): f for f in gt_files}
    pred_map: dict[str, Path] = {_stem(f): f for f in pred_files}

    matched: list[tuple[Path, Path]] = []
    unmatched_preds: list[str] = []
    unmatched_gts: list[str] = []

    for key, pf in pred_map.items():
        if key in gt_map:
            matched.append((pf, gt_map[key]))
        else:
            unmatched_preds.append(pf.name)

    matched_gt_keys = {_stem(gt) for _, gt in matched}
    for key, gf in gt_map.items():
        if key not in matched_gt_keys:
            unmatched_gts.append(gf.name)

    return matched, unmatched_preds, unmatched_gts


# ── Dataset-level aggregation ──────────────────────────────────────────────────


def _aggregate(image_metrics: list[ImageMetrics], thresholds: list[float]) -> DatasetMetrics:
    """Aggregate per-image metrics into dataset summary."""
    n = len(image_metrics)
    if n == 0:
        empty = [
            ThresholdMetrics(t, 0.0, 0.0, 0.0, 0, 0, 0, 0.0) for t in thresholds
        ]
        return DatasetMetrics(0, empty, 0.0, 0.0, 0.0, 0.0)

    agg_thresh: list[ThresholdMetrics] = []
    for idx, thresh in enumerate(thresholds):
        tp = sum(im.per_threshold[idx].tp for im in image_metrics)
        fp = sum(im.per_threshold[idx].fp for im in image_metrics)
        fn = sum(im.per_threshold[idx].fn for im in image_metrics)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mean_ts = float(
            np.mean([im.per_threshold[idx].mean_true_score for im in image_metrics])
        )
        agg_thresh.append(
            ThresholdMetrics(
                threshold=thresh,
                precision=precision,
                recall=recall,
                f1=f1,
                tp=tp,
                fp=fp,
                fn=fn,
                mean_true_score=mean_ts,
            )
        )

    mean_f1 = float(np.mean([im.mean_f1 for im in image_metrics]))
    pq = float(np.mean([im.panoptic_quality for im in image_metrics]))
    mean_gt = float(np.mean([im.gt_count for im in image_metrics]))
    mean_pred = float(np.mean([im.pred_count for im in image_metrics]))

    return DatasetMetrics(
        n_images=n,
        per_threshold=agg_thresh,
        mean_f1=mean_f1,
        panoptic_quality=pq,
        mean_gt_count=mean_gt,
        mean_pred_count=mean_pred,
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def evaluate_masks(
    pred_dir: Path,
    gt_dir: Path,
    thresholds: list[float] | None = None,
    force_fallback: bool = False,
) -> EvaluationResult:
    """Evaluate predicted masks against ground-truth masks.

    Parameters
    ----------
    pred_dir : Path
        Directory containing predicted label masks.
    gt_dir : Path
        Directory containing ground-truth label masks.
    thresholds : list[float], optional
        IoU thresholds. Defaults to [0.5, 0.75, 0.9].
    force_fallback : bool
        Force scipy fallback even when stardist is available (for testing).

    Returns
    -------
    EvaluationResult
    """
    if thresholds is None:
        thresholds = [0.5, 0.75, 0.9]

    matched, unmatched_preds, unmatched_gts = _match_files(Path(pred_dir), Path(gt_dir))

    per_image: list[ImageMetrics] = []
    for pred_path, gt_path in matched:
        y_pred = _load_mask(pred_path)
        y_true = _load_mask(gt_path)
        im_metrics = _evaluate_pair(
            y_true, y_pred, pred_path.name, thresholds, force_fallback=force_fallback
        )
        per_image.append(im_metrics)

    summary = _aggregate(per_image, thresholds)

    def _f1_at_05(im: ImageMetrics) -> float:
        m = next((m for m in im.per_threshold if abs(m.threshold - 0.5) < 1e-9), im.per_threshold[0])
        return m.f1

    sorted_by_f1 = sorted(per_image, key=_f1_at_05)
    worst = [im.filename for im in sorted_by_f1[:3]]
    best = [im.filename for im in sorted_by_f1[-3:][::-1]]

    return EvaluationResult(
        per_image=per_image,
        summary=summary,
        best_images=best,
        worst_images=worst,
        unmatched_preds=unmatched_preds,
        unmatched_gts=unmatched_gts,
    )


def compare_runs(
    result_a: EvaluationResult, result_b: EvaluationResult
) -> ComparisonResult:
    """Compare two EvaluationResults and compute deltas.

    Parameters
    ----------
    result_a : EvaluationResult
        Baseline result.
    result_b : EvaluationResult
        New result to compare against baseline.

    Returns
    -------
    ComparisonResult
    """
    def _summary_dict(r: EvaluationResult) -> dict[str, float]:
        d: dict[str, float] = {"mean_f1": r.summary.mean_f1, "pq": r.summary.panoptic_quality}
        for tm in r.summary.per_threshold:
            t = f"{tm.threshold:.2f}"
            d[f"f1@{t}"] = tm.f1
            d[f"precision@{t}"] = tm.precision
            d[f"recall@{t}"] = tm.recall
        return d

    sa = _summary_dict(result_a)
    sb = _summary_dict(result_b)
    deltas = {k: sb.get(k, 0.0) - sa.get(k, 0.0) for k in sa}

    a_map = {im.filename: im for im in result_a.per_image}
    b_map = {im.filename: im for im in result_b.per_image}

    def _f1_05(im: ImageMetrics) -> float:
        m = next((m for m in im.per_threshold if abs(m.threshold - 0.5) < 1e-9), im.per_threshold[0])
        return m.f1

    improved: list[str] = []
    regressed: list[str] = []
    for fname in set(a_map) & set(b_map):
        da = _f1_05(a_map[fname])
        db = _f1_05(b_map[fname])
        if db > da + 1e-9:
            improved.append(fname)
        elif da > db + 1e-9:
            regressed.append(fname)

    return ComparisonResult(
        metric_deltas=deltas,
        improved_images=improved,
        regressed_images=regressed,
    )
