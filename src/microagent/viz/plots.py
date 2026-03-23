"""Metric charts and training curves for microscopy analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np


def plot_metrics_summary(eval_result, output_path: Path) -> Path:
    """Bar chart of F1/Precision/Recall at each IoU threshold.

    Parameters
    ----------
    eval_result : EvaluationResult
        Result from evaluate_masks().
    output_path : Path
        Output PNG file path.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    import matplotlib.pyplot as plt

    summary = eval_result.summary
    thresholds = [f"{tm.threshold:.2f}" for tm in summary.per_threshold]
    f1s = [tm.f1 for tm in summary.per_threshold]
    precisions = [tm.precision for tm in summary.per_threshold]
    recalls = [tm.recall for tm in summary.per_threshold]

    x = np.arange(len(thresholds))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(thresholds) * 1.5), 4))
    ax.bar(x - width, f1s, width, label="F1", color="#4C72B0")
    ax.bar(x, precisions, width, label="Precision", color="#DD8452")
    ax.bar(x + width, recalls, width, label="Recall", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels([f"IoU {t}" for t in thresholds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Metrics by IoU Threshold")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_object_size_distribution(masks: np.ndarray, output_path: Path) -> Path:
    """Histogram of object areas from regionprops.

    Parameters
    ----------
    masks : np.ndarray
        Integer label array.
    output_path : Path
        Output PNG file path.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    import matplotlib.pyplot as plt
    from skimage.measure import regionprops

    props = regionprops(masks.astype(np.int32))
    areas = [p.area for p in props]

    fig, ax = plt.subplots(figsize=(6, 4))
    if areas:
        ax.hist(areas, bins=min(30, max(5, len(areas) // 2)), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Object area (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Object Size Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metric_per_image(eval_result, metric: str = "f1", output_path: Path = None) -> Path:
    """Bar chart per image, sorted by metric value, worst highlighted.

    Parameters
    ----------
    eval_result : EvaluationResult
        Result from evaluate_masks().
    metric : str
        Which metric to plot: "f1", "precision", or "recall".
    output_path : Path
        Output PNG file path.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    import matplotlib.pyplot as plt

    per_image = eval_result.per_image

    def _get_metric(im, metric_name: str) -> float:
        # Use IoU=0.5 threshold
        m = next(
            (m for m in im.per_threshold if abs(m.threshold - 0.5) < 1e-9),
            im.per_threshold[0],
        )
        return getattr(m, metric_name)

    data = sorted(per_image, key=lambda im: _get_metric(im, metric))
    names = [im.filename for im in data]
    values = [_get_metric(im, metric) for im in data]

    colors = ["#D9534F" if v == min(values) else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 4))
    x = np.arange(len(names))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} per Image (IoU=0.5)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
) -> Path:
    """Loss curves for training monitoring.

    Parameters
    ----------
    train_losses : list[float]
        Training loss per epoch.
    val_losses : list[float]
        Validation loss per epoch.
    output_path : Path
        Output PNG file path.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="Train", color="#4C72B0", linewidth=1.5)
    if val_losses:
        val_epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, label="Validation", color="#DD8452", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
