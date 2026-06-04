"""Segmentation overlay composites for microscopy images."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to H×W uint8 grayscale-as-RGB for overlay.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale, or (C, H, W) / (H, W, C) multi-channel.

    Returns
    -------
    np.ndarray
        H×W×3 uint8.
    """
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        # Detect channel-first (C, H, W) vs channel-last (H, W, C)
        if image.shape[0] <= 4 and image.shape[0] < image.shape[1]:
            # channel-first: max-project along axis 0
            gray = image.max(axis=0)
        else:
            # channel-last: max-project along axis 2
            gray = image.max(axis=2)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Normalise to uint8
    gray = gray.astype(np.float64)
    vmin, vmax = gray.min(), gray.max()
    gray = (gray - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(gray)
    gray_u8 = (gray * 255).astype(np.uint8)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


def create_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "tab20",
) -> np.ndarray:
    """Create a colored mask overlay on a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D grayscale or multi-channel).
    masks : np.ndarray
        Integer label array (0 = background).
    alpha : float
        Overlay transparency, 0–1.
    colormap : str
        Matplotlib colormap name used for label colouring.

    Returns
    -------
    np.ndarray
        H×W×3 uint8 RGB image.
    """
    from skimage.color import label2rgb

    bg = _to_rgb(image).astype(np.float64) / 255.0
    colored = label2rgb(masks, image=bg, alpha=alpha, bg_label=0, kind="overlay")
    return (np.clip(colored, 0, 1) * 255).astype(np.uint8)


def create_error_overlay(
    image: np.ndarray,
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Overlay false positives (red), false negatives (blue), true positives (green).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    pred_masks : np.ndarray
        Integer label array of predicted objects.
    gt_masks : np.ndarray
        Integer label array of ground-truth objects.
    iou_threshold : float
        IoU threshold for matching.

    Returns
    -------
    np.ndarray
        H×W×3 uint8 RGB image.
    """
    from microagent.core.evaluate import _compute_iou_matrix, _match_at_threshold

    bg = _to_rgb(image).astype(np.float64) / 255.0
    result = bg.copy()

    pred_ids = np.unique(pred_masks)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_masks)
    gt_ids = gt_ids[gt_ids > 0]

    iou_mat = _compute_iou_matrix(gt_masks, pred_masks)
    tp, fp, fn, _ = _match_at_threshold(iou_mat, iou_threshold)

    # Determine matched indices via Hungarian matching
    if len(gt_ids) > 0 and len(pred_ids) > 0:
        from scipy.optimize import linear_sum_assignment

        cost = 1.0 - iou_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_gt_idx = set()
        matched_pred_idx = set()
        for r, c in zip(row_ind, col_ind, strict=False):
            if iou_mat[r, c] >= iou_threshold:
                matched_gt_idx.add(r)
                matched_pred_idx.add(c)
    else:
        matched_gt_idx: set[int] = set()
        matched_pred_idx: set[int] = set()

    overlay_color = np.zeros((*bg.shape[:2], 3), dtype=np.float64)
    overlay_mask = np.zeros(bg.shape[:2], dtype=bool)

    # True positives — green
    for idx in matched_pred_idx:
        region = pred_masks == pred_ids[idx]
        overlay_color[region] = [0.0, 0.8, 0.0]
        overlay_mask |= region

    # False positives — red (predicted but not matched)
    for idx, pid in enumerate(pred_ids):
        if idx not in matched_pred_idx:
            region = pred_masks == pid
            overlay_color[region] = [0.9, 0.1, 0.1]
            overlay_mask |= region

    # False negatives — blue (GT but not matched)
    for idx, gid in enumerate(gt_ids):
        if idx not in matched_gt_idx:
            region = gt_masks == gid
            overlay_color[region] = [0.1, 0.1, 0.9]
            overlay_mask |= region

    alpha = 0.5
    result[overlay_mask] = (
        (1 - alpha) * bg[overlay_mask] + alpha * overlay_color[overlay_mask]
    )
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def create_comparison(
    image: np.ndarray,
    masks_a: np.ndarray,
    masks_b: np.ndarray,
    labels: tuple[str, str] = ("Run A", "Run B"),
) -> np.ndarray:
    """Side-by-side overlay of two mask sets with burned-in labels.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    masks_a : np.ndarray
        Integer label array for left panel.
    masks_b : np.ndarray
        Integer label array for right panel.
    labels : tuple[str, str]
        Text labels to burn into each panel.

    Returns
    -------
    np.ndarray
        H×(2W)×3 uint8 RGB image.
    """
    import io

    import matplotlib.pyplot as plt

    ov_a = create_overlay(image, masks_a)
    ov_b = create_overlay(image, masks_b)

    h, w = ov_a.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(w * 2 / 100, h / 100), dpi=100)

    for ax, ov, label in zip(axes, [ov_a, ov_b], labels, strict=False):
        ax.imshow(ov)
        ax.set_title(label, fontsize=10, color="white", backgroundcolor="black", pad=2)
        ax.axis("off")

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(buf).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except ImportError:
        import imageio.v3 as iio
        buf.seek(0)
        return np.array(iio.imread(buf), dtype=np.uint8)[..., :3]


def save_overlay_montage(
    images: list[np.ndarray],
    masks_list: list[np.ndarray],
    output_path: Path,
    ncols: int = 3,
    figsize_per: tuple[float, float] = (4, 4),
) -> Path:
    """Save a grid of overlays for multiple images as PNG.

    Parameters
    ----------
    images : list[np.ndarray]
        List of input images.
    masks_list : list[np.ndarray]
        List of corresponding label arrays.
    output_path : Path
        Output PNG file path.
    ncols : int
        Number of columns in the grid.
    figsize_per : tuple[float, float]
        Figure size per subplot (width, height) in inches.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
    )
    axes_flat = np.array(axes).flatten()

    for i, (img, masks) in enumerate(zip(images, masks_list, strict=False)):
        ov = create_overlay(img, masks)
        axes_flat[i].imshow(ov)
        axes_flat[i].axis("off")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(pad=0.5)
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
