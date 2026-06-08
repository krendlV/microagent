"""Registry of small CC0 / permissive annotated datasets for train/optimize/demo.

Each entry knows how to download, verify, and lay out its data as::

    ~/.cache/microagent/datasets/<name>/
        images/        ← raw microscopy TIFFs
        ground_truth/  ← integer-labelled mask TIFFs (0 = background, 1/2/… = instances)

Override the cache root with the MICROAGENT_CACHE_DIR environment variable or
pass ``cache_dir`` explicitly to :func:`fetch_dataset`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

_CACHE_ENV = "MICROAGENT_CACHE_DIR"
_DEFAULT_CACHE = Path.home() / ".cache" / "microagent" / "datasets"


def cache_root(override: Path | None = None) -> Path:
    """Return the effective cache root directory."""
    if override is not None:
        return override
    env = os.environ.get(_CACHE_ENV)
    return Path(env) if env else _DEFAULT_CACHE


@dataclass
class DatasetSpec:
    """Metadata + download recipe for one registered dataset."""

    name: str
    description: str
    license: str
    citation: str
    urls: list[str]
    sha256s: list[str | None]
    # loader(archive_dir, dataset_dir) converts downloaded archives into
    # dataset_dir/images/ + dataset_dir/ground_truth/ layout.
    loader: Callable[[Path, Path], None]


_REGISTRY: dict[str, DatasetSpec] = {}


def _register(spec: DatasetSpec) -> None:
    _REGISTRY[spec.name] = spec


def list_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    return sorted(_REGISTRY.keys())


def get_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for *name*, raising ValueError if unknown."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {', '.join(list_datasets())}"
        )
    return _REGISTRY[name]


# ── Download helpers ──────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path, expected_sha256: str | None) -> None:
    """Download *url* → *dest* with a rich progress bar, then verify checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        try:
            response = urllib.request.urlopen(url, timeout=60)  # noqa: S310
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach {url!r}. Check your internet connection.\nDetail: {exc}"
            ) from exc

        total = int(response.headers.get("Content-Length", 0) or 0)
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Downloading {dest.name}", total=total or None)
            with tmp.open("wb") as fh:
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    fh.write(chunk)
                    progress.advance(task, len(chunk))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if expected_sha256 is not None:
        actual = _sha256_file(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}:\n"
                f"  expected: {expected_sha256}\n"
                f"  got:      {actual}"
            )
    else:
        console.print(
            f"[yellow]⚠[/yellow] No expected checksum for {dest.name} — skipping verification."
        )

    tmp.rename(dest)


# ── Loaders ───────────────────────────────────────────────────────────────────


def _flat_zip_loader(archive_dir: Path, dataset_dir: Path, filename: str) -> None:
    """Extract *filename* from *archive_dir*.

    The zip must contain ``images/`` and ``ground_truth/`` subdirectories
    with TIFF files ready for microagent's evaluate/train/optimize commands.
    """
    archive = archive_dir / filename
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dataset_dir)
    if not (dataset_dir / "images").exists():
        raise RuntimeError(
            f"Archive {filename!r} did not produce an images/ subdirectory. "
            "Expected zip layout: images/*.tif + ground_truth/*.tif at the top level."
        )


def _load_bbbc039(archive_dir: Path, dataset_dir: Path) -> None:
    """Convert BBBC039 (DSB2018 validation) archives to images/ + ground_truth/ TIFFs.

    BBBC039 provides one PNG per image and one or more binary-mask PNGs per nucleus.
    This loader merges per-nucleus masks into a single integer-labelled TIFF.
    """
    import imageio.v3 as iio
    import numpy as np
    import tifffile

    images_out = dataset_dir / "images"
    gt_out = dataset_dir / "ground_truth"
    images_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)

    tmp_img = archive_dir / "_tmp_img"
    tmp_gt = archive_dir / "_tmp_gt"
    tmp_img.mkdir(exist_ok=True)
    tmp_gt.mkdir(exist_ok=True)

    with zipfile.ZipFile(archive_dir / "BBBC039_v1_images.zip") as zf:
        zf.extractall(tmp_img)
    with zipfile.ZipFile(archive_dir / "BBBC039_v1_ground_truth.zip") as zf:
        zf.extractall(tmp_gt)

    try:
        for img_file in sorted(tmp_img.rglob("*.png")):
            stem = img_file.stem
            arr = np.array(iio.imread(img_file))
            if arr.ndim == 3:
                arr = arr[..., 0]
            tifffile.imwrite(str(images_out / f"{stem}.tif"), arr)

            # Ground truth: single integer-label PNG, or multiple binary-mask PNGs
            gt_matches = sorted(tmp_gt.rglob(f"{stem}*.png"))
            if not gt_matches:
                continue
            if len(gt_matches) == 1:
                gt_arr = np.array(iio.imread(gt_matches[0]))
                if gt_arr.ndim == 3:
                    gt_arr = gt_arr[..., 0]
                tifffile.imwrite(str(gt_out / f"{stem}.tif"), gt_arr.astype(np.uint16))
            else:
                # Merge separate per-nucleus binary masks into one labelled mask
                combined = np.zeros(arr.shape[-2:], dtype=np.uint16)
                for idx, mf in enumerate(gt_matches, 1):
                    m = np.array(iio.imread(mf))
                    if m.ndim == 3:
                        m = m[..., 0]
                    combined[m > 0] = idx
                tifffile.imwrite(str(gt_out / f"{stem}.tif"), combined)
    finally:
        shutil.rmtree(tmp_img, ignore_errors=True)
        shutil.rmtree(tmp_gt, ignore_errors=True)


# ── Registry ──────────────────────────────────────────────────────────────────
# sha256=None → download is not integrity-verified (fill in after first successful fetch).
# URLs point to well-known public sources; update sha256s before a production release.

_register(
    DatasetSpec(
        name="dsb2018",
        description=(
            "2018 Data Science Bowl nuclei dataset — BBBC039 validation set "
            "(200 DAPI-stained fluorescence images with integer-labelled nuclear masks, CC0)"
        ),
        license="CC0 1.0",
        citation=(
            "Caicedo J.C. et al. (2019) Nucleus segmentation across imaging experiments: "
            "the 2018 Data Science Bowl. Nat. Methods 16, 1247–1253. "
            "Source: BBBC039 via the Broad Bioimage Benchmark Collection "
            "(https://bbbc.broadinstitute.org/BBBC039)."
        ),
        urls=[
            "https://data.broadinstitute.org/bbbc/BBBC039/BBBC039_v1_images.zip",
            "https://data.broadinstitute.org/bbbc/BBBC039/BBBC039_v1_ground_truth.zip",
        ],
        sha256s=[None, None],  # TODO: fill in after first verified download
        loader=_load_bbbc039,
    )
)

_register(
    DatasetSpec(
        name="cellpose-sample",
        description=(
            "CellPose cytoplasm sample — small subset of CellPose test images "
            "with instance-labelled masks (BSD-3-Clause)"
        ),
        license="BSD-3-Clause",
        citation=(
            "Stringer C. et al. (2021) Cellpose: a generalist algorithm for cellular "
            "segmentation. Nat. Methods 18, 100–106."
        ),
        urls=[
            "https://github.com/MouseLand/cellpose/releases/download/v3.0.10/"
            "cellpose_sample_microagent.zip",
        ],
        sha256s=[None],  # TODO: fill in after zip is published and verified
        loader=lambda ad, dd: _flat_zip_loader(ad, dd, "cellpose_sample_microagent.zip"),
    )
)

_register(
    DatasetSpec(
        name="stardist-demo",
        description=(
            "StarDist demo fluorescence nuclei — small paired image/mask set "
            "from the StarDist paper tutorials (BSD-3-Clause)"
        ),
        license="BSD-3-Clause",
        citation=(
            "Schmidt U. et al. (2018) Cell Detection with Star-convex Polygons. "
            "MICCAI 2018. https://github.com/stardist/stardist"
        ),
        urls=[
            "https://github.com/stardist/stardist/releases/download/0.9.1/"
            "stardist_demo_microagent.zip",
        ],
        sha256s=[None],  # TODO: fill in after zip is published and verified
        loader=lambda ad, dd: _flat_zip_loader(ad, dd, "stardist_demo_microagent.zip"),
    )
)


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_dataset(name: str, cache_dir: Path | None = None) -> Path:
    """Download and cache *name*, returning the dataset root directory.

    The returned path is guaranteed to contain ``images/`` and ``ground_truth/``
    subdirectories populated with TIFF files compatible with microagent's
    evaluate, train, and optimize commands.

    Skips download if already cached.  Re-downloads if the cached archive
    fails checksum verification (only when ``sha256`` is registered).

    Parameters
    ----------
    name:
        Registry key — one of :func:`list_datasets`.
    cache_dir:
        Override the cache root.  Defaults to ``~/.cache/microagent/datasets/``
        or the value of the ``MICROAGENT_CACHE_DIR`` environment variable.

    Returns
    -------
    Path
        Dataset root, e.g. ``~/.cache/microagent/datasets/dsb2018/``.
    """
    spec = get_spec(name)
    root = cache_root(cache_dir)
    dataset_dir = root / name
    archive_dir = root / f"{name}_archives"

    if (dataset_dir / "images").exists():
        console.print(f"[green]✓[/green] Dataset {name!r} already cached → {dataset_dir}")
        return dataset_dir

    archive_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for url, sha256 in zip(spec.urls, spec.sha256s, strict=True):
        filename = url.rsplit("/", 1)[-1]
        dest = archive_dir / filename
        if dest.exists():
            console.print(f"[dim]Archive {filename} already present — skipping download[/dim]")
            if sha256 is not None:
                actual = _sha256_file(dest)
                if actual != sha256:
                    dest.unlink()
                    console.print(
                        f"[yellow]Checksum mismatch on cached {filename} — re-downloading[/yellow]"
                    )
                    _download_file(url, dest, sha256)
        else:
            console.print(f"[bold]Fetching[/bold] {url}")
            _download_file(url, dest, sha256)

    console.print(f"[bold green]Extracting[/bold green] {name} …")
    try:
        spec.loader(archive_dir, dataset_dir)
    except Exception as exc:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract dataset {name!r}: {exc}") from exc

    console.print(
        f"[green]✓ Dataset {name!r} ready[/green] → {dataset_dir}\n"
        f"  [dim]License: {spec.license}[/dim]\n"
        f"  [dim]Cite as: {spec.citation}[/dim]"
    )
    return dataset_dir
