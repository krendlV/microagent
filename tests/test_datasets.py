"""Tests for the dataset registry and downloader.

Network is fully mocked — no real HTTP requests are made in CI.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tifffile

from microagent.data.datasets import (
    DatasetSpec,
    _download_file,
    _flat_zip_loader,
    _sha256_file,
    cache_root,
    fetch_dataset,
    get_spec,
    list_datasets,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_flat_zip(tmp_path: Path, n_images: int = 3) -> tuple[bytes, str]:
    """Return (zip_bytes, sha256) for a minimal flat zip with images/ + ground_truth/."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(n_images):
            img = np.zeros((64, 64), dtype=np.uint16)
            img[10 + i * 5 : 20 + i * 5, 10:20] = i + 1
            gt = (img > 0).astype(np.uint16)

            img_buf = io.BytesIO()
            tifffile.imwrite(img_buf, img)
            zf.writestr(f"images/img_{i:03d}.tif", img_buf.getvalue())

            gt_buf = io.BytesIO()
            tifffile.imwrite(gt_buf, gt)
            zf.writestr(f"ground_truth/img_{i:03d}.tif", gt_buf.getvalue())

    raw = buf.getvalue()
    sha256 = hashlib.sha256(raw).hexdigest()
    return raw, sha256


def _fake_urlopen(content: bytes):
    """Return a mock http.client.HTTPResponse-like object yielding *content*."""
    resp = MagicMock()
    resp.headers = {"Content-Length": str(len(content))}
    remaining = bytearray(content)

    def read(n: int) -> bytes:
        chunk = bytes(remaining[:n])
        del remaining[:n]
        return chunk

    resp.read.side_effect = read
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ── Registry tests ────────────────────────────────────────────────────────────


def test_list_datasets_returns_known_names():
    names = list_datasets()
    assert "dsb2018" in names
    assert "cellpose-sample" in names
    assert "stardist-demo" in names
    assert names == sorted(names)


def test_get_spec_known():
    spec = get_spec("dsb2018")
    assert isinstance(spec, DatasetSpec)
    assert spec.name == "dsb2018"
    assert spec.license
    assert spec.citation
    assert len(spec.urls) >= 1
    assert len(spec.sha256s) == len(spec.urls)


def test_get_spec_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_spec("nonexistent-dataset-xyz")


def test_spec_has_loader():
    for name in list_datasets():
        spec = get_spec(name)
        assert callable(spec.loader), f"{name!r} spec.loader must be callable"


# ── Checksum helper tests ─────────────────────────────────────────────────────


def test_sha256_file(tmp_path):
    data = b"hello microagent"
    f = tmp_path / "test.bin"
    f.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert _sha256_file(f) == expected


# ── Download helper tests ─────────────────────────────────────────────────────


def test_download_file_verifies_checksum(tmp_path):
    content = b"fake archive content"
    good_sha = hashlib.sha256(content).hexdigest()

    with patch("urllib.request.urlopen", return_value=_fake_urlopen(content)):
        dest = tmp_path / "archive.zip"
        _download_file("http://example.com/archive.zip", dest, good_sha)
    assert dest.exists()
    assert dest.read_bytes() == content


def test_download_file_rejects_bad_checksum(tmp_path):
    content = b"fake archive content"
    bad_sha = "a" * 64  # wrong checksum

    with patch("urllib.request.urlopen", return_value=_fake_urlopen(content)):
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            _download_file("http://example.com/archive.zip", tmp_path / "out.zip", bad_sha)

    # Temp file must be cleaned up on failure
    assert not list(tmp_path.glob("*.tmp"))


def test_download_file_no_checksum_skips_verify(tmp_path):
    content = b"some bytes"
    with patch("urllib.request.urlopen", return_value=_fake_urlopen(content)):
        dest = tmp_path / "out.zip"
        _download_file("http://example.com/out.zip", dest, None)
    assert dest.exists()


def test_download_file_network_error(tmp_path):
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
        with pytest.raises(RuntimeError, match="Cannot reach"):
            _download_file("http://example.com/out.zip", tmp_path / "out.zip", None)


# ── Flat-zip loader tests ─────────────────────────────────────────────────────


def test_flat_zip_loader(tmp_path):
    raw, _ = _make_flat_zip(tmp_path, n_images=2)
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    (archive_dir / "sample.zip").write_bytes(raw)
    _flat_zip_loader(archive_dir, dataset_dir, "sample.zip")

    assert (dataset_dir / "images").is_dir()
    assert (dataset_dir / "ground_truth").is_dir()
    assert len(list((dataset_dir / "images").glob("*.tif"))) == 2
    assert len(list((dataset_dir / "ground_truth").glob("*.tif"))) == 2


def test_flat_zip_loader_missing_images_raises(tmp_path):
    # Zip without an images/ folder
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("other/file.txt", "hi")
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (archive_dir / "bad.zip").write_bytes(buf.getvalue())

    with pytest.raises(RuntimeError, match="images/"):
        _flat_zip_loader(archive_dir, dataset_dir, "bad.zip")


# ── fetch_dataset end-to-end tests ────────────────────────────────────────────


def _register_mock_dataset(raw: bytes, sha256: str) -> None:
    """Temporarily register a test dataset using the flat-zip loader."""
    from microagent.data.datasets import _REGISTRY, _flat_zip_loader

    _REGISTRY["_test_mock"] = DatasetSpec(
        name="_test_mock",
        description="Mock dataset for unit tests",
        license="CC0",
        citation="N/A",
        urls=["http://example.com/mock_dataset.zip"],
        sha256s=[sha256],
        loader=lambda ad, dd: _flat_zip_loader(ad, dd, "mock_dataset.zip"),
    )


def test_fetch_dataset_full_flow(tmp_path):
    from microagent.data.datasets import _REGISTRY

    raw, sha256 = _make_flat_zip(tmp_path, n_images=4)
    _register_mock_dataset(raw, sha256)

    try:
        with patch("urllib.request.urlopen", return_value=_fake_urlopen(raw)):
            result = fetch_dataset("_test_mock", cache_dir=tmp_path)

        assert result == tmp_path / "_test_mock"
        assert (result / "images").is_dir()
        assert (result / "ground_truth").is_dir()
        assert len(list((result / "images").glob("*.tif"))) == 4
        assert len(list((result / "ground_truth").glob("*.tif"))) == 4
    finally:
        _REGISTRY.pop("_test_mock", None)


def test_fetch_dataset_skips_if_cached(tmp_path):
    from microagent.data.datasets import _REGISTRY

    raw, sha256 = _make_flat_zip(tmp_path, n_images=2)
    _register_mock_dataset(raw, sha256)

    try:
        # Pre-populate cache
        cached_images = tmp_path / "_test_mock" / "images"
        cached_images.mkdir(parents=True)

        with patch("urllib.request.urlopen") as mock_urlopen:
            fetch_dataset("_test_mock", cache_dir=tmp_path)
            mock_urlopen.assert_not_called()
    finally:
        _REGISTRY.pop("_test_mock", None)


def test_fetch_dataset_unknown_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        fetch_dataset("does-not-exist", cache_dir=tmp_path)


def test_fetch_dataset_redownloads_on_checksum_mismatch(tmp_path):
    from microagent.data.datasets import _REGISTRY

    raw, sha256 = _make_flat_zip(tmp_path, n_images=2)
    _register_mock_dataset(raw, sha256)

    try:
        # Pre-place a corrupt archive
        archive_dir = tmp_path / "_test_mock_archives"
        archive_dir.mkdir(parents=True)
        (archive_dir / "mock_dataset.zip").write_bytes(b"corrupt data")

        with patch("urllib.request.urlopen", return_value=_fake_urlopen(raw)):
            result = fetch_dataset("_test_mock", cache_dir=tmp_path)

        assert (result / "images").is_dir()
    finally:
        _REGISTRY.pop("_test_mock", None)


# ── cache_root tests ──────────────────────────────────────────────────────────


def test_cache_root_override(tmp_path):
    assert cache_root(tmp_path) == tmp_path


def test_cache_root_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MICROAGENT_CACHE_DIR", str(tmp_path))
    assert cache_root() == tmp_path


def test_cache_root_default(monkeypatch):
    monkeypatch.delenv("MICROAGENT_CACHE_DIR", raising=False)
    result = cache_root()
    assert "microagent" in str(result)
    assert "datasets" in str(result)


# ── CLI command tests ─────────────────────────────────────────────────────────


def test_cli_fetch_dataset_list(tmp_path):
    from typer.testing import CliRunner

    from microagent.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["fetch-dataset", "--list", "_placeholder"])
    # --list exits before using the name argument
    assert result.exit_code == 0
    assert "dsb2018" in result.output


def test_cli_fetch_dataset_unknown(tmp_path):
    from typer.testing import CliRunner

    from microagent.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["fetch-dataset", "definitely-not-a-real-dataset", "--cache-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "Unknown" in result.output or "Error" in result.output


def test_cli_optimize_dataset_shortcut(tmp_path, monkeypatch):
    """--dataset on optimize: fetch mocked, then pass dirs to run_optimization."""
    from typer.testing import CliRunner

    from microagent.cli import app
    from microagent.data.datasets import _REGISTRY

    raw, sha256 = _make_flat_zip(tmp_path, n_images=2)
    _register_mock_dataset(raw, sha256)

    try:
        # Pre-populate cache so no HTTP needed
        ds_root = tmp_path / "_test_mock"
        images_dir = ds_root / "images"
        gt_dir = ds_root / "ground_truth"
        images_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for i in range(2):
            img = np.zeros((64, 64), dtype=np.uint16)
            img[10:20, 10:20] = i + 1
            tifffile.imwrite(images_dir / f"img_{i:03d}.tif", img)
            tifffile.imwrite(gt_dir / f"img_{i:03d}.tif", (img > 0).astype(np.uint16))

        from microagent.core.optimize import OptimizationResult

        fake_result = OptimizationResult(
            best_params={"diameter": 15.0},
            best_value=0.75,
            baseline_value=0.60,
            improvement=0.15,
            trials=[],
            study_path=None,
        )

        runner = CliRunner()
        monkeypatch.chdir(tmp_path)

        with patch("microagent.core.optimize.run_optimization", return_value=fake_result):
            result = runner.invoke(
                app,
                [
                    "--no-track",
                    "optimize",
                    "--dataset",
                    "_test_mock",
                    "--dataset-cache",
                    str(tmp_path),
                    "--trials",
                    "1",
                    "--output-json",
                    str(tmp_path / "opt.json"),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Best Hyperparameters" in result.output
    finally:
        _REGISTRY.pop("_test_mock", None)


def test_cli_optimize_missing_args(tmp_path):
    """optimize without --dataset or positional args should exit 1."""
    from typer.testing import CliRunner

    from microagent.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["optimize", "--trials", "1"])
    assert result.exit_code != 0
