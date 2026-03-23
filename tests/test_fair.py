"""Tests for fair/provenance.py and fair/tracking.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from microagent.fair.provenance import RunMetadata, collect_metadata, hash_directory
from microagent.fair.tracking import ExperimentTracker, tracked_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> RunMetadata:
    """Return a minimal RunMetadata for tests."""
    defaults = dict(
        microagent_version="0.1.0",
        python_version="3.10.0",
        platform="Linux-6.5.0-x86_64",
        cellpose_version=None,
        stardist_version=None,
        torch_version="2.0.0",
        numpy_version="1.24.0",
        cuda_version=None,
        gpu_name=None,
        gpu_vram_mb=None,
        cpu_model="x86_64",
        ram_total_gb=16.0,
        data_hash="abc123",
        parameters={"model": "cyto3"},
        random_seed=42,
        timestamp_utc="2026-03-23T00:00:00+00:00",
        wall_clock_seconds=1.5,
        git_commit="deadbeef",
        git_dirty=False,
        command="microagent segment .",
    )
    defaults.update(overrides)
    return RunMetadata(**defaults)


# ---------------------------------------------------------------------------
# provenance tests
# ---------------------------------------------------------------------------


def test_collect_metadata_all_fields():
    """collect_metadata() should return a RunMetadata with all fields populated."""
    meta = collect_metadata(
        command="microagent segment .",
        parameters={"model": "cyto3"},
        random_seed=7,
    )
    assert isinstance(meta, RunMetadata)
    assert meta.microagent_version != ""
    assert meta.python_version != ""
    assert meta.platform != ""
    assert meta.numpy_version not in ("", "unknown")
    assert meta.torch_version != ""
    assert meta.cpu_model != ""
    assert meta.ram_total_gb >= 0.0
    assert meta.timestamp_utc != ""
    assert meta.command == "microagent segment ."
    assert meta.random_seed == 7
    assert meta.parameters == {"model": "cyto3"}
    # GPU fields may be None (no CUDA in CI), just ensure they exist
    assert hasattr(meta, "cuda_version")
    assert hasattr(meta, "gpu_name")
    assert hasattr(meta, "gpu_vram_mb")


def test_collect_metadata_no_gpu():
    """When torch.cuda is unavailable, GPU fields should be None."""
    with patch("microagent.fair.provenance._torch_info", return_value=("2.0.0", None, None, None)):
        meta = collect_metadata()
    assert meta.cuda_version is None
    assert meta.gpu_name is None
    assert meta.gpu_vram_mb is None
    assert meta.torch_version == "2.0.0"


def test_collect_metadata_git_fallback():
    """When not in a git repo, git fields should be None."""
    with patch("microagent.fair.provenance._git_info", return_value=(None, None)):
        meta = collect_metadata()
    assert meta.git_commit is None
    assert meta.git_dirty is None


def test_hash_directory_deterministic(tmp_path: Path):
    """Same directory contents always produce the same hash."""
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    h1 = hash_directory(tmp_path)
    h2 = hash_directory(tmp_path)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_hash_directory_different(tmp_path: Path):
    """Different file contents produce different hashes."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "data.txt").write_text("content_A")
    (dir_b / "data.txt").write_text("content_B")
    assert hash_directory(dir_a) != hash_directory(dir_b)


def test_hash_single_file(tmp_path: Path):
    """hash_directory works on a single file path."""
    f = tmp_path / "img.tif"
    f.write_bytes(b"\x00\x01\x02")
    h = hash_directory(f)
    assert len(h) == 64


def test_run_metadata_frozen():
    """RunMetadata is frozen (immutable)."""
    meta = _make_metadata()
    with pytest.raises((AttributeError, TypeError)):
        meta.command = "changed"  # type: ignore[misc]


def test_run_metadata_to_dict():
    """to_dict() returns a plain dict with all expected keys."""
    meta = _make_metadata()
    d = meta.to_dict()
    assert isinstance(d, dict)
    for field in (
        "microagent_version",
        "python_version",
        "platform",
        "torch_version",
        "numpy_version",
        "cpu_model",
        "ram_total_gb",
        "data_hash",
        "parameters",
        "random_seed",
        "timestamp_utc",
        "wall_clock_seconds",
        "git_commit",
        "git_dirty",
        "command",
    ):
        assert field in d


# ---------------------------------------------------------------------------
# tracking tests
# ---------------------------------------------------------------------------


def test_tracker_log_and_retrieve(tmp_path: Path):
    """log_run() returns a run_id; get_run() retrieves the same record."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    meta = _make_metadata()
    results = {"ap50": 0.87, "f1": 0.91}

    run_id = tracker.log_run(meta, results)
    assert len(run_id) == 8

    record = tracker.get_run(run_id)
    assert record["run_id"] == run_id
    assert record["results"]["ap50"] == pytest.approx(0.87)
    assert record["metadata"]["command"] == "microagent segment ."


def test_tracker_list_runs(tmp_path: Path):
    """list_runs(last_n=2) returns only the 2 most recent of 3 runs."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    ids = []
    for i in range(3):
        meta = _make_metadata(command=f"run_{i}")
        ids.append(tracker.log_run(meta, {"i": i}))

    recent = tracker.list_runs(last_n=2)
    assert len(recent) == 2
    assert recent[0]["run_id"] == ids[1]
    assert recent[1]["run_id"] == ids[2]


def test_tracker_list_runs_fewer_than_n(tmp_path: Path):
    """list_runs(last_n=10) with only 2 runs returns 2 records."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    for i in range(2):
        tracker.log_run(_make_metadata(command=f"run_{i}"), {})
    assert len(tracker.list_runs(last_n=10)) == 2


def test_tracker_jsonl_format(tmp_path: Path):
    """Each line in the JSONL file must be a valid, self-contained JSON object."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    for i in range(3):
        tracker.log_run(_make_metadata(command=f"cmd_{i}"), {"x": i})

    lines = (tmp_path / "exp.jsonl").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "run_id" in obj
        assert "metadata" in obj
        assert "results" in obj


def test_tracker_get_run_missing(tmp_path: Path):
    """get_run() raises KeyError for an unknown run_id."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    with pytest.raises(KeyError):
        tracker.get_run("deadbeef")


def test_tracker_compare_runs(tmp_path: Path):
    """compare_runs() reports differing fields between two runs."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    id_a = tracker.log_run(_make_metadata(command="cmd_a"), {"ap50": 0.8})
    id_b = tracker.log_run(_make_metadata(command="cmd_b"), {"ap50": 0.9})

    diff = tracker.compare_runs(id_a, id_b)
    assert "metadata.command" in diff
    assert diff["metadata.command"] == {"a": "cmd_a", "b": "cmd_b"}
    assert "results.ap50" in diff


def test_tracked_run_context_manager(tmp_path: Path):
    """tracked_run() times the block and writes a record to the tracker."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    with tracked_run(tracker, "microagent segment .", {"model": "cyto3"}) as results:
        results["ap50"] = 0.75

    records = tracker.list_runs()
    assert len(records) == 1
    assert records[0]["results"]["ap50"] == pytest.approx(0.75)
    assert records[0]["metadata"]["command"] == "microagent segment ."
    assert records[0]["metadata"]["wall_clock_seconds"] >= 0.0


def test_tracked_run_logs_on_exception(tmp_path: Path):
    """tracked_run() still logs the record when an exception occurs."""
    tracker = ExperimentTracker(tmp_path / "exp.jsonl")
    with pytest.raises(RuntimeError):
        with tracked_run(tracker, "cmd", {}) as results:
            results["partial"] = True
            raise RuntimeError("boom")

    records = tracker.list_runs()
    assert len(records) == 1
    assert records[0]["results"]["partial"] is True
