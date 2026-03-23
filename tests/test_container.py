"""Tests for fair/container.py — Dockerfile, Apptainer, and reproducibility bundle."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from microagent.fair.container import (
    export_reproducibility_bundle,
    generate_apptainer_def,
    generate_dockerfile,
    generate_environment_lock,
)
from microagent.fair.provenance import RunMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_provenance(**overrides) -> RunMetadata:
    defaults = dict(
        microagent_version="0.1.0",
        python_version="3.11.7",
        platform="Linux-6.5.0-x86_64-with-glibc2.35",
        cellpose_version="3.0.1",
        stardist_version=None,
        torch_version="2.1.0",
        numpy_version="1.26.0",
        cuda_version=None,
        gpu_name=None,
        gpu_vram_mb=None,
        cpu_model="x86_64",
        ram_total_gb=16.0,
        data_hash="abc123",
        parameters={"model": "cellpose", "diameter": 30},
        random_seed=42,
        timestamp_utc="2026-03-23T12:00:00+00:00",
        wall_clock_seconds=12.3,
        git_commit="deadbeef01234567",
        git_dirty=False,
        command="microagent segment /data/images",
    )
    defaults.update(overrides)
    return RunMetadata(**defaults)


def _make_experiments_jsonl(tmp_path: Path, run_id: str, meta: RunMetadata) -> Path:
    """Write a minimal experiments.jsonl with a single run entry."""
    record = {
        "run_id": run_id,
        "metadata": meta.to_dict(),
        "results": {"n_cells": 42},
    }
    exp_path = tmp_path / "experiments.jsonl"
    exp_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return exp_path


# ---------------------------------------------------------------------------
# generate_dockerfile
# ---------------------------------------------------------------------------


class TestGenerateDockerfile:
    def test_creates_file(self, tmp_path):
        meta = _make_provenance()
        dest = tmp_path / "Dockerfile"
        result = generate_dockerfile(meta, dest)
        assert result == dest
        assert dest.exists()

    def test_multistage_build(self, tmp_path):
        meta = _make_provenance()
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "AS builder" in content
        assert "AS runtime" in content
        assert "COPY --from=builder" in content

    def test_entrypoint_is_microagent(self, tmp_path):
        meta = _make_provenance()
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "ENTRYPOINT" in content
        assert "microagent" in content

    def test_labels_present(self, tmp_path):
        meta = _make_provenance()
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "LABEL maintainer" in content
        assert "LABEL version" in content
        assert "LABEL description" in content
        assert "LABEL license" in content

    def test_python_version_pinned(self, tmp_path):
        meta = _make_provenance(python_version="3.11.7")
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "3.11.7" in content

    def test_microagent_version_pinned(self, tmp_path):
        meta = _make_provenance(microagent_version="0.2.5")
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "0.2.5" in content

    def test_git_commit_included(self, tmp_path):
        meta = _make_provenance(git_commit="cafebabe")
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "cafebabe" in content

    def test_cpu_uses_python_base(self, tmp_path):
        meta = _make_provenance(cuda_version=None)
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "python:3.11-slim" in content
        assert "nvidia/cuda" not in content

    def test_gpu_uses_cuda_base(self, tmp_path):
        meta = _make_provenance(cuda_version="12.1", gpu_name="NVIDIA A100")
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "nvidia/cuda" in content
        assert "12.1" in content

    def test_mplbackend_set(self, tmp_path):
        meta = _make_provenance()
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        assert "MPLBACKEND=Agg" in content

    def test_creates_parent_dirs(self, tmp_path):
        meta = _make_provenance()
        dest = tmp_path / "subdir" / "nested" / "Dockerfile"
        generate_dockerfile(meta, dest)
        assert dest.exists()

    def test_no_syntax_errors_basic(self, tmp_path):
        """Verify each FROM line has a valid image reference."""
        meta = _make_provenance()
        content = generate_dockerfile(meta, tmp_path / "Dockerfile").read_text()
        from_lines = [l.strip() for l in content.splitlines() if l.startswith("FROM")]
        assert len(from_lines) >= 2
        for line in from_lines:
            # Each FROM must reference an image (contains a colon or slash or AS)
            assert any(c in line for c in [":", "/", "AS"]), f"Suspicious FROM: {line}"


# ---------------------------------------------------------------------------
# generate_apptainer_def
# ---------------------------------------------------------------------------


class TestGenerateApptainerDef:
    def test_creates_file(self, tmp_path):
        meta = _make_provenance()
        dest = tmp_path / "microagent.def"
        result = generate_apptainer_def(meta, dest)
        assert result == dest
        assert dest.exists()

    def test_bootstrap_header(self, tmp_path):
        meta = _make_provenance()
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "Bootstrap:" in content
        assert "From:" in content

    def test_sections_present(self, tmp_path):
        meta = _make_provenance()
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        for section in ["%labels", "%environment", "%post", "%runscript", "%help"]:
            assert section in content, f"Missing section: {section}"

    def test_runscript_calls_microagent(self, tmp_path):
        meta = _make_provenance()
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "microagent" in content

    def test_labels_contain_version(self, tmp_path):
        meta = _make_provenance(microagent_version="0.3.0")
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "0.3.0" in content

    def test_gpu_env_vars_present_for_cuda(self, tmp_path):
        meta = _make_provenance(cuda_version="12.1")
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "NVIDIA_VISIBLE_DEVICES" in content

    def test_gpu_env_vars_absent_for_cpu(self, tmp_path):
        meta = _make_provenance(cuda_version=None)
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "NVIDIA_VISIBLE_DEVICES" not in content

    def test_path_env_set(self, tmp_path):
        meta = _make_provenance()
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "/opt/venv/bin" in content

    def test_git_commit_in_labels(self, tmp_path):
        meta = _make_provenance(git_commit="0000ffff")
        content = generate_apptainer_def(meta, tmp_path / "microagent.def").read_text()
        assert "0000ffff" in content


# ---------------------------------------------------------------------------
# generate_environment_lock
# ---------------------------------------------------------------------------


class TestGenerateEnvironmentLock:
    def test_creates_file(self, tmp_path):
        dest = tmp_path / "requirements.txt"
        result = generate_environment_lock(dest)
        assert result == dest
        assert dest.exists()

    def test_contains_system_info_comments(self, tmp_path):
        dest = tmp_path / "requirements.txt"
        content = generate_environment_lock(dest).read_text()
        assert "# OS:" in content
        assert "# Python:" in content
        assert "# CUDA:" in content

    def test_contains_packages(self, tmp_path):
        dest = tmp_path / "requirements.txt"
        content = generate_environment_lock(dest).read_text()
        # At minimum we know these are installed in the test env
        assert "==" in content  # at least one pinned package

    def test_architecture_recorded(self, tmp_path):
        dest = tmp_path / "requirements.txt"
        content = generate_environment_lock(dest).read_text()
        assert "# Architecture:" in content

    def test_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "nested" / "dir" / "requirements.txt"
        generate_environment_lock(dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# export_reproducibility_bundle
# ---------------------------------------------------------------------------


class TestExportReproducibilityBundle:
    RUN_ID = "abcd1234"

    def _setup(self, tmp_path) -> tuple[Path, Path]:
        meta = _make_provenance()
        exp_path = _make_experiments_jsonl(tmp_path, self.RUN_ID, meta)
        output_dir = tmp_path / "bundles"
        return exp_path, output_dir

    def test_creates_zip(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        assert zip_path.exists()
        assert zip_path.suffix == ".zip"
        assert self.RUN_ID in zip_path.name

    def test_zip_contains_dockerfile(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "Dockerfile" in names

    def test_zip_contains_requirements(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "requirements.txt" in names

    def test_zip_contains_metadata(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "run_metadata.json" in names

    def test_zip_contains_experiments_jsonl(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "experiments.jsonl" in names

    def test_zip_contains_readme(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "README.md" in names

    def test_readme_contains_run_id(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            readme = zf.read("README.md").decode()
        assert self.RUN_ID in readme

    def test_readme_contains_docker_instructions(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            readme = zf.read("README.md").decode()
        assert "docker build" in readme

    def test_metadata_json_parseable(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            data = json.loads(zf.read("run_metadata.json"))
        assert data["run_id"] == self.RUN_ID

    def test_experiments_jsonl_single_entry(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("experiments.jsonl").decode()
        records = [json.loads(l) for l in content.splitlines() if l.strip()]
        assert len(records) == 1
        assert records[0]["run_id"] == self.RUN_ID

    def test_includes_project_yaml_when_provided(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        yaml_path = tmp_path / "project.yaml"
        yaml_path.write_text("name: test-project\n", encoding="utf-8")
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path, project_yaml=yaml_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "project.yaml" in names

    def test_raises_for_missing_experiments(self, tmp_path):
        output_dir = tmp_path / "out"
        with pytest.raises(FileNotFoundError):
            export_reproducibility_bundle(
                "doesntmatter", output_dir, experiments_path=tmp_path / "missing.jsonl"
            )

    def test_raises_for_unknown_run_id(self, tmp_path):
        meta = _make_provenance()
        exp_path = _make_experiments_jsonl(tmp_path, "realrun1", meta)
        output_dir = tmp_path / "out"
        with pytest.raises(KeyError):
            export_reproducibility_bundle(
                "notexist", output_dir, experiments_path=exp_path
            )

    def test_dockerfile_in_bundle_has_entrypoint(self, tmp_path):
        exp_path, output_dir = self._setup(tmp_path)
        zip_path = export_reproducibility_bundle(
            self.RUN_ID, output_dir, experiments_path=exp_path
        )
        with zipfile.ZipFile(zip_path) as zf:
            dockerfile = zf.read("Dockerfile").decode()
        assert "ENTRYPOINT" in dockerfile
        assert "microagent" in dockerfile
