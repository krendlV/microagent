"""Auto-collection of run metadata for reproducibility and provenance tracking."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunMetadata:
    """Environment and run parameters collected automatically for reproducibility.

    Parameters
    ----------
    microagent_version : str
        Version of the microagent package.
    python_version : str
        Python interpreter version string.
    platform : str
        OS platform identifier (e.g., "Linux-6.5.0-x86_64").
    cellpose_version : str | None
        Installed cellpose version, or None if not installed.
    stardist_version : str | None
        Installed stardist version, or None if not installed.
    torch_version : str
        Installed torch version.
    numpy_version : str
        Installed numpy version.
    cuda_version : str | None
        CUDA version reported by torch, or None.
    gpu_name : str | None
        Name of the first CUDA device, or None.
    gpu_vram_mb : int | None
        Total VRAM of the first CUDA device in MiB, or None.
    cpu_model : str
        CPU model string from platform.processor().
    ram_total_gb : float
        Total system RAM in GiB.
    data_hash : str
        SHA-256 of all input files (sorted by path).
    parameters : dict
        All run parameters including defaults.
    random_seed : int
        Random seed used for the run.
    timestamp_utc : str
        ISO 8601 UTC timestamp of run start.
    wall_clock_seconds : float
        Elapsed wall-clock time in seconds.
    git_commit : str | None
        Git HEAD commit SHA, or None if not in a git repo.
    git_dirty : bool | None
        True if there are uncommitted changes, None if not in a git repo.
    command : str
        Full CLI command string that was run.
    """

    microagent_version: str
    python_version: str
    platform: str
    cellpose_version: str | None
    stardist_version: str | None
    torch_version: str
    numpy_version: str
    cuda_version: str | None
    gpu_name: str | None
    gpu_vram_mb: int | None
    cpu_model: str
    ram_total_gb: float
    data_hash: str
    parameters: dict
    random_seed: int
    timestamp_utc: str
    wall_clock_seconds: float
    git_commit: str | None
    git_dirty: bool | None
    command: str

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def collect(cls, command: str = "", seed: int = 0, **kwargs) -> RunMetadata:
        """Collect run metadata using the canonical module-level collector."""
        return collect_metadata(command=command, random_seed=seed, **kwargs)


def _pkg_version(name: str) -> str | None:
    """Return installed package version or None."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_info() -> tuple[str, str | None, str | None, int | None]:
    """Return (torch_version, cuda_version, gpu_name, gpu_vram_mb)."""
    torch_ver = _pkg_version("torch") or "unknown"
    cuda_ver: str | None = None
    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    try:
        import torch

        torch_ver = torch.__version__
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except ImportError:
        pass
    return torch_ver, cuda_ver, gpu_name, gpu_vram_mb


def _ram_total_gb() -> float:
    """Return total system RAM in GiB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    # Fallback: parse /proc/meminfo on Linux
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb / (1024**2)
    except Exception:
        pass
    return 0.0


def _git_info() -> tuple[str | None, bool | None]:
    """Return (commit_sha, is_dirty) or (None, None) if not in a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return commit, bool(dirty_output.strip())
    except Exception:
        return None, None


def hash_directory(path: Path) -> str:
    """Compute a deterministic SHA-256 hash over all files in *path*.

    Parameters
    ----------
    path:
        Directory (or single file) to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    target = Path(path)
    files = [target] if target.is_file() else sorted(target.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        # Include relative path so renames change the hash
        h.update(str(f.relative_to(target) if target.is_dir() else f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def collect_metadata(
    command: str = "",
    parameters: dict | None = None,
    random_seed: int = 0,
    data_path: Path | None = None,
    wall_clock_seconds: float = 0.0,
) -> RunMetadata:
    """Collect metadata from the current runtime environment.

    Parameters
    ----------
    command:
        Full CLI command string that was run.
    parameters:
        All run parameters (including defaults).
    random_seed:
        Random seed used for the run.
    data_path:
        Path to input data directory/file for computing data_hash.
    wall_clock_seconds:
        Elapsed wall-clock time in seconds.

    Returns
    -------
    RunMetadata
        Fully populated metadata instance.
    """
    from microagent import __version__

    py = sys.version_info
    python_ver = f"{py.major}.{py.minor}.{py.micro}"

    torch_ver, cuda_ver, gpu_name, gpu_vram_mb = _torch_info()
    git_commit, git_dirty = _git_info()

    numpy_ver = _pkg_version("numpy") or "unknown"
    cellpose_ver = _pkg_version("cellpose")
    stardist_ver = _pkg_version("stardist")

    data_hash = hash_directory(data_path) if data_path is not None else ""

    return RunMetadata(
        microagent_version=__version__,
        python_version=python_ver,
        platform=platform.platform(),
        cellpose_version=cellpose_ver,
        stardist_version=stardist_ver,
        torch_version=torch_ver,
        numpy_version=numpy_ver,
        cuda_version=cuda_ver,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        cpu_model=platform.processor() or platform.machine(),
        ram_total_gb=round(_ram_total_gb(), 2),
        data_hash=data_hash,
        parameters=parameters or {},
        random_seed=random_seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        wall_clock_seconds=wall_clock_seconds,
        git_commit=git_commit,
        git_dirty=git_dirty,
        command=command,
    )
