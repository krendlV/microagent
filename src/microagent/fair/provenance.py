"""Auto-collection of run metadata for reproducibility and provenance tracking."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class RunMetadata:
    """Environment and run parameters collected automatically for reproducibility.

    Parameters
    ----------
    timestamp : str
        ISO-8601 UTC timestamp of when the run started.
    microagent_version : str
        Version of the microagent package.
    python_version : str
        Python version string (major.minor.patch).
    platform : str
        OS and hardware platform string.
    cpu_count : int | None
        Number of logical CPUs available, or None if unknown.
    hostname : str
        Machine hostname.
    command : str
        CLI command or description of the operation that was run.
    seed : int | None
        Random seed used, if any.
    software_versions : dict[str, str]
        Versions of key dependencies found in the environment.
    """

    timestamp: str
    microagent_version: str
    python_version: str
    platform: str
    cpu_count: int | None
    hostname: str
    command: str
    seed: int | None
    software_versions: dict[str, str]

    @classmethod
    def collect(cls, command: str = "", seed: int | None = None) -> RunMetadata:
        """Collect metadata from the current runtime environment.

        Parameters
        ----------
        command:
            Short description of the operation being performed.
        seed:
            Random seed used in the run, if applicable.

        Returns
        -------
        RunMetadata
            Populated metadata instance.
        """
        from microagent import __version__

        py = sys.version_info
        python_ver = f"{py.major}.{py.minor}.{py.micro}"

        software_versions: dict[str, str] = {
            "microagent": __version__,
            "python": python_ver,
        }
        for pkg in ("numpy", "cellpose", "stardist", "tifffile", "PIL", "scipy", "skimage"):
            try:
                mod = importlib.import_module(pkg)
                ver = getattr(mod, "__version__", None)
                if ver:
                    software_versions[pkg] = str(ver)
            except ImportError:
                pass

        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            microagent_version=__version__,
            python_version=python_ver,
            platform=platform.platform(),
            cpu_count=os.cpu_count(),
            hostname=platform.node(),
            command=command,
            seed=seed,
            software_versions=software_versions,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)
