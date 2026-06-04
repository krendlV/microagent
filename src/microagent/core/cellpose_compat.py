"""Compatibility helpers for CellPose API differences."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rich.console import Console

console = Console(stderr=True)

_CELLPOSE_V4_DEPRECATION = (4, 0, 1)


def cellpose_version() -> str | None:
    """Return the installed CellPose version, or None when unavailable."""
    try:
        return version("cellpose")
    except PackageNotFoundError:
        return None


def _version_tuple(raw_version: str) -> tuple[int, ...]:
    """Parse leading numeric version components from a package version string."""
    match = re.match(r"^(\d+(?:\.\d+)*)", raw_version)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def is_cellpose_v4_or_newer(raw_version: str | None = None) -> bool:
    """Return True for CellPose versions where v4.0.1 API deprecations apply."""
    installed = cellpose_version() if raw_version is None else raw_version
    if installed is None:
        return False
    return _version_tuple(installed) >= _CELLPOSE_V4_DEPRECATION


def cellpose_model_kwargs(pretrained: str, gpu: bool) -> dict[str, Any]:
    """Build CellposeModel kwargs without using ignored v4 arguments."""
    if is_cellpose_v4_or_newer():
        return {"gpu": gpu, "pretrained_model": pretrained}
    return {"gpu": gpu, "model_type": pretrained}


def warn_cellpose_v4_channels_ignored(channels: list[int]) -> None:
    """Warn that CellPose v4 no longer honours the legacy channels selector."""
    console.print(
        "[yellow]Warning:[/yellow] CellPose v4.0.1+ deprecates the "
        f"channels argument; requested channels={channels!r} will not be "
        "forwarded to CellPose. Provide a preselected image/channel stack "
        "instead."
    )
