"""Dataclasses for project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ChannelConfig:
    """Configuration for a single imaging channel."""

    name: str
    index: int
    target: str  # nuclei, membrane, cytoplasm, etc.


@dataclass
class ComputeConfig:
    """Compute environment description."""

    gpu_model: str | None = None
    vram_gb: float | None = None
    ram_gb: float | None = None


@dataclass
class ProjectConfig:
    """Full project configuration."""

    name: str
    organism: str
    sample_type: str
    modality: str
    structures: list[str]
    channels: list[ChannelConfig]
    image_format: str
    bit_depth: int
    has_ground_truth: bool
    analysis_goal: str
    data_dir: Path
    typical_dimensions: tuple[int, int] | None = None
    ground_truth_format: str | None = None
    gt_dir: Path | None = None
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    recommended_model: str = ""
    recommended_params: dict = field(default_factory=dict)
