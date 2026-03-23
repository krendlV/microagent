"""Shared fixtures for integration tests.

Session-scoped synthetic datasets are generated once and reused across all
integration tests, keeping the total runtime low while exercising the full
pipeline on realistic data.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    """Small clean synthetic dataset (session scope).

    5 images, 256×256, well-separated objects — designed to be
    easily segmentable by CellPose with F1 > 0.5.
    """
    base = tmp_path_factory.mktemp("synthetic_clean")
    from microagent.demo.synthetic import generate_synthetic_dataset

    image_dir, gt_dir = generate_synthetic_dataset(
        base,
        n_images=5,
        image_size=(256, 256),
        n_objects_range=(5, 12),
        noise_level=0.05,
        seed=42,
    )
    return base, image_dir, gt_dir


@pytest.fixture(scope="session")
def challenging_data(tmp_path_factory):
    """Challenging synthetic dataset with touching/border objects (session scope)."""
    base = tmp_path_factory.mktemp("synthetic_challenging")
    from microagent.demo.synthetic import generate_challenging_dataset

    image_dir, gt_dir = generate_challenging_dataset(
        base,
        n_images=5,
        image_size=(256, 256),
        n_objects_range=(8, 20),
        noise_level=0.12,
        seed=99,
    )
    return base, image_dir, gt_dir
