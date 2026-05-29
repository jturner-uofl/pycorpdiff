"""Known-answer tests for orthogonal Procrustes alignment."""

from __future__ import annotations

import numpy as np
import pytest

from pycorpdiff.semantic.alignment import procrustes_align


def test_identity_align_is_identity() -> None:
    # Aligning a matrix to itself should leave it unchanged (R = I).
    rng = np.random.default_rng(seed=0)
    a = rng.standard_normal((20, 5))
    aligned = procrustes_align(a, a)
    np.testing.assert_allclose(aligned, a, atol=1e-12)


def test_recovers_known_rotation() -> None:
    # Build target = source @ R0, recover R0 via Procrustes.
    rng = np.random.default_rng(seed=42)
    source = rng.standard_normal((30, 4))
    # Random orthogonal R0 from QR of a random matrix.
    q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    target = source @ q
    aligned = procrustes_align(source, target)
    # Recovered alignment should bring source onto target precisely.
    np.testing.assert_allclose(aligned, target, atol=1e-10)


def test_preserves_row_norms() -> None:
    rng = np.random.default_rng(seed=1)
    source = rng.standard_normal((10, 6))
    # Normalise each row so we have a clean invariant to check.
    source = source / np.linalg.norm(source, axis=1, keepdims=True)
    target = rng.standard_normal((10, 6))
    aligned = procrustes_align(source, target)
    aligned_norms = np.linalg.norm(aligned, axis=1)
    np.testing.assert_allclose(aligned_norms, 1.0, atol=1e-10)


def test_shape_mismatch_raises() -> None:
    a = np.zeros((10, 3))
    b = np.zeros((10, 4))
    with pytest.raises(ValueError, match="same shape"):
        procrustes_align(a, b)


def test_minimises_frobenius_distance() -> None:
    """Procrustes-aligned source should be closer to target than the un-aligned one,
    for non-trivial inputs."""
    rng = np.random.default_rng(seed=3)
    source = rng.standard_normal((20, 5))
    q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    target = source @ q
    pre = np.linalg.norm(source - target)
    post = np.linalg.norm(procrustes_align(source, target) - target)
    assert post < pre
