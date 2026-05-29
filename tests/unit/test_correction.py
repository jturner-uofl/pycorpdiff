"""Known-answer tests for multiple-comparison correction."""

from __future__ import annotations

import numpy as np

from pycorpdiff.keyness.correction import benjamini_hochberg, bonferroni


def test_bh_monotonicity() -> None:
    # The adjusted p-values must be monotonic with the input.
    raw = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    adj = benjamini_hochberg(raw)
    sorted_pairs = sorted(zip(raw, adj, strict=True))
    sorted_adj = [a for _, a in sorted_pairs]
    assert sorted_adj == sorted(sorted_adj)


def test_bh_smallest_pvalue() -> None:
    # For BH, the smallest p has adjusted = min over the sequence of p_(k)*n/k.
    # For raw [0.001, 0.5, 0.9] with n=3: raw[0]*3/1=0.003, 0.5*3/2=0.75, 0.9*3/3=0.9.
    # Cumulative min from right gives [0.003, 0.75, 0.9].
    adj = benjamini_hochberg(np.array([0.001, 0.5, 0.9]))
    np.testing.assert_allclose(adj, [0.003, 0.75, 0.9], rtol=1e-12)


def test_bh_preserves_order_of_input() -> None:
    raw = np.array([0.5, 0.001, 0.9])
    adj = benjamini_hochberg(raw)
    # Adjusted for 0.001 should be the smallest of the three.
    assert adj[1] < adj[0] and adj[1] < adj[2]


def test_bh_clips_to_unit_interval() -> None:
    raw = np.array([0.8, 0.9, 0.95])
    adj = benjamini_hochberg(raw)
    assert (adj <= 1.0).all() and (adj >= 0.0).all()


def test_bh_empty_input() -> None:
    out = benjamini_hochberg(np.array([], dtype=np.float64))
    assert out.size == 0


def test_bonferroni_basic() -> None:
    raw = np.array([0.01, 0.04, 0.5])
    adj = bonferroni(raw)
    np.testing.assert_allclose(adj, [0.03, 0.12, 1.0], rtol=1e-12)
