"""Tests for the Embedder protocol and built-in implementations."""

from __future__ import annotations

import numpy as np
import pytest

from pycorpdiff.semantic.embed import Embedder, HashEmbedder


def test_hash_embedder_is_deterministic() -> None:
    e = HashEmbedder(dim=16)
    a = e.encode(["alpha", "beta", "gamma"])
    b = e.encode(["alpha", "beta", "gamma"])
    np.testing.assert_array_equal(a, b)


def test_hash_embedder_returns_unit_vectors() -> None:
    e = HashEmbedder(dim=24)
    vectors = e.encode(["alpha", "beta", "gamma", "delta", "epsilon"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_hash_embedder_different_strings_give_different_vectors() -> None:
    e = HashEmbedder(dim=16)
    vectors = e.encode(["alpha", "beta", "gamma"])
    # Pairwise dot products should be far from 1.0 (orthogonal-ish vectors).
    similarity = vectors @ vectors.T
    off_diag = similarity[~np.eye(3, dtype=bool)]
    assert (np.abs(off_diag) < 0.95).all()


def test_hash_embedder_shape_matches_input() -> None:
    e = HashEmbedder(dim=8)
    out = e.encode(["a", "b", "c", "d"])
    assert out.shape == (4, 8)


def test_hash_embedder_satisfies_embedder_protocol() -> None:
    e = HashEmbedder()
    assert isinstance(e, Embedder)


def test_hash_embedder_empty_input() -> None:
    e = HashEmbedder(dim=8)
    out = e.encode([])
    assert out.shape == (0, 8)


def test_sbert_lazy_import_does_not_pull_torch() -> None:
    """Constructing SBERTEmbedder must not import torch — only on first
    .encode() call, which lets the base install stay light. This test
    only verifies the construction half (we can't undo prior torch
    imports in this process); the lazy-encode half is exercised when
    the [semantic] extra is installed in a clean environment.
    """
    from pycorpdiff.semantic.embed import SBERTEmbedder

    e = SBERTEmbedder()  # should be cheap, no encode call
    assert e._model is None


def test_sbert_encode_raises_friendly_error_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If sentence-transformers can't be imported, SBERTEmbedder.encode
    should raise an ImportError pointing the user at the extras install.
    """
    # Force ImportError on the lazy import path by aliasing the module to None.
    import sys

    from pycorpdiff.semantic.embed import SBERTEmbedder

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    e = SBERTEmbedder()
    with pytest.raises(ImportError, match="pycorpdiff\\[semantic\\]"):
        e.encode(["hello"])
