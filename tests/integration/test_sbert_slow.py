"""Slow integration tests that exercise the real SBERT embedder.

These tests download the ``all-MiniLM-L6-v2`` model (~22 MB) on first
run and skip silently if ``sentence-transformers`` isn't installed.
They're marked ``slow`` so the default ``pytest`` run skips them; the
dedicated ``sbert-slow`` CI job opts in with ``-m slow``.

The point of these tests isn't to validate SBERT itself — that's
upstream's job — but to verify that pycorpdiff's wiring (lazy import,
encode-shape contract, semantic_shift orchestration, neighbor_drift
end-to-end) survives contact with a real model.
"""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd

# Skip the entire module if sentence-transformers isn't installed.
sentence_transformers = pytest.importorskip("sentence_transformers")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def frame_corpus() -> pcd.Corpus:
    pos = [
        "the migrant worker arrived and the migrant family settled here",
        "the migrant community grew and migrant workers thrived together",
        "the migrant family settled and the migrant community welcomed them",
    ]
    neg = [
        "the migrant criminal threat and the migrant invasion grew worse",
        "the migrant threat and the migrant crime increased again",
        "the migrant invasion of migrant criminal gangs spread further",
    ]
    rows = [{"text": d, "frame": "humanising"} for d in pos] + [
        {"text": d, "frame": "criminalising"} for d in neg
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("frame",))


@pytest.fixture(scope="module")
def sbert_embedder() -> pcd.SBERTEmbedder:
    """Construct + warm the SBERT embedder.

    Skip the whole module if the model can't be loaded (network flake,
    HuggingFace outage, gated-model auth gap, transformers version
    mismatch). The point of these tests is pycorpdiff's wiring, not
    SBERT's; an upstream-download failure is no signal.
    """
    e = pcd.SBERTEmbedder(model_name="all-MiniLM-L6-v2")
    try:
        e.encode(["warmup"])  # triggers the lazy model download
    except (OSError, ValueError, RuntimeError) as exc:  # pragma: no cover
        pytest.skip(f"SBERT model unavailable: {exc}")
    return e


def test_sbert_encode_returns_correct_shape(
    sbert_embedder: pcd.SBERTEmbedder,
) -> None:
    vectors = sbert_embedder.encode(["hello world", "goodbye world", "third sentence"])
    assert vectors.shape[0] == 3
    # all-MiniLM-L6-v2 has 384-dimensional output.
    assert vectors.shape[1] == 384


def test_sbert_encode_is_deterministic(
    sbert_embedder: pcd.SBERTEmbedder,
) -> None:
    # Same input twice → identical vectors.
    a = sbert_embedder.encode(["the migrant worker"])
    b = sbert_embedder.encode(["the migrant worker"])
    import numpy as np

    np.testing.assert_allclose(a, b, atol=1e-6)


def test_semantic_shift_with_real_sbert(
    frame_corpus: pcd.Corpus, sbert_embedder: pcd.SBERTEmbedder
) -> None:
    """The two engineered frames should produce non-trivial cosine distance.

    Real semantic test: humanising contexts ("worker", "family",
    "community") should give a different centroid than criminalising
    contexts ("criminal", "threat", "invasion"), and the cosine
    distance should be well above zero.
    """
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        target="migrant", embedder=sbert_embedder, window=4
    )
    distance = float(result.table["cosine_distance"].iloc[0])
    # SBERT cosine distances between meaningfully-different contexts
    # are typically 0.1–0.5; we require strictly > 0.05 as a sanity
    # threshold without overfitting the test to a specific value.
    assert distance > 0.05
    assert distance < 1.0


def test_semantic_shift_self_comparison_is_near_zero(
    frame_corpus: pcd.Corpus, sbert_embedder: pcd.SBERTEmbedder
) -> None:
    """Same corpus on both sides → cosine distance ≈ 0."""
    a = frame_corpus.slice(frame="humanising")
    result = pcd.compare(a, a).semantic_shift(
        target="migrant", embedder=sbert_embedder, window=4
    )
    distance = float(result.table["cosine_distance"].iloc[0])
    assert distance < 1e-6


def test_neighborhood_drift_with_real_sbert(
    frame_corpus: pcd.Corpus, sbert_embedder: pcd.SBERTEmbedder
) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        target="migrant", embedder=sbert_embedder, window=4
    )
    nb = result.neighbors_before(n=5)
    na = result.neighbors_after(n=5)
    # Both should return non-empty neighbour sets on a corpus this size.
    assert len(nb) > 0
    assert len(na) > 0
    # Neighbour similarities should be in the reasonable cosine range.
    assert nb["sim_a"].between(-1.0, 1.0).all()
    assert na["sim_b"].between(-1.0, 1.0).all()
