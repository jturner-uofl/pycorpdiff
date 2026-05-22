"""Unit tests for semantic_shift and neighborhood_drift.

All tests use HashEmbedder for determinism — the math under test is
the averaging / cosine / Procrustes pipeline, not the embedding model.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.semantic.embed import HashEmbedder
from pycorpdiff.semantic.shift import neighborhood_drift, semantic_shift


def _corpus(texts: list[str]) -> pcd.Corpus:
    return pcd.from_dataframe(pd.DataFrame({"text": texts}), text_col="text")


def test_semantic_shift_zero_for_identical_corpora() -> None:
    a = _corpus(["the migrant worker arrived and the family settled"])
    b = _corpus(["the migrant worker arrived and the family settled"])
    df = semantic_shift(a, b, "migrant", embedder=HashEmbedder(), window=3)
    assert math.isclose(df["cosine_distance"].iloc[0], 0.0, abs_tol=1e-12)
    assert math.isclose(df["cosine_similarity"].iloc[0], 1.0, abs_tol=1e-12)


def test_semantic_shift_positive_for_different_contexts() -> None:
    a = _corpus(["the migrant worker family arrived peacefully"] * 3)
    b = _corpus(["the migrant criminal threat invasion grew"] * 3)
    df = semantic_shift(a, b, "migrant", embedder=HashEmbedder(), window=3)
    # With HashEmbedder, identical windows → identical vectors → identical
    # centroids. Different windows → different vectors → cosine < 1.
    assert df["cosine_distance"].iloc[0] > 0.1


def test_semantic_shift_records_context_counts() -> None:
    a = _corpus(["migrant migrant filler", "migrant alone here"])
    b = _corpus(["migrant once only"])
    df = semantic_shift(a, b, "migrant", embedder=HashEmbedder(), window=2)
    assert df["n_contexts_a"].iloc[0] == 3
    assert df["n_contexts_b"].iloc[0] == 1


def test_semantic_shift_returns_nan_when_target_absent() -> None:
    a = _corpus(["the cat sat"])
    b = _corpus(["migrant worker"])
    df = semantic_shift(a, b, "migrant", embedder=HashEmbedder())
    assert math.isnan(df["cosine_distance"].iloc[0])
    assert df["n_contexts_a"].iloc[0] == 0
    assert df["n_contexts_b"].iloc[0] == 1


def test_semantic_shift_multi_target() -> None:
    a = _corpus(["the migrant worker arrived and the asylum seeker stayed"] * 2)
    b = _corpus(["the migrant criminal threat and the asylum invasion grew"] * 2)
    df = semantic_shift(a, b, ["migrant", "asylum"], embedder=HashEmbedder())
    assert set(df["target"]) == {"migrant", "asylum"}
    assert (df["n_contexts_a"] > 0).all()


def test_semantic_shift_procrustes_runs_end_to_end() -> None:
    a = _corpus(["the migrant worker family arrived"] * 4)
    b = _corpus(["the migrant criminal threat invasion"] * 4)
    df_no_align = semantic_shift(
        a, b, "migrant", embedder=HashEmbedder(), align="none"
    )
    df_procrustes = semantic_shift(
        a, b, "migrant", embedder=HashEmbedder(), align="procrustes"
    )
    # After Procrustes alignment (which rotates source to fit target),
    # cosine similarity should be ≥ the un-aligned value — alignment can
    # only improve the fit.
    sim_aligned = df_procrustes["cosine_similarity"].iloc[0]
    sim_unaligned = df_no_align["cosine_similarity"].iloc[0]
    assert sim_aligned >= sim_unaligned - 1e-9


def test_semantic_shift_swap_symmetry_under_hash_embedder() -> None:
    a = _corpus(["the migrant worker family arrived"] * 2)
    b = _corpus(["the migrant criminal threat invasion"] * 2)
    forward = semantic_shift(a, b, "migrant", embedder=HashEmbedder())
    reverse = semantic_shift(b, a, "migrant", embedder=HashEmbedder())
    # Cosine similarity is symmetric under corpus swap.
    assert math.isclose(
        forward["cosine_similarity"].iloc[0],
        reverse["cosine_similarity"].iloc[0],
        abs_tol=1e-12,
    )


def test_neighborhood_drift_returns_expected_columns() -> None:
    a = _corpus(
        [
            "the migrant worker arrived and the migrant family settled",
            "the migrant community welcomed the migrant family",
            "the migrant worker thrived",
        ]
    )
    b = _corpus(
        [
            "the migrant criminal threat and the migrant invasion grew",
            "the migrant criminal gangs and the migrant invasion stayed",
            "the migrant threat persisted",
        ]
    )
    df = neighborhood_drift(
        a, b, "migrant", k=5, embedder=HashEmbedder(), window=3, min_count=1
    )
    assert set(df.columns) >= {"neighbor", "sim_a", "sim_b", "rank_a", "rank_b", "drift", "status"}
    assert (df["neighbor"] != "migrant").all()
    statuses = set(df["status"])
    assert statuses.issubset({"shared", "gained_in_a", "lost_in_a"})


def test_neighborhood_drift_target_absent_raises() -> None:
    a = _corpus(["the cat sat"])
    b = _corpus(["the migrant worker"])
    with pytest.raises(ValueError, match="not found in corpus a"):
        neighborhood_drift(a, b, "migrant", embedder=HashEmbedder())
