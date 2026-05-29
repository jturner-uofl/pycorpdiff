"""Tests for ``SemanticShiftResult.neighbors_before / neighbors_after``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def frame_corpus() -> pcd.Corpus:
    pos = [
        "the migrant worker arrived and the migrant family settled",
        "the migrant community grew and migrant workers thrived",
        "the migrant worker advanced",
    ]
    neg = [
        "the migrant criminal threat and the migrant invasion grew",
        "the migrant invasion of migrant criminal gangs spread",
        "the migrant criminal element alarmed",
    ]
    rows = [{"text": d, "frame": "humanising"} for d in pos] + [
        {"text": d, "frame": "criminalising"} for d in neg
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("frame",))


def test_neighbors_before_returns_dataframe(frame_corpus: pcd.Corpus) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        "migrant", embedder=pcd.HashEmbedder(dim=32), window=3
    )
    nb = result.neighbors_before(n=5)
    assert isinstance(nb, pd.DataFrame)
    assert "neighbor" in nb.columns
    assert "sim_a" in nb.columns
    # neighbors_before keeps only rows where sim_a is populated.
    assert nb["sim_a"].notna().all()
    assert len(nb) <= 5


def test_neighbors_after_returns_dataframe(frame_corpus: pcd.Corpus) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        "migrant", embedder=pcd.HashEmbedder(dim=32), window=3
    )
    na = result.neighbors_after(n=5)
    assert na["sim_b"].notna().all()


def test_neighbors_sorted_by_similarity(frame_corpus: pcd.Corpus) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        "migrant", embedder=pcd.HashEmbedder(dim=32), window=3
    )
    nb = result.neighbors_before(n=10)
    assert (nb["sim_a"].diff().dropna() <= 1e-12).all()


def test_neighbors_without_source_corpora_raises() -> None:
    bare = pcd.SemanticShiftResult(
        targets=["x"],
        table=pd.DataFrame({"target": ["x"], "cosine_distance": [0.5]}),
        alignment="none",
    )
    with pytest.raises(ValueError, match="require source corpora"):
        bare.neighbors_before()


def test_neighbors_multi_target_requires_explicit(
    frame_corpus: pcd.Corpus,
) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        ["migrant", "the"], embedder=pcd.HashEmbedder(dim=32), window=3
    )
    with pytest.raises(ValueError, match="pass target= to pick one"):
        result.neighbors_before()


def test_neighbors_unknown_target_raises(frame_corpus: pcd.Corpus) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        "migrant", embedder=pcd.HashEmbedder(dim=32), window=3
    )
    with pytest.raises(ValueError, match="not in result targets"):
        result.neighbors_before(target="unicorn")
