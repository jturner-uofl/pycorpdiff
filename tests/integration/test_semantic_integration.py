"""End-to-end tests for ``compare(a, b).semantic_shift()``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.semantic.embed import HashEmbedder


@pytest.fixture
def frame_corpus() -> pcd.Corpus:
    pos = [
        "the migrant worker arrived and the migrant family settled here",
        "the migrant community grew and migrant workers thrived together",
        "the migrant family settled and the migrant community welcomed them",
        "the migrant worker and migrant rights advanced in the workplace",
    ]
    neg = [
        "the migrant criminal threat and the migrant invasion grew worse",
        "the migrant threat and the migrant crime increased again",
        "the migrant invasion of migrant criminal gangs spread further",
        "the migrant criminal gangs and migrant invasion stayed dangerous",
    ]
    rows = [{"text": d, "frame": "humanising"} for d in pos] + [
        {"text": d, "frame": "criminalising"} for d in neg
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("frame",))


def test_comparison_semantic_shift_returns_result(frame_corpus: pcd.Corpus) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        "migrant", embedder=HashEmbedder()
    )
    assert isinstance(result, pcd.SemanticShiftResult)
    assert result.targets == ["migrant"]
    assert result.alignment == "none"
    assert "cosine_distance" in result.table.columns


def test_comparison_semantic_shift_detects_frame_contrast(
    frame_corpus: pcd.Corpus,
) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift("migrant", embedder=HashEmbedder())
    # Different-frame contexts → contextual centroids differ → non-trivial distance.
    distance = result.table["cosine_distance"].iloc[0]
    assert distance > 0.0


def test_comparison_semantic_shift_multi_target_labels(
    frame_corpus: pcd.Corpus,
) -> None:
    a = frame_corpus.slice(frame="humanising")
    b = frame_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).semantic_shift(
        ["migrant", "the"], embedder=HashEmbedder()
    )
    assert result.label_a == "frame='humanising'"
    assert result.label_b == "frame='criminalising'"
    assert set(result.table["target"]) == {"migrant", "the"}
