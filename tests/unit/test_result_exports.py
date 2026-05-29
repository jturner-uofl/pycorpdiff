"""Tests for ``Result.to_html`` and ``Result.to_json`` across every Result type."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def two_outlet_corpus() -> pcd.Corpus:
    rows = [
        {"text": "the migrant worker arrived and settled", "outlet": "A", "date": "2020-01-15"},
        {"text": "the migrant family thrived in the community", "outlet": "A", "date": "2020-06-15"},
        {"text": "the migrant criminal threat grew worse", "outlet": "B", "date": "2022-01-15"},
        {"text": "the migrant invasion of gangs spread", "outlet": "B", "date": "2022-06-15"},
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("outlet", "date"))


def test_keyness_result_to_html(
    two_outlet_corpus: pcd.Corpus, tmp_path: Path
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=1)
    out_path = tmp_path / "keyness.html"
    html = result.to_html(out_path)
    assert "<table" in html
    assert out_path.exists()
    assert "<table" in out_path.read_text()


def test_keyness_result_to_json(
    two_outlet_corpus: pcd.Corpus, tmp_path: Path
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=1)
    out_path = tmp_path / "keyness.json"
    json_str = result.to_json(out_path)
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert "term" in parsed[0]
    written = json.loads(out_path.read_text())
    assert written == parsed


def test_to_html_without_path_returns_string(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=1)
    html = result.to_html()
    assert isinstance(html, str)
    assert "<table" in html


def test_collocation_shift_result_exports(
    two_outlet_corpus: pcd.Corpus, tmp_path: Path
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).collocation_shift("migrant", min_count=1)
    html_path = tmp_path / "shift.html"
    json_path = tmp_path / "shift.json"
    result.to_html(html_path)
    result.to_json(json_path)
    assert html_path.exists() and "<table" in html_path.read_text()
    assert json_path.exists() and isinstance(
        json.loads(json_path.read_text()), list
    )


def test_temporal_trajectory_exports(
    two_outlet_corpus: pcd.Corpus, tmp_path: Path
) -> None:
    trajectory = pcd.track(two_outlet_corpus, "migrant").over_time(
        freq="Y", time_col="date"
    )
    html = trajectory.to_html()
    json_str = trajectory.to_json()
    assert "<table" in html
    rows = json.loads(json_str)
    assert len(rows) >= 1
    assert "period" in rows[0]


def test_concordance_result_exports(two_outlet_corpus: pcd.Corpus) -> None:
    result = pcd.compare(
        two_outlet_corpus.slice(outlet="A"), two_outlet_corpus.slice(outlet="B")
    ).concordance("migrant", n=2)
    html = result.to_html()
    json_str = result.to_json()
    assert "<table" in html
    assert isinstance(json.loads(json_str), list)


def test_semantic_shift_result_exports(two_outlet_corpus: pcd.Corpus) -> None:
    result = pcd.compare(
        two_outlet_corpus.slice(outlet="A"), two_outlet_corpus.slice(outlet="B")
    ).semantic_shift("migrant", embedder=pcd.HashEmbedder(dim=16))
    html = result.to_html()
    rows = json.loads(result.to_json())
    assert "<table" in html
    assert "cosine_distance" in rows[0]
