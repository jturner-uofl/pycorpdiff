"""Tests for ``Comparison.concordance(target)``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def two_outlet_corpus() -> pcd.Corpus:
    rows = [
        {"text": "the migrant worker arrived and the family settled", "outlet": "A"},
        {"text": "the migrant family stayed", "outlet": "A"},
        {"text": "the migrant criminal threat grew", "outlet": "B"},
        {"text": "the migrant invasion grew worse", "outlet": "B"},
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("outlet",))


def test_comparison_concordance_returns_concordance_result(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=2, window=2)
    assert isinstance(result, pcd.ConcordanceResult)
    assert (result.table["keyword"] == "migrant").all()


def test_comparison_concordance_labels_each_side(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=5)
    assert set(result.table["corpus"]) == {"outlet='A'", "outlet='B'"}


def test_comparison_concordance_respects_n_per_side(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=1)
    by_corpus = result.table.groupby("corpus").size()
    assert by_corpus.loc["outlet='A'"] == 1
    assert by_corpus.loc["outlet='B'"] == 1


def test_comparison_concordance_missing_target_returns_empty(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("unicorn")
    assert len(result.table) == 0
