"""End-to-end tests for ``compare(a, b).keyness()``."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def two_outlet_corpus() -> pcd.Corpus:
    """A 10-document, 2-outlet fixture engineered to produce clear keyness.

    Outlet A overuses 'migrant', outlet B overuses 'asylum'. Both share a
    common-vocab carrier ('the', 'said', etc.). The 'and' tokens give us
    a high-frequency null term we expect near zero keyness on.
    """
    a_docs = [
        "the migrant said the migrant arrived and the migrant left",
        "the migrant and the migrant and the migrant",
        "the migrant said the migrant and the migrant",
        "the migrant arrived and the migrant said",
        "the migrant and the migrant left",
    ]
    b_docs = [
        "the asylum seeker said the asylum centre and the asylum",
        "the asylum and the asylum and the asylum",
        "the asylum said and the asylum",
        "the asylum centre and the asylum arrived",
        "the asylum seeker and the asylum",
    ]
    rows = [{"text": d, "outlet": "A"} for d in a_docs] + [
        {"text": d, "outlet": "B"} for d in b_docs
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("outlet",))


def test_keyness_returns_keynessresult(two_outlet_corpus: pcd.Corpus) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=2)
    assert isinstance(result, pcd.KeynessResult)
    assert result.method == "log_likelihood"
    assert "g2" in result.table.columns
    assert "log_ratio" in result.table.columns
    assert "percent_diff" in result.table.columns
    assert "bayes_factor" in result.table.columns
    assert "p_value" in result.table.columns
    assert "p_adjusted" in result.table.columns


def test_keyness_identifies_overused_terms(two_outlet_corpus: pcd.Corpus) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=2)
    df = result.table.set_index("term")
    # 'migrant' should be the strongest positive-direction term (overused in A).
    assert df.loc["migrant", "g2"] > 0
    # 'asylum' should be the strongest negative-direction term (overused in B).
    assert df.loc["asylum", "g2"] < 0
    # And both should rank near the top by |g2|.
    top_two = result.table.head(2)["term"].tolist()
    assert set(top_two) == {"migrant", "asylum"}


def test_keyness_min_count_filter_drops_rare_terms(two_outlet_corpus: pcd.Corpus) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result_low = pcd.compare(a, b).keyness(min_count=1)
    result_high = pcd.compare(a, b).keyness(min_count=10)
    assert len(result_low.table) > len(result_high.table)


def test_keyness_dispersion_flag_added_when_requested(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=2, dispersion=True)
    for col in ("dispersion_a", "dispersion_b", "dispersion_flag"):
        assert col in result.table.columns


def test_keyness_corpus_swap_negates_signed_g2(two_outlet_corpus: pcd.Corpus) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    forward = pcd.compare(a, b).keyness(min_count=2).table.set_index("term")
    reverse = pcd.compare(b, a).keyness(min_count=2).table.set_index("term")
    for term in forward.index:
        assert math.isclose(
            forward.loc[term, "g2"], -reverse.loc[term, "g2"], rel_tol=1e-9
        )


def test_keyness_method_dispatch_chooses_sort_column(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    by_lr = pcd.compare(a, b).keyness(min_count=2, method="log_ratio")
    # Sorted by |log_ratio| descending.
    diffs = by_lr.table["log_ratio"].abs().diff().dropna()
    assert (diffs <= 1e-12).all()


def test_keyness_method_without_effect_size_raises(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    with pytest.raises(ValueError, match="requires effect_size=True"):
        pcd.compare(a, b).keyness(method="log_ratio", effect_size=False)


def test_keyness_empty_corpus_raises() -> None:
    df_a = pd.DataFrame({"text": ["", ""], "outlet": ["A", "A"]})
    df_b = pd.DataFrame({"text": ["the cat sat"], "outlet": ["B"]})
    corpus = pcd.from_dataframe(
        pd.concat([df_a, df_b], ignore_index=True),
        text_col="text",
        meta_cols=("outlet",),
    )
    a = corpus.slice(outlet="A")
    b = corpus.slice(outlet="B")
    with pytest.raises(ValueError, match="at least one token"):
        pcd.compare(a, b).keyness()


def test_keyness_bonferroni_alternative(two_outlet_corpus: pcd.Corpus) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    bh = pcd.compare(a, b).keyness(min_count=2, multiple_comparisons="bh")
    bonf = pcd.compare(a, b).keyness(min_count=2, multiple_comparisons="bonferroni")
    # Bonferroni is more conservative than BH on the same vector.
    assert (bonf.table["p_adjusted"].to_numpy() >= bh.table["p_adjusted"].to_numpy()).all()


def test_keyness_none_skips_adjustment(two_outlet_corpus: pcd.Corpus) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).keyness(min_count=2, multiple_comparisons="none")
    assert "p_adjusted" not in result.table.columns
