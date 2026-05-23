"""Tests for N-way keyness via :func:`pycorpdiff.keyness_multi`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.keyness.multicorpus import keyness_multi


def _corpus(text: list[str]) -> pcd.Corpus:
    return pcd.from_dataframe(pd.DataFrame({"text": text}), text_col="text")


def test_returns_sorted_dataframe_with_expected_columns() -> None:
    a = _corpus(["foo bar"] * 5)
    b = _corpus(["bar baz"] * 5)
    c = _corpus(["baz qux"] * 5)
    table = keyness_multi([a, b, c], labels=["a", "b", "c"], min_count=1)
    assert set(["count_a", "count_b", "count_c", "g2", "p_value", "p_adjusted"]).issubset(
        table.columns
    )
    # Sorted by G² descending.
    assert (np.diff(table["g2"].to_numpy()) <= 1e-12).all()


def test_n_equals_2_agrees_with_pairwise_keyness() -> None:
    """For N=2 the multi-corpus G² must match the absolute value of the
    signed G² emitted by compare(a, b).keyness() — same math, same row.
    """
    a = _corpus(["the cat sat"] * 12 + ["the dog ran"] * 8)
    b = _corpus(["the cat ran"] * 10 + ["the dog sat"] * 10)
    two_way = pcd.compare(a, b).keyness(min_count=1).to_df()
    multi = keyness_multi([a, b], labels=["a", "b"], min_count=1)
    for term in multi.index:
        m_g2 = float(multi.loc[term, "g2"])
        pair_g2 = float(two_way.loc[two_way["term"] == term, "g2"].iloc[0])
        assert m_g2 == pytest.approx(abs(pair_g2), abs=1e-9, rel=1e-9), (
            f"disagreement on {term!r}: multi={m_g2}, |2-way|={abs(pair_g2)}"
        )


def test_terms_unique_to_one_corpus_rank_above_shared_terms() -> None:
    """Terms unique to one corpus must rank above terms shared across corpora."""
    a = _corpus(["the cat sat"] * 10)
    b = _corpus(["the cat ran"] * 10)
    c = _corpus(["the bird flew"] * 5)
    table = keyness_multi([a, b, c], labels=["a", "b", "c"], min_count=1)
    # 'bird' and 'flew' are unique to corpus c; 'sat' and 'ran' are each
    # unique to one of (a, b). All four must outrank 'cat' (shared by a+b)
    # and 'the' (shared by all three).
    unique_to_one = {"bird", "flew", "sat", "ran"}
    g2_unique_min = table.loc[list(unique_to_one), "g2"].min()
    g2_cat = table.loc["cat", "g2"]
    g2_the = table.loc["the", "g2"]
    assert g2_unique_min > g2_cat
    assert g2_cat > g2_the
    # 'the' appears at equal rates in all three → G² ≈ 0.
    assert table.loc["the", "g2"] == pytest.approx(0.0, abs=1e-9)


def test_p_value_uses_df_n_minus_1() -> None:
    """Asymptotic p must come from chi²(df = N − 1)."""
    from scipy.stats import chi2

    a = _corpus(["a"] * 100 + ["b"] * 1)
    b = _corpus(["a"] * 1 + ["b"] * 100)
    c = _corpus(["a"] * 50 + ["b"] * 50)
    table = keyness_multi([a, b, c], labels=["a", "b", "c"], min_count=1)
    for term in table.index:
        g2 = float(table.loc[term, "g2"])
        p = float(table.loc[term, "p_value"])
        expected_p = chi2.sf(g2, df=2)
        assert p == pytest.approx(expected_p, rel=1e-9)


def test_min_count_drops_rare_terms() -> None:
    a = _corpus(["common common common rare"] * 5)
    b = _corpus(["common common common"] * 5)
    c = _corpus(["common common common"] * 5)
    table = keyness_multi([a, b, c], labels=["a", "b", "c"], min_count=10)
    assert "rare" not in table.index
    assert "common" in table.index


def test_stop_words_filter_post_min_count() -> None:
    a = _corpus(["the cat"] * 5)
    b = _corpus(["the dog"] * 5)
    c = _corpus(["the bird"] * 5)
    table = keyness_multi(
        [a, b, c], labels=["a", "b", "c"], min_count=1, stop_words={"the"}
    )
    assert "the" not in table.index
    assert {"cat", "dog", "bird"}.issubset(set(table.index))


def test_default_labels_are_indexed() -> None:
    a = _corpus(["foo"] * 5)
    b = _corpus(["bar"] * 5)
    table = keyness_multi([a, b], min_count=1)
    assert "count_corpus_0" in table.columns
    assert "count_corpus_1" in table.columns


def test_label_count_mismatch_raises() -> None:
    a = _corpus(["a"] * 5)
    b = _corpus(["b"] * 5)
    with pytest.raises(ValueError, match="labels must have one entry per corpus"):
        keyness_multi([a, b], labels=["just_one"])


def test_single_corpus_raises() -> None:
    a = _corpus(["a"] * 5)
    with pytest.raises(ValueError, match="need at least 2 corpora"):
        keyness_multi([a])


def test_empty_corpus_in_list_raises() -> None:
    a = _corpus(["a"] * 5)
    empty_df = pd.DataFrame({"text": []})
    empty_df["text"] = empty_df["text"].astype(str)
    empty = pcd.from_dataframe(empty_df, text_col="text")
    with pytest.raises(ValueError, match="empty corpora"):
        keyness_multi([a, empty])


def test_multiple_comparisons_bonferroni() -> None:
    a = _corpus(["a"] * 100 + ["b"] * 1)
    b = _corpus(["a"] * 1 + ["b"] * 100)
    c = _corpus(["a"] * 50 + ["b"] * 50)
    bh = keyness_multi(
        [a, b, c], labels=["a", "b", "c"], min_count=1, multiple_comparisons="bh"
    )
    bonf = keyness_multi(
        [a, b, c], labels=["a", "b", "c"], min_count=1, multiple_comparisons="bonferroni"
    )
    # Bonferroni adjusts strictly upward of (or equal to) BH.
    assert (bonf["p_adjusted"] >= bh["p_adjusted"] - 1e-12).all()


def test_multiple_comparisons_none_omits_column() -> None:
    a = _corpus(["a"] * 5)
    b = _corpus(["b"] * 5)
    table = keyness_multi([a, b], min_count=1, multiple_comparisons="none")
    assert "p_adjusted" not in table.columns


def test_multiple_comparisons_invalid_raises() -> None:
    a = _corpus(["a"] * 5)
    b = _corpus(["b"] * 5)
    with pytest.raises(ValueError, match="multiple_comparisons must be"):
        keyness_multi([a, b], min_count=1, multiple_comparisons="bogus")


def test_empty_result_when_no_terms_survive_min_count() -> None:
    """A min_count higher than any term's total returns a valid empty frame."""
    a = _corpus(["alpha"])
    b = _corpus(["beta"])
    table = keyness_multi([a, b], labels=["a", "b"], min_count=100)
    assert len(table) == 0
    assert "g2" in table.columns
    assert "count_a" in table.columns


def test_works_with_corpus_slices() -> None:
    """N-way keyness should accept CorpusSlice inputs (not just Corpus)."""
    df = pd.DataFrame(
        {
            "text": ["a"] * 5 + ["b"] * 5 + ["c"] * 5,
            "group": ["X"] * 5 + ["Y"] * 5 + ["Z"] * 5,
        }
    )
    corpus = pcd.from_dataframe(df, text_col="text", meta_cols=("group",))
    slices = [corpus.slice(group=g) for g in ("X", "Y", "Z")]
    table = keyness_multi(slices, labels=["X", "Y", "Z"], min_count=1)
    assert set(table.index) == {"a", "b", "c"}


def test_exported_at_package_root() -> None:
    """`pcd.keyness_multi` should be importable directly."""
    assert pcd.keyness_multi is keyness_multi
