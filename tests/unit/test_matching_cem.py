"""Tests for Coarsened Exact Matching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.matching import match
from pycorpdiff.matching.cem import _l1_imbalance


def _corpus_with_metadata(n: int = 30, *, seed: int = 0) -> pcd.Corpus:
    """Build a small corpus with year + party metadata for matching."""
    rng = np.random.default_rng(seed)
    years = rng.integers(2010, 2025, size=n)
    parties = rng.choice(["A", "B", "C"], size=n)
    df = pd.DataFrame(
        {
            "text": [f"document {i} alpha beta" for i in range(n)],
            "year": years,
            "party": parties,
        }
    )
    return pcd.from_dataframe(df, text_col="text", meta_cols=("year", "party"))


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


def test_match_returns_balanced_slices() -> None:
    a = _corpus_with_metadata(40, seed=1)
    b = _corpus_with_metadata(40, seed=2)
    m = match(a, b, on=["year", "party"], seed=0)
    # Equal counts on both sides under subsample=True (k-to-k matching).
    assert m.n_a_post == m.n_b_post
    # Some matches must occur with this much overlap.
    assert m.n_a_post > 0


def test_match_reduces_l1_imbalance() -> None:
    """Post-match L1 imbalance must be <= pre-match for every covariate."""
    a = _corpus_with_metadata(40, seed=3)
    b = _corpus_with_metadata(40, seed=4)
    m = match(a, b, on=["year", "party"], seed=0)
    assert (m.imbalance["l1_post"] <= m.imbalance["l1_pre"] + 1e-9).all()


def test_match_is_reproducible_under_seed() -> None:
    a = _corpus_with_metadata(40, seed=5)
    b = _corpus_with_metadata(40, seed=6)
    m1 = match(a, b, on=["year", "party"], seed=42)
    m2 = match(a, b, on=["year", "party"], seed=42)
    assert m1.n_a_post == m2.n_a_post
    assert (m1.a_matched.mask == m2.a_matched.mask).all()
    assert (m1.b_matched.mask == m2.b_matched.mask).all()


def test_different_seeds_diverge() -> None:
    a = _corpus_with_metadata(60, seed=7)
    b = _corpus_with_metadata(60, seed=8)
    m1 = match(a, b, on=["year", "party"], seed=1)
    m2 = match(a, b, on=["year", "party"], seed=2)
    # The two subsamples should not be byte-identical when the
    # stratum is over-populated on at least one side.
    if (m1.strata["n_a_pre"] > m1.strata["n_b_pre"]).any():
        assert not (m1.a_matched.mask == m2.a_matched.mask).all()


def test_matched_slice_carries_full_metadata() -> None:
    """The matched slice should expose the same columns as the parent."""
    a = _corpus_with_metadata(30)
    b = _corpus_with_metadata(30, seed=10)
    m = match(a, b, on=["year"], seed=0)
    assert "year" in m.a_matched.docs.columns
    assert "party" in m.a_matched.docs.columns


def test_match_summary_string_mentions_covariates() -> None:
    a = _corpus_with_metadata(30)
    b = _corpus_with_metadata(30, seed=10)
    m = match(a, b, on=["year"], seed=0)
    s = m.summary()
    assert "CEM" in s
    assert "year" in s
    assert "L1" in s


# ----------------------------------------------------------------------
# Coarsening behaviour
# ----------------------------------------------------------------------


def test_explicit_cuts_override_default_quantile_binning() -> None:
    a = _corpus_with_metadata(30)
    b = _corpus_with_metadata(30, seed=10)
    m = match(a, b, on=["year"], cuts={"year": [2010, 2015, 2020, 2025]}, seed=0)
    # Strata index values reflect the explicit cut edges.
    strata_keys = list(m.strata.index)
    # At least one stratum's key contains one of the new bin labels.
    assert any("2015" in str(k) or "2020" in str(k) for k in strata_keys)


def test_categorical_covariate_used_as_is() -> None:
    a = _corpus_with_metadata(40, seed=20)
    b = _corpus_with_metadata(40, seed=21)
    m = match(a, b, on=["party"], seed=0)
    # Strata are the three party labels.
    keys = {str(k) for k in m.strata.index}
    assert {"A", "B", "C"}.issubset(keys)


# ----------------------------------------------------------------------
# Error conditions
# ----------------------------------------------------------------------


def test_match_empty_covariates_raises() -> None:
    a = _corpus_with_metadata(10)
    b = _corpus_with_metadata(10, seed=1)
    with pytest.raises(ValueError, match="at least one covariate"):
        match(a, b, on=[])


def test_match_missing_covariate_raises() -> None:
    a = _corpus_with_metadata(10)
    b = _corpus_with_metadata(10, seed=1)
    with pytest.raises(ValueError, match="missing from corpus a"):
        match(a, b, on=["nonexistent"])


def test_match_empty_side_raises() -> None:
    a = _corpus_with_metadata(10)
    empty_df = pd.DataFrame({"text": [], "year": [], "party": []})
    empty_df["text"] = empty_df["text"].astype(str)
    empty_df["year"] = empty_df["year"].astype("int64")
    empty_df["party"] = empty_df["party"].astype(str)
    b = pcd.from_dataframe(empty_df, text_col="text")
    with pytest.raises(ValueError, match="at least one document on each side"):
        match(a, b, on=["year"])


def test_match_no_overlapping_stratum_raises() -> None:
    """When every stratum is single-sided, match() must fail loudly."""
    df_a = pd.DataFrame(
        {"text": ["alpha"] * 5, "year": [2010, 2011, 2012, 2013, 2014]}
    )
    df_b = pd.DataFrame(
        {"text": ["beta"] * 5, "year": [2020, 2021, 2022, 2023, 2024]}
    )
    a = pcd.from_dataframe(df_a, text_col="text")
    b = pcd.from_dataframe(df_b, text_col="text")
    with pytest.raises(ValueError, match="no stratum contains documents from both sides"):
        match(a, b, on=["year"], cuts={"year": [2010, 2015, 2025]})


# ----------------------------------------------------------------------
# Slices as input
# ----------------------------------------------------------------------


def test_match_accepts_corpusslice_input() -> None:
    """Pre-sliced inputs (Comparison.compare(corpus.slice(...), ...))
    must work — the resulting masks live over the parent corpus."""
    corpus = pcd.load_hansard_sample()
    a = corpus.slice(topic="immigration")
    b = corpus.slice(topic="nhs")
    m = match(a, b, on=["year", "party"], seed=0)
    # The matched slices stay views of the same parent.
    assert m.a_matched.parent is corpus
    assert m.b_matched.parent is corpus
    assert m.n_a_post > 0
    assert m.n_b_post > 0


def test_match_then_compare_keyness_pipeline_runs() -> None:
    """End-to-end: match → compare → keyness gives a valid table."""
    corpus = pcd.load_hansard_sample()
    a = corpus.slice(topic="immigration")
    b = corpus.slice(topic="nhs")
    m = match(a, b, on=["year", "party"], seed=0)
    result = pcd.compare(m.a_matched, m.b_matched).keyness(min_count=1)
    assert "g2" in result.table.columns
    assert len(result.table) > 0


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def test_l1_imbalance_zero_for_identical_distributions() -> None:
    s = pd.Series(["a", "a", "b", "b", "c"])
    assert _l1_imbalance(s, s.copy()) == pytest.approx(0.0)


def test_l1_imbalance_one_for_disjoint_distributions() -> None:
    a = pd.Series(["x", "x", "x"])
    b = pd.Series(["y", "y", "y"])
    assert _l1_imbalance(a, b) == pytest.approx(1.0)
