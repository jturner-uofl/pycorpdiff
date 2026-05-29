"""Tests for permutation-based *p*-values on keyness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.keyness.permutation import permutation_pvalues


def _strong_signal_corpora() -> tuple[pcd.Corpus, pcd.Corpus]:
    """Two corpora where 'distinctive' is strong-A and 'shared' is null."""
    a = pd.DataFrame({"text": ["distinctive shared shared"] * 30})
    b = pd.DataFrame({"text": ["mundane shared shared"] * 30})
    return pcd.from_dataframe(a, text_col="text"), pcd.from_dataframe(b, text_col="text")


def _null_corpora() -> tuple[pcd.Corpus, pcd.Corpus]:
    """Two corpora drawn from the same vocabulary in identical proportions."""
    rng = np.random.default_rng(0)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
    a_docs = [" ".join(rng.choice(vocab, size=10)) for _ in range(30)]
    b_docs = [" ".join(rng.choice(vocab, size=10)) for _ in range(30)]
    return (
        pcd.from_dataframe(pd.DataFrame({"text": a_docs}), text_col="text"),
        pcd.from_dataframe(pd.DataFrame({"text": b_docs}), text_col="text"),
    )


def test_returns_named_series_indexed_by_term() -> None:
    a, b = _strong_signal_corpora()
    p = permutation_pvalues(a, b, n_permutations=99, seed=0)
    assert isinstance(p, pd.Series)
    assert p.name == "p_permutation"
    assert set(p.index) == {"distinctive", "shared", "mundane"}


def test_p_values_in_bounded_range() -> None:
    """Every permutation p must be in [1/(B+1), 1] by Phipson–Smyth."""
    a, b = _null_corpora()
    n_perms = 199
    p = permutation_pvalues(a, b, n_permutations=n_perms, seed=42)
    assert (p >= 1.0 / (n_perms + 1)).all()
    assert (p <= 1.0).all()


def test_strong_signals_yield_smallest_possible_p() -> None:
    """A term that's deterministic for one side hits the floor 1/(B+1)."""
    a, b = _strong_signal_corpora()
    n_perms = 199
    p = permutation_pvalues(a, b, n_permutations=n_perms, seed=42)
    expected_floor = 1.0 / (n_perms + 1)
    assert p["distinctive"] == pytest.approx(expected_floor)
    assert p["mundane"] == pytest.approx(expected_floor)
    # 'shared' is balanced; should be near 1.
    assert p["shared"] > 0.5


def test_null_corpora_distribute_around_uniform() -> None:
    """Under H₀ the permutation *p* should look uniformly distributed —
    in particular, the *mean* should be close to 0.5 and most terms
    shouldn't cross the 0.05 threshold by chance.
    """
    a, b = _null_corpora()
    p = permutation_pvalues(a, b, n_permutations=499, seed=11)
    # mean of a Uniform(0,1) sample is 0.5; with 5 terms this is loose
    # but a stark deviation (e.g. mean < 0.2) would reveal a bug.
    assert 0.2 < p.mean() < 0.8
    # No more than ~half of the terms should be "significant" by chance.
    assert (p < 0.05).sum() <= len(p) // 2


def test_seed_makes_results_reproducible() -> None:
    a, b = _strong_signal_corpora()
    p1 = permutation_pvalues(a, b, n_permutations=99, seed=12345)
    p2 = permutation_pvalues(a, b, n_permutations=99, seed=12345)
    pd.testing.assert_series_equal(p1, p2)


def test_different_seeds_can_diverge() -> None:
    """Smoke check: different seeds shouldn't be byte-identical when the
    test is informative (the null case)."""
    a, b = _null_corpora()
    p1 = permutation_pvalues(a, b, n_permutations=99, seed=1)
    p2 = permutation_pvalues(a, b, n_permutations=99, seed=2)
    assert not p1.equals(p2)


def test_terms_filter_restricts_output_index() -> None:
    a, b = _strong_signal_corpora()
    p = permutation_pvalues(
        a, b, terms=["distinctive"], n_permutations=99, seed=0
    )
    assert list(p.index) == ["distinctive"]


def test_n_permutations_zero_raises() -> None:
    a, b = _strong_signal_corpora()
    with pytest.raises(ValueError, match="n_permutations must be >= 1"):
        permutation_pvalues(a, b, n_permutations=0)


def test_empty_corpus_side_raises() -> None:
    """Single-doc on one side, none on the other, must fail loudly."""
    a = pcd.from_dataframe(pd.DataFrame({"text": ["one"]}), text_col="text")
    empty = pd.DataFrame({"text": []})
    empty["text"] = empty["text"].astype(str)
    b = pcd.from_dataframe(empty, text_col="text")
    with pytest.raises(ValueError, match="at least one document on each side"):
        permutation_pvalues(a, b, n_permutations=10)


def test_wires_into_compare_keyness() -> None:
    """compare(a, b).keyness(permutation_n=...) should populate the
    p_permutation column on the result table."""
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, permutation_n=99, permutation_seed=7
    )
    assert "p_permutation" in result.table.columns
    # The distinctive terms should both have p_permutation == 1/100.
    dist = result.table[result.table["term"] == "distinctive"]
    assert dist["p_permutation"].iloc[0] == pytest.approx(1 / 100)


def test_permutation_n_zero_omits_column() -> None:
    """The default (no permutation) keeps the result table column-clean."""
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(min_count=1)
    assert "p_permutation" not in result.table.columns


def test_permutation_n_recorded_in_params() -> None:
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, permutation_n=99, permutation_seed=7
    )
    assert result.params["permutation_n"] == 99
    assert result.params["permutation_seed"] == 7


def test_permutation_pvalues_match_asymptotic_ordering_when_signal_is_strong() -> None:
    """When the signal is strong, permutation and asymptotic *p*-values
    must agree on *which* terms are significant, even though the
    exact numbers differ.
    """
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, permutation_n=199, permutation_seed=0
    )
    # Every term significant by chi² should also be sig by permutation.
    significant_chi2 = set(result.table[result.table["p_value"] < 0.01]["term"])
    significant_perm = set(result.table[result.table["p_permutation"] < 0.05]["term"])
    assert significant_chi2.issubset(significant_perm)
