"""Tests for bootstrap percentile CIs on keyness G²."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.keyness.bootstrap import bootstrap_g2_ci


def _strong_signal_corpora() -> tuple[pcd.Corpus, pcd.Corpus]:
    """Two corpora where 'distinctive' is strong-A and 'shared' is null."""
    a = pd.DataFrame({"text": ["distinctive shared shared"] * 30})
    b = pd.DataFrame({"text": ["mundane shared shared"] * 30})
    return pcd.from_dataframe(a, text_col="text"), pcd.from_dataframe(b, text_col="text")


def _balanced_corpora() -> tuple[pcd.Corpus, pcd.Corpus]:
    """Two corpora drawn from the same vocabulary in identical proportions."""
    rng = np.random.default_rng(0)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon"]
    a_docs = [" ".join(rng.choice(vocab, size=12)) for _ in range(40)]
    b_docs = [" ".join(rng.choice(vocab, size=12)) for _ in range(40)]
    return (
        pcd.from_dataframe(pd.DataFrame({"text": a_docs}), text_col="text"),
        pcd.from_dataframe(pd.DataFrame({"text": b_docs}), text_col="text"),
    )


def test_returns_indexed_frame_with_ci_columns() -> None:
    a, b = _strong_signal_corpora()
    out = bootstrap_g2_ci(a, b, n_boot=99, seed=0)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["g2_ci_lower", "g2_ci_upper"]
    assert set(out.index) == {"distinctive", "shared", "mundane"}


def test_lower_bound_never_exceeds_upper() -> None:
    """For every term: g2_ci_lower <= g2_ci_upper."""
    a, b = _balanced_corpora()
    out = bootstrap_g2_ci(a, b, n_boot=199, seed=42)
    assert (out["g2_ci_lower"] <= out["g2_ci_upper"]).all()


def test_point_estimate_typically_lies_inside_ci() -> None:
    """The signed G² point estimate should usually lie within its own
    percentile bootstrap CI. With strong, non-pathological data and a
    reasonable n_boot, this holds for every term.
    """
    a, b = _strong_signal_corpora()
    # Pull the signed G² point estimate via the public API to avoid
    # re-implementing the corpus-totals plumbing here.
    keyness = pcd.compare(a, b).keyness(min_count=1)
    point = keyness.table.set_index("term")["g2"]
    out = bootstrap_g2_ci(a, b, n_boot=399, seed=7)
    aligned = point.reindex(out.index)
    # Allow tiny floating-point slack at the boundary.
    eps = 1e-9
    assert ((aligned >= out["g2_ci_lower"] - eps)
            & (aligned <= out["g2_ci_upper"] + eps)).all()


def test_seed_makes_results_reproducible() -> None:
    a, b = _strong_signal_corpora()
    out1 = bootstrap_g2_ci(a, b, n_boot=99, seed=12345)
    out2 = bootstrap_g2_ci(a, b, n_boot=99, seed=12345)
    pd.testing.assert_frame_equal(out1, out2)


def test_different_seeds_diverge() -> None:
    a, b = _balanced_corpora()
    out1 = bootstrap_g2_ci(a, b, n_boot=99, seed=1)
    out2 = bootstrap_g2_ci(a, b, n_boot=99, seed=2)
    # Frames must not be byte-identical when the test is informative.
    assert not out1.equals(out2)


def test_terms_filter_restricts_output_index() -> None:
    a, b = _strong_signal_corpora()
    out = bootstrap_g2_ci(a, b, terms=["distinctive"], n_boot=49, seed=0)
    assert list(out.index) == ["distinctive"]


def test_balanced_corpora_cis_typically_straddle_zero() -> None:
    """Under approximate H₀ (matched vocabularies), most bootstrap CIs
    should include zero — a CI that excludes 0 corresponds to a
    statistically distinguishable effect, and we wouldn't expect many.
    """
    a, b = _balanced_corpora()
    out = bootstrap_g2_ci(a, b, n_boot=399, seed=11)
    straddles_zero = (out["g2_ci_lower"] <= 0.0) & (out["g2_ci_upper"] >= 0.0)
    # At least half of the terms should have CIs that straddle zero.
    assert straddles_zero.sum() >= len(out) // 2


def test_strong_signals_yield_directional_cis() -> None:
    """For a term that's overwhelmingly on one side (e.g. 'distinctive'
    appears only in A), the bootstrap CI should fall entirely on the
    positive side of zero.
    """
    a, b = _strong_signal_corpora()
    out = bootstrap_g2_ci(a, b, n_boot=399, seed=0)
    # 'distinctive' is exclusive to A → CI strictly above zero.
    assert out.loc["distinctive", "g2_ci_lower"] > 0.0
    # 'mundane' is exclusive to B → CI strictly below zero.
    assert out.loc["mundane", "g2_ci_upper"] < 0.0


def test_dunning_formula_runs_and_returns_finite_cis() -> None:
    """formula='dunning' should produce a finite CI on every term."""
    a, b = _strong_signal_corpora()
    out = bootstrap_g2_ci(a, b, formula="dunning", n_boot=99, seed=0)
    assert np.isfinite(out["g2_ci_lower"]).all()
    assert np.isfinite(out["g2_ci_upper"]).all()
    assert (out["g2_ci_lower"] <= out["g2_ci_upper"]).all()


def test_n_boot_zero_raises() -> None:
    a, b = _strong_signal_corpora()
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        bootstrap_g2_ci(a, b, n_boot=0)


def test_invalid_ci_level_raises() -> None:
    a, b = _strong_signal_corpora()
    with pytest.raises(ValueError, match="ci_level must be in"):
        bootstrap_g2_ci(a, b, n_boot=49, ci_level=0.0)
    with pytest.raises(ValueError, match="ci_level must be in"):
        bootstrap_g2_ci(a, b, n_boot=49, ci_level=1.0)
    with pytest.raises(ValueError, match="ci_level must be in"):
        bootstrap_g2_ci(a, b, n_boot=49, ci_level=1.5)


def test_invalid_formula_raises() -> None:
    a, b = _strong_signal_corpora()
    with pytest.raises(ValueError, match="formula must be 'rayson' or 'dunning'"):
        bootstrap_g2_ci(a, b, formula="bogus", n_boot=49)  # type: ignore[arg-type]


def test_empty_corpus_side_raises() -> None:
    """A side with zero documents must fail loudly."""
    a = pcd.from_dataframe(pd.DataFrame({"text": ["one two three"]}), text_col="text")
    empty = pd.DataFrame({"text": []})
    empty["text"] = empty["text"].astype(str)
    b = pcd.from_dataframe(empty, text_col="text")
    with pytest.raises(ValueError, match="at least one document on each side"):
        bootstrap_g2_ci(a, b, n_boot=10)


def test_ci_level_widens_interval() -> None:
    """A 99 % CI must be at least as wide as a 90 % CI on the same data."""
    a, b = _balanced_corpora()
    narrow = bootstrap_g2_ci(a, b, n_boot=499, ci_level=0.90, seed=3)
    wide = bootstrap_g2_ci(a, b, n_boot=499, ci_level=0.99, seed=3)
    narrow_w = narrow["g2_ci_upper"] - narrow["g2_ci_lower"]
    wide_w = wide["g2_ci_upper"] - wide["g2_ci_lower"]
    # Allow ties at zero-width terms; strictly: wide >= narrow everywhere.
    assert (wide_w >= narrow_w - 1e-9).all()


# ------------------- end-to-end wiring through compare.keyness -------------------


def test_wires_into_compare_keyness_ci_columns() -> None:
    """compare(a, b).keyness(ci='bootstrap') populates the CI columns."""
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=99, bootstrap_seed=7
    )
    assert "g2_ci_lower" in result.table.columns
    assert "g2_ci_upper" in result.table.columns
    # Every CI row has lower <= upper.
    assert (result.table["g2_ci_lower"] <= result.table["g2_ci_upper"]).all()


def test_keyness_default_omits_ci_columns() -> None:
    """Without ci='bootstrap', the result table stays column-clean."""
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(min_count=1)
    assert "g2_ci_lower" not in result.table.columns
    assert "g2_ci_upper" not in result.table.columns


def test_invalid_ci_kwarg_raises_via_compare_keyness() -> None:
    a, b = _strong_signal_corpora()
    with pytest.raises(ValueError, match="ci must be 'none' or 'bootstrap'"):
        pcd.compare(a, b).keyness(min_count=1, ci="bogus")  # type: ignore[arg-type]


def test_bootstrap_params_recorded_in_result() -> None:
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=99, ci_level=0.9, bootstrap_seed=3
    )
    assert result.params["ci"] == "bootstrap"
    assert result.params["n_boot"] == 99
    assert result.params["ci_level"] == 0.9
    assert result.params["bootstrap_seed"] == 3


def test_bootstrap_default_params_are_none_when_ci_off() -> None:
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(min_count=1)
    assert result.params["ci"] == "none"
    assert result.params["n_boot"] is None
    assert result.params["ci_level"] is None
    assert result.params["bootstrap_seed"] is None


def test_simultaneous_ci_returns_both_column_pairs() -> None:
    """``simultaneous_ci=True`` returns per-term AND simultaneous columns.

    The API contract (0.1.0a26 onwards) is:
      * ``simultaneous_ci=False``: returns ``g2_ci_lower`` and
        ``g2_ci_upper`` only (the per-term percentile CI).
      * ``simultaneous_ci=True``: returns the per-term columns
        unchanged AND adds ``g2_ci_lower_simultaneous`` /
        ``g2_ci_upper_simultaneous`` (the Westfall-Young
        studentized-max simultaneous CI).

    This way a single call gives both inferential perspectives so
    the user can report per-term CIs for any pre-specified term and
    simultaneous CIs for the top-ranked rows of a sorted table.
    """
    a, b = _strong_signal_corpora()
    result_per_term = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=199, bootstrap_seed=0,
        simultaneous_ci=False,
    )
    result_simul = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=199, bootstrap_seed=0,
        simultaneous_ci=True,
    )
    # simultaneous_ci=False: only per-term columns present
    assert "g2_ci_lower" in result_per_term.table.columns
    assert "g2_ci_upper" in result_per_term.table.columns
    assert "g2_ci_lower_simultaneous" not in result_per_term.table.columns
    assert "g2_ci_upper_simultaneous" not in result_per_term.table.columns
    # simultaneous_ci=True: both pairs present
    assert "g2_ci_lower" in result_simul.table.columns
    assert "g2_ci_upper" in result_simul.table.columns
    assert "g2_ci_lower_simultaneous" in result_simul.table.columns
    assert "g2_ci_upper_simultaneous" in result_simul.table.columns


def test_simultaneous_ci_widens_per_term_ci() -> None:
    """Simultaneous CIs are at least as wide as per-term CIs.

    Studentized-max CIs have family-wise coverage across the whole
    vocabulary, so their width is bounded below by the per-term
    percentile CI width. Both columns are returned from a single
    ``simultaneous_ci=True`` call (0.1.0a26 contract); width
    comparison reads them off the same table.
    """
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=199, bootstrap_seed=0,
        simultaneous_ci=True,
    )
    per_term_width = (
        result.table["g2_ci_upper"] - result.table["g2_ci_lower"]
    )
    simul_width = (
        result.table["g2_ci_upper_simultaneous"]
        - result.table["g2_ci_lower_simultaneous"]
    )
    # Mean simultaneous width should be strictly greater than per-term
    # (max-z quantile is > per-term 97.5 percentile by construction
    # for any non-trivial vocab).
    assert simul_width.mean() > per_term_width.mean()


def test_simultaneous_ci_recorded_in_params() -> None:
    a, b = _strong_signal_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=99, bootstrap_seed=0,
        simultaneous_ci=True,
    )
    assert result.params["simultaneous_ci"] is True

    result_default = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=99, bootstrap_seed=0,
    )
    assert result_default.params["simultaneous_ci"] is False


def _clustered_corpora() -> tuple[pcd.Corpus, pcd.Corpus]:
    """Two corpora where documents are nested in a few speakers, and
    each speaker has a *distinctive* vocabulary.

    Because each term is concentrated in one speaker, cluster-level
    resampling (a speaker is picked 0-N times) swings the marginal
    counts far more than IID document resampling (which keeps the
    term's document share roughly stable). So the cluster-robust CI
    must be wider than the IID CI — the whole point of the option.
    """
    import pandas as pd

    rows_a, rows_b = [], []
    # Side A: 4 speakers, each repeating a *distinct* enforcement line 6x.
    a_lines = {
        "A0": "illegal illegal illegal",
        "A1": "borders borders criminal",
        "A2": "criminal enforcement enforcement",
        "A3": "removal removal deportation",
    }
    for sp, line in a_lines.items():
        for _ in range(6):
            rows_a.append({"text": line, "speaker": sp})
    b_lines = {
        "B0": "rights rights rights",
        "B1": "protection protection refugees",
        "B2": "refugees welfare welfare",
        "B3": "support support sanctuary",
    }
    for sp, line in b_lines.items():
        for _ in range(6):
            rows_b.append({"text": line, "speaker": sp})
    a = pcd.from_dataframe(pd.DataFrame(rows_a), text_col="text", meta_cols=("speaker",))
    b = pcd.from_dataframe(pd.DataFrame(rows_b), text_col="text", meta_cols=("speaker",))
    return a, b


def test_cluster_col_widens_ci_vs_iid() -> None:
    """Cluster-robust CIs are wider than IID CIs when documents are
    nested in speakers (effective n is the speaker count)."""
    a, b = _clustered_corpora()
    iid = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=199, bootstrap_seed=0,
    )
    clustered = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=199, bootstrap_seed=0,
        cluster_col="speaker",
    )
    iid_w = (iid.table["g2_ci_upper"] - iid.table["g2_ci_lower"]).mean()
    clu_w = (clustered.table["g2_ci_upper"] - clustered.table["g2_ci_lower"]).mean()
    assert clu_w > iid_w


def test_cluster_col_recorded_in_params() -> None:
    a, b = _clustered_corpora()
    result = pcd.compare(a, b).keyness(
        min_count=1, ci="bootstrap", n_boot=99, bootstrap_seed=0,
        cluster_col="speaker",
    )
    assert result.params["cluster_col"] == "speaker"


def test_cluster_col_missing_column_raises() -> None:
    a, b = _clustered_corpora()
    with pytest.raises(ValueError, match="cluster_col='nonexistent' not found"):
        pcd.compare(a, b).keyness(
            min_count=1, ci="bootstrap", n_boot=99, bootstrap_seed=0,
            cluster_col="nonexistent",
        )
