"""Tests for the lexical-diversity metrics and the lexical_diversity verb."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.lexical.diversity import (
    LexicalDiversityResult,
    LexicalDiversityTrajectory,
    hdd,
    hill_numbers,
    mattr,
    mtld,
    rarefaction,
    ttr,
)

# ----------------------------------------------------------------------
# Pure metrics
# ----------------------------------------------------------------------


def test_ttr_basic_arithmetic() -> None:
    assert ttr(["a", "b", "c"]) == pytest.approx(1.0)
    assert ttr(["a", "a", "a"]) == pytest.approx(1 / 3)
    assert ttr(["a", "b", "a", "b"]) == pytest.approx(0.5)


def test_ttr_empty_returns_nan() -> None:
    assert np.isnan(ttr([]))


def test_mattr_equals_ttr_for_short_text() -> None:
    """Text shorter than window falls back to plain TTR."""
    tokens = ["a", "b", "c"]
    assert mattr(tokens, window=100) == pytest.approx(ttr(tokens))


def test_mattr_length_robust_relative_to_ttr() -> None:
    """For a 1000-token text where TTR would crater, MATTR stays stable."""
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(50)]
    tokens = list(rng.choice(vocab, size=1000))
    plain_ttr = ttr(tokens)
    mattr_val = mattr(tokens, window=100)
    # MATTR should be much higher than TTR for a 1000-token text with
    # 50-word vocab — TTR floors near 0.05, MATTR around 0.4–0.5.
    assert mattr_val > plain_ttr + 0.2


def test_mattr_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        mattr(["a", "b", "c"], window=0)


def test_mtld_empty_returns_nan() -> None:
    assert np.isnan(mtld([]))


def test_mtld_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        mtld(["a", "b"], threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be in"):
        mtld(["a", "b"], threshold=1.0)


def test_mtld_high_diversity_yields_long_factors() -> None:
    """A high-diversity stream should yield a high MTLD (long factors
    before the running TTR drops to threshold)."""
    # Interleave a 100-word fresh-vocab run with sparse repetition so
    # the running TTR descends slowly through 0.72 — gives a high but
    # finite MTLD. (Pure all-unique input is degenerate per McCarthy
    # & Jarvis 2010 and yields NaN; that's the literature behaviour.)
    rng = np.random.default_rng(0)
    fresh = [f"u{i}" for i in range(200)]
    repeats = rng.choice(fresh[:30], size=80).tolist()
    tokens = fresh + repeats
    rng.shuffle(tokens)
    val = mtld(tokens, threshold=0.72)
    assert val > 50


def test_mtld_low_diversity_yields_short_factors() -> None:
    """A repetitive stream should give a short MTLD."""
    tokens = ["a", "b", "c", "a", "b", "c"] * 50
    val = mtld(tokens, threshold=0.72)
    # Running TTR crashes through 0.72 within the first ~10 tokens.
    assert val < 20


def test_hdd_empty_returns_nan() -> None:
    assert np.isnan(hdd([]))


def test_hdd_short_text_returns_unique_type_count() -> None:
    """For texts <= sample_size, HD-D collapses to the unique type count."""
    tokens = ["a", "b", "c", "a"]
    assert hdd(tokens, sample_size=42) == pytest.approx(3.0)


def test_hdd_bounded_by_sample_size_in_practice() -> None:
    """For a long, high-diversity text HD-D approaches sample_size."""
    rng = np.random.default_rng(1)
    tokens = list(rng.choice([f"w{i}" for i in range(500)], size=2000))
    val = hdd(tokens, sample_size=42)
    # With huge vocab and a 42-token sample, expected unique types
    # should be close to (but less than) 42.
    assert 30 < val <= 42


def test_hdd_invalid_sample_size_raises() -> None:
    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        hdd(["a", "b"], sample_size=0)


def test_hdd_increases_with_diversity() -> None:
    rng = np.random.default_rng(2)
    diverse = list(rng.choice([f"w{i}" for i in range(200)], size=500))
    repetitive = list(rng.choice([f"w{i}" for i in range(10)], size=500))
    assert hdd(diverse) > hdd(repetitive)


# ----------------------------------------------------------------------
# Public lexical_diversity — pooled mode
# ----------------------------------------------------------------------


def _corpus_of(texts: list[str], dates: list[str] | None = None) -> pcd.Corpus:
    df = pd.DataFrame({"text": texts})
    if dates is not None:
        df["date"] = pd.to_datetime(dates)
    return pcd.from_dataframe(df, text_col="text")


def test_lexical_diversity_returns_result_dataclass() -> None:
    corpus = _corpus_of(["alpha beta gamma delta epsilon"] * 5)
    result = pcd.lexical_diversity(corpus)
    assert isinstance(result, LexicalDiversityResult)
    assert result.n_tokens == 25
    assert result.n_types == 5


def test_lexical_diversity_to_df_has_all_metrics() -> None:
    corpus = _corpus_of(["alpha beta gamma"] * 3)
    df = pcd.lexical_diversity(corpus).to_df()
    assert set(df["metric"]) == {"TTR", "MATTR", "MTLD", "HD-D"}


def test_lexical_diversity_per_doc_table() -> None:
    corpus = _corpus_of(["alpha alpha beta", "gamma delta delta"])
    result = pcd.lexical_diversity(corpus, per_doc=True)
    assert result.per_doc is not None
    assert len(result.per_doc) == 2
    assert {"doc_id", "n_tokens", "ttr", "mattr", "mtld", "hdd"}.issubset(
        result.per_doc.columns
    )


def test_lexical_diversity_empty_corpus_raises() -> None:
    empty_df = pd.DataFrame({"text": []})
    empty_df["text"] = empty_df["text"].astype(str)
    corpus = pcd.from_dataframe(empty_df, text_col="text")
    with pytest.raises(ValueError, match="at least one document"):
        pcd.lexical_diversity(corpus)


def test_lexical_diversity_metrics_subset_only_computes_requested() -> None:
    corpus = _corpus_of(["alpha beta gamma"] * 4)
    result = pcd.lexical_diversity(corpus, metrics=["mtld", "hdd"])
    # The unrequested metrics are populated as NaN.
    assert np.isnan(result.ttr)
    assert np.isnan(result.mattr)
    assert not np.isnan(result.mtld)
    assert not np.isnan(result.hdd)


def test_lexical_diversity_unknown_metric_raises() -> None:
    corpus = _corpus_of(["alpha beta"])
    with pytest.raises(ValueError, match="unknown metric"):
        pcd.lexical_diversity(corpus, metrics=["bogus"])


# ----------------------------------------------------------------------
# Public lexical_diversity — temporal trajectory mode
# ----------------------------------------------------------------------


def _temporal_corpus() -> pcd.Corpus:
    texts = []
    dates = []
    for year in range(2010, 2020):
        # Each year gets 10 docs of ~15 tokens each.
        for i in range(10):
            txt = " ".join(
                f"w{(year + i + j) % 30}" for j in range(15)
            )
            texts.append(txt)
            dates.append(f"{year}-06-15")
    return _corpus_of(texts, dates=dates)


def test_lexical_diversity_temporal_returns_trajectory() -> None:
    corpus = _temporal_corpus()
    traj = pcd.lexical_diversity(corpus, freq="Y", time_col="date")
    assert isinstance(traj, LexicalDiversityTrajectory)
    assert traj.freq == "Y"
    assert traj.table["metric"].nunique() == 4
    assert traj.table["period"].nunique() == 10


def test_lexical_diversity_temporal_missing_time_col_raises() -> None:
    corpus = _corpus_of(["alpha beta gamma"] * 3)
    with pytest.raises(ValueError, match="missing from corpus.docs"):
        pcd.lexical_diversity(corpus, freq="Y", time_col="date")


def test_lexical_diversity_temporal_bootstrap_adds_ci_columns() -> None:
    corpus = _temporal_corpus()
    traj = pcd.lexical_diversity(
        corpus, freq="Y", time_col="date", ci="bootstrap", n_boot=29, seed=0
    )
    assert "ci_lower" in traj.table.columns
    assert "ci_upper" in traj.table.columns
    # Lower <= upper for every row (with NaN tolerance — degenerate
    # resamples can produce NaN; ignore those).
    finite = traj.table.dropna(subset=["ci_lower", "ci_upper"])
    assert (finite["ci_lower"] <= finite["ci_upper"]).all()


def test_lexical_diversity_temporal_bootstrap_reproducible_under_seed() -> None:
    corpus = _temporal_corpus()
    t1 = pcd.lexical_diversity(
        corpus, freq="Y", time_col="date", ci="bootstrap", n_boot=49, seed=7
    )
    t2 = pcd.lexical_diversity(
        corpus, freq="Y", time_col="date", ci="bootstrap", n_boot=49, seed=7
    )
    pd.testing.assert_frame_equal(t1.table, t2.table)


def test_lexical_diversity_temporal_ci_level_widens_band() -> None:
    corpus = _temporal_corpus()
    narrow = pcd.lexical_diversity(
        corpus, freq="Y", time_col="date", ci="bootstrap", n_boot=99,
        ci_level=0.80, seed=3,
    )
    wide = pcd.lexical_diversity(
        corpus, freq="Y", time_col="date", ci="bootstrap", n_boot=99,
        ci_level=0.99, seed=3,
    )
    # Width on TTR (order-independent statistic so monotone bound holds)
    narrow_ttr = narrow.table.query("metric == 'TTR'").set_index("period")
    wide_ttr = wide.table.query("metric == 'TTR'").set_index("period")
    narrow_w = narrow_ttr["ci_upper"] - narrow_ttr["ci_lower"]
    wide_w = wide_ttr["ci_upper"] - wide_ttr["ci_lower"]
    assert (wide_w >= narrow_w - 1e-9).all()


def test_lexical_diversity_temporal_invalid_ci_raises() -> None:
    corpus = _temporal_corpus()
    with pytest.raises(ValueError, match="ci must be 'none' or 'bootstrap'"):
        pcd.lexical_diversity(corpus, freq="Y", time_col="date", ci="bogus")  # type: ignore[arg-type]


def test_lexical_diversity_invalid_ci_level_raises() -> None:
    corpus = _temporal_corpus()
    with pytest.raises(ValueError, match="ci_level must be in"):
        pcd.lexical_diversity(corpus, freq="Y", time_col="date", ci_level=1.5)


def test_lexical_diversity_invalid_n_boot_raises() -> None:
    corpus = _temporal_corpus()
    with pytest.raises(ValueError, match="n_boot must be >= 1"):
        pcd.lexical_diversity(corpus, freq="Y", time_col="date", n_boot=0)


def test_lexical_diversity_metrics_subset_in_trajectory() -> None:
    corpus = _temporal_corpus()
    traj = pcd.lexical_diversity(
        corpus, freq="Y", time_col="date", metrics=["mtld"]
    )
    assert set(traj.table["metric"]) == {"MTLD"}


def test_lexical_diversity_short_period_warns() -> None:
    """Periods with very few tokens should emit a warning."""
    df = pd.DataFrame(
        {
            "text": ["a b", "c d"] + ["w" + str(i) for i in range(60)],
            "date": (
                ["2010-01-01", "2010-01-02"]
                + [f"2011-{(i % 12) + 1:02d}-01" for i in range(60)]
            ),
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    corpus = pcd.from_dataframe(df, text_col="text")
    with pytest.warns(UserWarning, match="tokens"):
        pcd.lexical_diversity(corpus, freq="Y", time_col="date")


# ----------------------------------------------------------------------
# Hansard sample integration
# ----------------------------------------------------------------------


def test_lexical_diversity_runs_on_hansard_sample_pooled() -> None:
    corpus = pcd.load_hansard_sample()
    result = pcd.lexical_diversity(corpus)
    assert result.n_tokens > 1_000
    # All four metrics finite.
    assert all(
        not np.isnan(getattr(result, m))
        for m in ["ttr", "mattr", "mtld", "hdd"]
    )


def test_lexical_diversity_runs_on_hansard_sample_temporal() -> None:
    corpus = pcd.load_hansard_sample()
    traj = pcd.lexical_diversity(corpus, freq="Y", time_col="date")
    # Synthetic corpus spans 2005-2023 → 19 yearly periods.
    assert traj.table["period"].nunique() == 19


# ----------------------------------------------------------------------
# Hill numbers + rarefaction (ecology diversity, size-fair comparison)
# ----------------------------------------------------------------------
def test_hill_numbers_uniform_equals_richness_all_orders() -> None:
    # a perfectly even community of S types -> Hill_q = S for every q
    counts = [10] * 50
    for q in (0.0, 1.0, 2.0):
        assert hill_numbers(counts, q) == pytest.approx(50.0)


def test_hill_numbers_ordering_for_skewed() -> None:
    # for an uneven community, richness >= exp-Shannon >= inv-Simpson
    counts = [100, 50, 10, 5, 1, 1, 1]
    d0, d1, d2 = (hill_numbers(counts, q) for q in (0.0, 1.0, 2.0))
    assert d0 > d1 > d2
    assert d0 == pytest.approx(7.0)  # q=0 is plain richness


def test_hill_numbers_empty_is_zero() -> None:
    assert hill_numbers([]) == 0.0
    assert hill_numbers([0, 0]) == 0.0


def test_rarefaction_at_full_size_is_richness() -> None:
    counts = [5, 3, 2, 1]
    assert rarefaction(counts, sum(counts)) == pytest.approx(4.0)


def test_rarefaction_controls_for_size() -> None:
    # two samples from the SAME relative community, different totals:
    # naive richness differs, but rarefied-to-common-size is ~equal.
    base = np.array([100, 60, 40, 25, 15, 10, 6, 4, 2, 1], dtype=float)
    small = (base * 3).astype(int)
    big = (base * 30).astype(int)
    assert (big > 0).sum() == (small > 0).sum()  # same #types here, but...
    # richness is identical only because no zeros; the real test is that
    # rarefying the big one to the small one's size matches small's richness
    n = int(small.sum())
    assert rarefaction(big, n) == pytest.approx(rarefaction(small, n), rel=0.02)


def test_rarefaction_matches_hdd_at_42() -> None:
    tokens = (["the"] * 40 + ["cat"] * 20 + ["sat"] * 10
              + [f"w{i}" for i in range(30)])
    counts = list(pd.Series(tokens).value_counts())
    assert rarefaction(counts, 42) == pytest.approx(hdd(tokens, 42), rel=1e-9)


def test_rarefaction_invalid_size_raises() -> None:
    with pytest.raises(ValueError, match="sample_size must be in"):
        rarefaction([5, 3, 2], 99)
