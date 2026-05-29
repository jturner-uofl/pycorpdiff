"""Tests for reference-corpus baseline keyness and the bundled baselines."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.datasets.baselines import (
    Baseline,
    baseline_from_corpus,
    list_baselines,
    load_baseline,
)


def _corpus_from_strings(texts: list[str]) -> pcd.Corpus:
    return pcd.from_dataframe(pd.DataFrame({"text": texts}), text_col="text")


def _toy_baseline(counts: dict[str, int], total: int | None = None) -> Baseline:
    """Hand-built tiny baseline so tests don't need the bundled parquet."""
    series = pd.Series(counts, dtype="int64", name="toy")
    return Baseline(
        counts=series,
        total_tokens=int(total if total is not None else series.sum()),
        n_documents=1,
        name="toy",
        metadata={"description": "toy baseline for tests"},
    )


# ----------------------------------------------------------------------
# Bundled-baseline plumbing
# ----------------------------------------------------------------------


def test_list_baselines_includes_gutenberg_fiction() -> None:
    names = list_baselines()
    assert "gutenberg_fiction" in names


def test_load_baseline_returns_populated_object() -> None:
    bl = load_baseline("gutenberg_fiction")
    assert isinstance(bl, Baseline)
    assert bl.name == "gutenberg_fiction"
    assert bl.total_tokens > 100_000  # ~500K in the bundled file
    assert len(bl.counts) > 1_000
    # Counts are integer-valued non-negative.
    assert (bl.counts >= 0).all()
    assert bl.counts.dtype.kind == "i"


def test_load_baseline_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="unknown baseline"):
        load_baseline("does_not_exist")


def test_baseline_metadata_carries_provenance() -> None:
    bl = load_baseline("gutenberg_fiction")
    assert "license" in bl.metadata
    assert "Public Domain" in str(bl.metadata["license"])
    assert "books" in bl.metadata
    # The five sourced books should each have an entry.
    assert len(bl.metadata["books"]) == 5


# ----------------------------------------------------------------------
# against_baseline (core path)
# ----------------------------------------------------------------------


def test_against_baseline_against_bundled_returns_keyness_table() -> None:
    corpus = pcd.load_hansard_sample()
    result = pcd.against_baseline(corpus, "gutenberg_fiction", min_count=5)
    # The result is a real KeynessResult with the usual columns.
    assert set(result.table.columns) >= {
        "term", "count_a", "count_b", "g2", "p_value"
    }
    # corpus_a is the input; corpus_b is None (baseline has no docs).
    assert result.corpus_a is corpus
    assert result.corpus_b is None
    # The baseline-name shows up in label_b and params.
    assert "gutenberg_fiction" in result.label_b
    assert result.params["baseline_name"] == "gutenberg_fiction"


def test_against_baseline_accepts_baseline_object_directly() -> None:
    """Passing a :class:`Baseline` should bypass the bundled-name path."""
    corpus = _corpus_from_strings(["alpha alpha beta gamma"] * 10)
    bl = _toy_baseline({"alpha": 1, "beta": 100, "delta": 50}, total=200)
    result = pcd.against_baseline(corpus, bl, min_count=1)
    # 'alpha' is over-represented in the corpus → positive G².
    row_alpha = result.table[result.table["term"] == "alpha"].iloc[0]
    assert row_alpha["g2"] > 0.0
    # 'beta' is under-represented → negative G².
    row_beta = result.table[result.table["term"] == "beta"].iloc[0]
    assert row_beta["g2"] < 0.0


def test_against_baseline_min_count_filters_low_freq_terms() -> None:
    corpus = _corpus_from_strings(["alpha beta gamma " * 5] * 5)
    bl = _toy_baseline({"alpha": 10, "beta": 10, "gamma": 10, "delta": 1}, total=31)
    result = pcd.against_baseline(corpus, bl, min_count=100)
    # min_count=100 should knock out every term.
    assert len(result.table) == 0


def test_against_baseline_dunning_formula_runs() -> None:
    corpus = _corpus_from_strings(["distinctive shared shared"] * 20)
    bl = _toy_baseline({"shared": 100, "mundane": 50}, total=150)
    result = pcd.against_baseline(corpus, bl, formula="dunning", min_count=1)
    assert "g2" in result.table.columns
    # 'distinctive' (corpus-only) should be the top key term.
    assert result.table.iloc[0]["term"] == "distinctive"


def test_against_baseline_empty_corpus_raises() -> None:
    empty = pd.DataFrame({"text": []})
    empty["text"] = empty["text"].astype(str)
    corpus = pcd.from_dataframe(empty, text_col="text")
    bl = _toy_baseline({"alpha": 10})
    with pytest.raises(ValueError, match="at least one document"):
        pcd.against_baseline(corpus, bl)


def test_against_baseline_zero_token_baseline_raises() -> None:
    corpus = _corpus_from_strings(["one two three"])
    bl = Baseline(
        counts=pd.Series({"alpha": 0}, dtype="int64"),
        total_tokens=0,
        n_documents=1,
        name="empty",
        metadata={},
    )
    with pytest.raises(ValueError, match="zero tokens"):
        pcd.against_baseline(corpus, bl)


def test_against_baseline_signed_g2_direction() -> None:
    """Sign convention: positive G² when the corpus over-uses the term."""
    corpus = _corpus_from_strings(["over over over over"] * 50)
    bl = _toy_baseline({"over": 1, "filler": 999}, total=1000)
    result = pcd.against_baseline(corpus, bl, min_count=1)
    over_row = result.table[result.table["term"] == "over"].iloc[0]
    assert over_row["g2"] > 0.0


def test_against_baseline_sort_method_log_ratio() -> None:
    corpus = _corpus_from_strings(["alpha alpha beta gamma"] * 20)
    bl = _toy_baseline({"alpha": 1, "beta": 100, "gamma": 100}, total=201)
    result = pcd.against_baseline(corpus, bl, method="log_ratio", min_count=1)
    assert result.method == "log_ratio"
    # Sorted by |log_ratio| descending → first row has largest absolute LR.
    abs_lr = result.table["log_ratio"].abs()
    assert (abs_lr.iloc[0] >= abs_lr).all()


def test_against_baseline_bh_correction_adds_p_adjusted() -> None:
    corpus = _corpus_from_strings(["alpha beta gamma " * 8] * 15)
    bl = _toy_baseline({"alpha": 100, "beta": 100, "gamma": 100, "delta": 100}, total=400)
    result = pcd.against_baseline(corpus, bl, multiple_comparisons="bh", min_count=1)
    assert "p_adjusted" in result.table.columns


# ----------------------------------------------------------------------
# explain() on a baseline result (corpus_b is None)
# ----------------------------------------------------------------------


def test_keyness_result_explain_with_no_corpus_b_returns_kwic() -> None:
    corpus = _corpus_from_strings(
        ["distinctive shared", "shared shared distinctive context"]
    )
    bl = _toy_baseline({"shared": 100, "mundane": 50}, total=150)
    result = pcd.against_baseline(corpus, bl, min_count=1)
    exp = result.explain("distinctive", n=10)
    # All KWIC lines come from corpus_a; no exception.
    assert len(exp.table) >= 1
    assert (exp.table["keyword"] == "distinctive").all()


def test_keyness_result_explain_without_any_corpora_raises() -> None:
    """Bare-DataFrame constructions still raise on explain()."""
    from pycorpdiff.results import KeynessResult

    bare = KeynessResult(
        table=pd.DataFrame({"term": ["x"], "g2": [1.0]}),
        method="log_likelihood",
        n_a=1,
        n_b=1,
    )
    with pytest.raises(ValueError, match="A-side corpus"):
        bare.explain("x")


# ----------------------------------------------------------------------
# baseline_from_corpus (user-supplied baselines)
# ----------------------------------------------------------------------


def test_baseline_from_corpus_round_trips_against_corpus_to_corpus_keyness() -> None:
    """A baseline built from a corpus, then used as the B-side via
    ``against_baseline``, should agree exactly with the equivalent
    ``compare(a, b).keyness()`` call (same math, same totals).
    """
    a = _corpus_from_strings(["alpha alpha beta", "alpha gamma gamma"] * 5)
    b = _corpus_from_strings(["beta beta gamma delta", "delta delta delta"] * 5)
    bl = baseline_from_corpus(b, min_count=1)
    via_baseline = pcd.against_baseline(a, bl, min_count=1, effect_size=False)
    via_compare = pcd.compare(a, b).keyness(min_count=1, effect_size=False)
    # Align tables by term and check G² agreement.
    bd = via_baseline.table.set_index("term")["g2"]
    cd = via_compare.table.set_index("term")["g2"]
    pd.testing.assert_series_equal(
        bd.sort_index(), cd.reindex(bd.index).sort_index(), check_names=False
    )


def test_baseline_from_corpus_min_count_zero_raises() -> None:
    a = _corpus_from_strings(["alpha beta gamma"])
    with pytest.raises(ValueError, match="min_count must be >= 1"):
        baseline_from_corpus(a, min_count=0)


def test_baseline_from_corpus_empty_raises() -> None:
    empty = pd.DataFrame({"text": []})
    empty["text"] = empty["text"].astype(str)
    a = pcd.from_dataframe(empty, text_col="text")
    with pytest.raises(ValueError, match="at least one document"):
        baseline_from_corpus(a)


def test_baseline_from_corpus_preserves_total_tokens_before_filtering() -> None:
    """Hapax filtering removes terms from the counts but ``total_tokens``
    stays the true source-corpus total — the keyness math needs the full
    denominator, not the post-filter sum."""
    a = _corpus_from_strings(
        [
            "common common common rare1 rare2 rare3",
            "common common rare4 rare5 rare6",
        ]
    )
    bl = baseline_from_corpus(a, min_count=2)
    # 'common' kept, every rareN dropped.
    assert "common" in bl.counts.index
    assert all(t not in bl.counts.index for t in ["rare1", "rare2", "rare3"])
    # But total_tokens reflects the un-filtered total.
    assert bl.total_tokens == 11  # 5+3 commons + 6 rares = 11


def test_baseline_repr_is_human_readable() -> None:
    bl = _toy_baseline({"alpha": 10, "beta": 20})
    text = repr(bl)
    assert "Baseline" in text and "toy" in text
