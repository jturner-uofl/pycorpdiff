"""Unit tests for the ``collocation_shift`` orchestrator."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.collocation.shift import collocation_shift


def _two_corpus_fixture() -> tuple[pcd.Corpus, pcd.Corpus]:
    a_docs = [
        "the immigrant worker arrived and the immigrant family settled",
        "the immigrant community grew the immigrant worker thrived",
        "the immigrant worker and the immigrant family stayed",
        "the immigrant worker and immigrant rights advanced",
    ]
    b_docs = [
        "the immigrant criminal threat and the immigrant invasion grew",
        "the immigrant threat and the immigrant crime increased",
        "the immigrant invasion of immigrant criminal gangs spread",
        "the immigrant criminal gangs and immigrant invasion stayed",
    ]
    a = pcd.from_dataframe(pd.DataFrame({"text": a_docs}), text_col="text")
    b = pcd.from_dataframe(pd.DataFrame({"text": b_docs}), text_col="text")
    return a, b


def test_collocation_shift_returns_expected_schema() -> None:
    a, b = _two_corpus_fixture()
    df = collocation_shift(a, b, target="immigrant", window=3, min_count=1)
    assert list(df.columns) == ["count_a", "count_b", "score_a", "score_b", "shift"]
    assert df.index.name == "collocate"
    # Adjacent target occurrences (e.g. "the immigrant immigrant worker")
    # do collocate with each other — the target may legitimately appear
    # in its own collocate list. That's the documented behaviour of the
    # cooccurrence extractor; see test_collocation_cooccurrence.


def test_collocation_shift_signs_match_corpus_overuse() -> None:
    a, b = _two_corpus_fixture()
    df = collocation_shift(a, b, target="immigrant", window=3, min_count=1)
    # 'worker' / 'family' / 'community' / 'rights' should swing toward A (positive shift).
    # 'criminal' / 'invasion' / 'threat' / 'gangs' should swing toward B (negative shift).
    pos_terms = {"worker", "family", "community", "rights"}
    neg_terms = {"criminal", "invasion", "threat", "gangs"}
    for term in pos_terms & set(df.index):
        assert df.loc[term, "shift"] > 0, (
            f"expected {term!r} to be positive (A-leaning); got {df.loc[term, 'shift']}"
        )
    for term in neg_terms & set(df.index):
        assert df.loc[term, "shift"] < 0, (
            f"expected {term!r} to be negative (B-leaning); got {df.loc[term, 'shift']}"
        )


def test_collocation_shift_swap_negates_per_collocate() -> None:
    a, b = _two_corpus_fixture()
    forward = collocation_shift(a, b, target="immigrant", window=3, min_count=1)
    reverse = collocation_shift(b, a, target="immigrant", window=3, min_count=1)
    # shift_ab[c] should equal -shift_ba[c] for every shared collocate.
    shared = forward.index.intersection(reverse.index)
    for term in shared:
        assert math.isclose(
            forward.loc[term, "shift"], -reverse.loc[term, "shift"], rel_tol=1e-9
        )


def test_collocation_shift_target_missing_raises() -> None:
    a, b = _two_corpus_fixture()
    with pytest.raises(ValueError, match="target 'unicorn' not found in corpus a"):
        collocation_shift(a, b, target="unicorn")


def test_collocation_shift_min_count_filters() -> None:
    a, b = _two_corpus_fixture()
    lo = collocation_shift(a, b, target="immigrant", min_count=1, window=3)
    hi = collocation_shift(a, b, target="immigrant", min_count=5, window=3)
    assert len(lo) > len(hi)


def test_collocation_shift_rejects_bad_smoothing() -> None:
    a, b = _two_corpus_fixture()
    with pytest.raises(ValueError, match="smoothing must be > 0"):
        collocation_shift(a, b, target="immigrant", smoothing=0)


def test_collocation_shift_rejects_unknown_measure() -> None:
    a, b = _two_corpus_fixture()
    with pytest.raises(ValueError, match="unknown measure"):
        collocation_shift(a, b, target="immigrant", measure="bogus")  # type: ignore[arg-type]


def test_collocation_shift_supports_all_measures() -> None:
    a, b = _two_corpus_fixture()
    for measure in ("logDice", "PMI", "t_score", "MI3"):
        df = collocation_shift(a, b, target="immigrant", window=3, min_count=1, measure=measure)  # type: ignore[arg-type]
        assert "shift" in df.columns
        assert len(df) > 0


def test_collocation_shift_below_min_count_returns_empty_frame() -> None:
    a, b = _two_corpus_fixture()
    df = collocation_shift(a, b, target="immigrant", min_count=1_000)
    assert len(df) == 0
    assert list(df.columns) == ["count_a", "count_b", "score_a", "score_b", "shift"]


def test_collocation_shift_sorted_by_abs_shift() -> None:
    a, b = _two_corpus_fixture()
    df = collocation_shift(a, b, target="immigrant", window=3, min_count=1)
    # |shift| should be non-increasing down the rows.
    abs_shifts = df["shift"].abs().to_numpy()
    assert all(abs_shifts[i] >= abs_shifts[i + 1] - 1e-12 for i in range(len(abs_shifts) - 1))
