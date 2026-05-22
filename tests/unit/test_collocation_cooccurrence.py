"""Tests for the window-based co-occurrence extractor."""

from __future__ import annotations

import pytest

from pycorpdiff.collocation.cooccurrence import collocate_counts


def test_basic_window_counts_neighbours() -> None:
    docs = [["a", "b", "TARGET", "c", "d"]]
    counts, n = collocate_counts(docs, target="TARGET", window=2)
    assert n == 1
    assert counts["a"] == 1
    assert counts["b"] == 1
    assert counts["c"] == 1
    assert counts["d"] == 1
    assert "TARGET" not in counts


def test_window_size_one_only_immediate_neighbours() -> None:
    docs = [["a", "b", "TARGET", "c", "d"]]
    counts, _ = collocate_counts(docs, target="TARGET", window=1)
    assert dict(counts) == {"b": 1, "c": 1}


def test_window_does_not_cross_document_boundaries() -> None:
    # If the window crossed documents we'd see 'after' counted as a
    # collocate of 'TARGET'. With per-doc isolation it's never adjacent.
    docs = [["TARGET"], ["after"]]
    counts, n = collocate_counts(docs, target="TARGET", window=5)
    assert n == 1
    assert "after" not in counts


def test_target_excluded_from_its_own_window() -> None:
    docs = [["TARGET"]]
    counts, n = collocate_counts(docs, target="TARGET", window=5)
    assert n == 1
    assert len(counts) == 0


def test_two_targets_in_same_window_each_contribute() -> None:
    # 'a' is within window of both target occurrences -> count = 2.
    docs = [["a", "TARGET", "TARGET", "b"]]
    counts, n = collocate_counts(docs, target="TARGET", window=2)
    assert n == 2
    assert counts["a"] == 2
    assert counts["b"] == 2
    # The two targets are within window of each other but excluded by the
    # self-position rule applied per-occurrence — so each is in the other's
    # window.
    assert counts.get("TARGET", 0) == 2


def test_target_absent_yields_zero_n() -> None:
    docs = [["a", "b", "c"]]
    counts, n = collocate_counts(docs, target="nope", window=5)
    assert n == 0
    assert len(counts) == 0


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window must be"):
        collocate_counts([["a"]], target="a", window=0)
    with pytest.raises(ValueError, match="window must be"):
        collocate_counts([["a"]], target="a", window=-1)


def test_handles_target_at_document_edges() -> None:
    # Target at position 0 has no left context; at the last position no right.
    docs = [["TARGET", "a", "b", "c", "d", "TARGET"]]
    counts, n = collocate_counts(docs, target="TARGET", window=2)
    assert n == 2
    # First target's window: positions 1, 2 → a, b
    # Second target's window: positions 3, 4 → c, d
    assert counts["a"] == 1 and counts["b"] == 1
    assert counts["c"] == 1 and counts["d"] == 1
