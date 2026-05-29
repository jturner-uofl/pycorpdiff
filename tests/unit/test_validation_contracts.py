"""Regression tests for the boundary-condition + contract fixes shipped in 0.1.0a7.

Each test corresponds to a specific finding that, prior to the fix,
either silently returned wrong values or raised an opaque internal
exception. Keep these as trip-wires so future refactors don't quietly
reintroduce the same issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.keyness.bayes import bayes_factor

# ---------------------------------------------------------------------
# bayes_factor honours the formula= choice (no longer silently rayson)
# ---------------------------------------------------------------------


def test_bayes_factor_differs_when_formula_changes() -> None:
    """Prior bug: keyness(formula='dunning') returned a row where g2 was
    Dunning 4-cell but bayes_factor was always Rayson 2-cell — two
    different statistics on the same row. Fix: forward formula= through."""
    a = pd.Series({"x": 50})
    b = pd.Series({"x": 50})
    bf_rayson = float(bayes_factor(a, b, 1_000, 100_000, formula="rayson").loc["x"])
    bf_dunning = float(bayes_factor(a, b, 1_000, 100_000, formula="dunning").loc["x"])
    assert bf_rayson != bf_dunning, (
        f"BF must change with formula; got rayson={bf_rayson}, dunning={bf_dunning}"
    )


def test_keyness_propagates_formula_to_bayes_factor() -> None:
    """End-to-end: a Result row must have g2 and bayes_factor that
    describe the *same* underlying G² formulation."""
    corpus = pcd.load_hansard_sample()
    a = corpus.slice(topic="immigration", frame="humanising")
    b = corpus.slice(topic="immigration", frame="criminalising")
    r = pcd.compare(a, b).keyness(min_count=3, formula="rayson")
    d = pcd.compare(a, b).keyness(min_count=3, formula="dunning")
    assert r.table.iloc[0]["bayes_factor"] != d.table.iloc[0]["bayes_factor"]
    assert r.params["formula"] == "rayson"
    assert d.params["formula"] == "dunning"


# ---------------------------------------------------------------------
# semantic_shift validates embedder output shape
# ---------------------------------------------------------------------


class _OneDEmbedder:
    """Returns a 1-D vector instead of (n_terms, d). The pre-fix code
    silently averaged this into a scalar and reported cosine 1.0."""

    def encode(self, terms: list[str]) -> np.ndarray:
        return np.ones(len(terms))


class _WrongRowCountEmbedder:
    """Returns the wrong number of rows. Pre-fix would silently broadcast."""

    def encode(self, terms: list[str]) -> np.ndarray:
        return np.ones((len(terms) + 5, 8))


def test_semantic_shift_rejects_one_dim_embedding() -> None:
    corpus = pcd.load_hansard_sample()
    a = corpus.slice(topic="immigration", frame="humanising")
    b = corpus.slice(topic="immigration", frame="criminalising")
    with pytest.raises(ValueError, match="rank"):
        pcd.compare(a, b).semantic_shift("immigrant", embedder=_OneDEmbedder())


def test_semantic_shift_rejects_wrong_row_count() -> None:
    corpus = pcd.load_hansard_sample()
    a = corpus.slice(topic="immigration", frame="humanising")
    b = corpus.slice(topic="immigration", frame="criminalising")
    with pytest.raises(ValueError, match="rows"):
        pcd.compare(a, b).semantic_shift("immigrant", embedder=_WrongRowCountEmbedder())


# ---------------------------------------------------------------------
# track() / by_time() reject empty corpora and missing time_col
# ---------------------------------------------------------------------


def test_track_empty_corpus_raises_value_error_not_raw_keyerror() -> None:
    empty = pcd.from_dataframe(pd.DataFrame({"text": []}), text_col="text")
    with pytest.raises(ValueError, match="non-empty"):
        pcd.track(empty, "foo").over_time(freq="Y", time_col="date")


def test_track_missing_time_col_raises_value_error_not_raw_keyerror() -> None:
    small = pcd.from_dataframe(pd.DataFrame({"text": ["a", "b"]}), text_col="text")
    with pytest.raises(ValueError, match="not found"):
        pcd.track(small, "a").over_time(freq="Y", time_col="date")


# ---------------------------------------------------------------------
# read_duckdb rejects a string path with a clear TypeError
# ---------------------------------------------------------------------


def test_read_duckdb_rejects_string_path() -> None:
    """Pre-fix: passing a path raised AttributeError: 'str' has no
    attribute 'execute'. Now it raises a typed error pointing at the
    duckdb.connect() pattern."""
    with pytest.raises(TypeError, match="connection"):
        pcd.read_duckdb("/tmp/nonexistent.duckdb", "SELECT 1")


# ---------------------------------------------------------------------
# SemanticShiftResult.plot returns a renderable chart
# ---------------------------------------------------------------------


def test_semantic_shift_plot_returns_chart() -> None:
    """Prior to 0.1.0a7 this raised NotImplementedError; now it returns
    a horizontal-bar altair chart of per-target cosine distance."""
    altair = pytest.importorskip("altair")
    corpus = pcd.load_hansard_sample()
    a = corpus.slice(topic="immigration", frame="humanising")
    b = corpus.slice(topic="immigration", frame="criminalising")
    r = pcd.compare(a, b).semantic_shift(
        ["immigrant", "criminal"], embedder=pcd.HashEmbedder()
    )
    chart = r.plot()
    assert isinstance(chart, altair.Chart)
    spec = chart.to_dict()
    assert spec["mark"]["type"] == "bar"
