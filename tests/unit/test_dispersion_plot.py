"""Tests for ``pycorpdiff.viz.dispersion_plot``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.viz.dispersion import dispersion_plot

pytest.importorskip("altair")


def _corpus_with_clustering() -> pcd.Corpus:
    """Corpus where 'cluster' appears only at the start, 'spread' is even."""
    rows = []
    # Docs 0-2: cluster + spread
    for _ in range(3):
        rows.append({"text": "the cluster term appears here and the spread term too"})
    # Docs 3-9: only spread (clustered at start = NOT here)
    for _ in range(7):
        rows.append({"text": "no special markers here just the spread term"})
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text")


def test_dispersion_plot_returns_altair_chart() -> None:
    import altair as alt

    corpus = _corpus_with_clustering()
    chart = dispersion_plot(corpus, "cluster")
    assert isinstance(chart, alt.Chart) or hasattr(chart, "to_dict")


def test_dispersion_plot_single_target_uses_correct_encoding() -> None:
    corpus = _corpus_with_clustering()
    spec = dispersion_plot(corpus, "cluster").to_dict()
    enc = spec["encoding"]
    assert enc["x"]["field"] == "doc_index"
    assert enc["y"]["field"] == "target"


def test_dispersion_plot_data_reflects_clustering() -> None:
    """'cluster' should appear only in docs 0-2; 'spread' in 0-9."""
    corpus = _corpus_with_clustering()
    spec = dispersion_plot(corpus, ["cluster", "spread"]).to_dict()
    # altair embeds the data inline by default
    data = spec["datasets"][next(iter(spec["datasets"]))]
    cluster_docs = {r["doc_index"] for r in data if r["target"] == "cluster"}
    spread_docs = {r["doc_index"] for r in data if r["target"] == "spread"}
    assert cluster_docs == {0, 1, 2}
    assert spread_docs == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_dispersion_plot_multi_target() -> None:
    corpus = _corpus_with_clustering()
    chart = dispersion_plot(corpus, ["cluster", "spread"])
    spec = chart.to_dict()
    # Y-axis sort order matches the target list.
    assert spec["encoding"]["y"]["sort"] == ["cluster", "spread"]


def test_dispersion_plot_handles_zero_occurrences() -> None:
    """A target that doesn't appear should produce an empty chart, not raise."""
    corpus = _corpus_with_clustering()
    chart = dispersion_plot(corpus, "unicorn")
    spec = chart.to_dict()
    # Should still be a valid spec.
    assert "encoding" in spec


def test_dispersion_plot_height_scales_with_target_count() -> None:
    corpus = _corpus_with_clustering()
    one = dispersion_plot(corpus, "cluster").to_dict()
    three = dispersion_plot(
        corpus, ["cluster", "spread", "term"]
    ).to_dict()
    assert three["height"] > one["height"]


def test_dispersion_plot_works_on_corpus_slice() -> None:
    """A slice of the corpus should work as input."""
    rows = [{"text": "one cluster", "outlet": "A"}] * 3 + [
        {"text": "no terms here", "outlet": "B"}
    ] * 5
    corpus = pcd.from_dataframe(
        pd.DataFrame(rows), text_col="text", meta_cols=("outlet",)
    )
    a = corpus.slice(outlet="A")
    chart = dispersion_plot(a, "cluster")
    assert chart.to_dict()["encoding"]["x"]["field"] == "doc_index"


def test_dispersion_plot_is_exported_at_module_level() -> None:
    """`from pycorpdiff.viz import dispersion_plot` should just work."""
    from pycorpdiff.viz import dispersion_plot as imported

    assert imported is dispersion_plot
