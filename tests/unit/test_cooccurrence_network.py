"""Tests for term co-occurrence networks and the network plot."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.collocation.network import NetworkResult, cooccurrence_network


def _doc_corpus(docs: list[str]) -> pcd.Corpus:
    return pcd.from_dataframe(pd.DataFrame({"text": docs}), text_col="text")


@pytest.fixture
def toy() -> pcd.Corpus:
    """Three docs with a clear bigram structure: 'foo bar' is the strong pair."""
    return _doc_corpus(
        ["foo bar baz qux"] * 10
        + ["foo bar"] * 5
        + ["lonely orphan"] * 3
    )


def test_returns_network_result(toy: pcd.Corpus) -> None:
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    assert isinstance(net, NetworkResult)
    assert net.measure == "PMI"
    assert net.window == 5


def test_nodes_indexed_by_term_with_count_and_degree(toy: pcd.Corpus) -> None:
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    assert net.nodes.index.name in (None, "term") or net.nodes.index.name == ""
    assert "count" in net.nodes.columns
    assert "degree" in net.nodes.columns
    # 'foo' and 'bar' should be the most-connected nodes.
    top_2_degree = set(net.nodes.sort_values("degree", ascending=False).head(2).index)
    assert {"foo", "bar"}.issubset(top_2_degree)


def test_edges_have_required_columns(toy: pcd.Corpus) -> None:
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    for col in ("source", "target", "cooccur_count", "weight", "rank"):
        assert col in net.edges.columns


def test_edges_sorted_by_absolute_weight_descending(toy: pcd.Corpus) -> None:
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    abs_weights = net.edges["weight"].abs().to_numpy()
    assert (abs_weights[:-1] >= abs_weights[1:] - 1e-12).all()
    # rank == 0..n-1
    assert list(net.edges["rank"]) == list(range(len(net.edges)))


def test_undirected_edges_have_canonical_ordering(toy: pcd.Corpus) -> None:
    """source < target lexicographically — every edge appears once."""
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    for _, row in net.edges.iterrows():
        assert row["source"] < row["target"]


def test_window_bounds_cooccurrence_correctly() -> None:
    """A 1-token window only counts immediate neighbors."""
    corpus = _doc_corpus(["a b c d e"] * 5)
    net = cooccurrence_network(corpus, top_n=10, window=1, min_count=1, min_cooccur=1)
    pairs = {(s, t) for s, t in zip(net.edges["source"], net.edges["target"], strict=True)}
    # Only adjacent pairs: (a,b), (b,c), (c,d), (d,e) — canonicalised.
    assert pairs == {("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")}


def test_min_cooccur_floor_drops_weak_edges(toy: pcd.Corpus) -> None:
    """Edges below the floor must not appear."""
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=5)
    assert (net.edges["cooccur_count"] >= 5).all()


def test_top_n_caps_node_count() -> None:
    corpus = _doc_corpus([f"word{i}" for i in range(50)])
    # Each word appears only once — but with min_count=1 they all qualify.
    net = cooccurrence_network(corpus, top_n=10, min_count=1, min_cooccur=1)
    assert len(net.nodes) == 10


def test_top_n_lt_2_raises(toy: pcd.Corpus) -> None:
    with pytest.raises(ValueError, match="top_n must be >= 2"):
        cooccurrence_network(toy, top_n=1)


def test_window_lt_1_raises(toy: pcd.Corpus) -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        cooccurrence_network(toy, top_n=5, window=0)


def test_smoothing_non_positive_raises(toy: pcd.Corpus) -> None:
    with pytest.raises(ValueError, match="smoothing must be > 0"):
        cooccurrence_network(toy, top_n=5, smoothing=0)


def test_too_few_terms_after_filter_raises() -> None:
    """If min_count drops the vocab below 2 terms, raise."""
    corpus = _doc_corpus(["a b"])
    with pytest.raises(ValueError, match="need at least 2 terms"):
        cooccurrence_network(corpus, top_n=10, min_count=10)


def test_zero_cooccurrences_returns_empty_edges() -> None:
    """A corpus where no pair ever lands within the window returns an
    empty edges frame and a valid (but degree-0) nodes frame."""
    corpus = _doc_corpus(["alpha"] * 5 + ["beta"] * 5)  # docs never share tokens
    net = cooccurrence_network(corpus, top_n=10, min_count=1, min_cooccur=1)
    assert len(net.edges) == 0
    assert set(net.nodes.index) == {"alpha", "beta"}
    assert (net.nodes["degree"] == 0).all()


def test_t_score_and_mi3_measures_run(toy: pcd.Corpus) -> None:
    """All four supported measures must run end-to-end."""
    for measure in ("PMI", "t_score", "MI3", "logDice"):
        net = cooccurrence_network(
            toy, top_n=10, measure=measure, min_count=1, min_cooccur=1
        )
        assert len(net.edges) > 0
        assert net.measure == measure


def test_unknown_measure_raises(toy: pcd.Corpus) -> None:
    with pytest.raises(ValueError, match="unknown measure"):
        cooccurrence_network(toy, top_n=10, measure="bogus")  # type: ignore[arg-type]


def test_works_on_corpus_slice() -> None:
    df = pd.DataFrame(
        {"text": ["foo bar baz"] * 5 + ["other words"] * 5, "g": ["A"] * 5 + ["B"] * 5}
    )
    corpus = pcd.from_dataframe(df, text_col="text", meta_cols=("g",))
    sliced = corpus.slice(g="A")
    net = cooccurrence_network(sliced, top_n=10, min_count=1, min_cooccur=1)
    assert "foo" in net.nodes.index
    assert "other" not in net.nodes.index


def test_summary_string(toy: pcd.Corpus) -> None:
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    s = net.summary()
    assert "NetworkResult" in s
    assert "PMI" in s
    assert f"nodes={len(net.nodes):,}" in s


def test_to_df_returns_edges_copy(toy: pcd.Corpus) -> None:
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    df = net.to_df()
    pd.testing.assert_frame_equal(df, net.edges)
    # Independent copy
    df.iloc[0, df.columns.get_loc("cooccur_count")] = 999  # type: ignore[call-overload]
    assert net.edges.iloc[0]["cooccur_count"] != 999


def test_exported_at_package_root() -> None:
    assert pcd.cooccurrence_network is cooccurrence_network
    assert pcd.NetworkResult is NetworkResult


# --------------------- viz / plot tests (need altair) -----------------------


def test_plot_returns_altair_chart(toy: pcd.Corpus) -> None:
    pytest.importorskip("altair")
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    chart = net.plot()
    spec = chart.to_dict()
    # Layered chart: edges + nodes + labels
    assert "layer" in spec
    assert len(spec["layer"]) == 3


def test_plot_respects_max_edges(toy: pcd.Corpus) -> None:
    pytest.importorskip("altair")
    net = cooccurrence_network(toy, top_n=10, min_count=1, min_cooccur=1)
    chart = net.plot(max_edges=3)
    spec = chart.to_dict()
    # Find the edges layer (mark rule)
    edge_data_name = None
    for layer in spec["layer"]:
        mark = layer.get("mark", {})
        if isinstance(mark, dict) and mark.get("type") == "rule":
            edge_data_name = (layer.get("data") or {}).get("name")
            break
    assert edge_data_name is not None
    assert len(spec["datasets"][edge_data_name]) <= 3


def test_plot_falls_back_to_circular_layout_when_networkx_missing(
    monkeypatch: pytest.MonkeyPatch, toy: pcd.Corpus
) -> None:
    """Force the networkx import to fail; the plot should still produce
    a valid chart using the circular fallback layout.
    """
    pytest.importorskip("altair")
    net = cooccurrence_network(toy, top_n=8, min_count=1, min_cooccur=1)

    # Replace `import networkx` inside _layout with a failing one.
    import builtins

    real_import = builtins.__import__

    def deny_networkx(name: str, *args: object, **kwargs: object) -> object:
        if name == "networkx":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type,misc]

    monkeypatch.setattr(builtins, "__import__", deny_networkx)
    chart = net.plot()
    # Should still produce a layered, valid spec.
    spec = chart.to_dict()
    assert "layer" in spec


def test_stop_words_excludes_terms_from_vocabulary() -> None:
    """``stop_words`` keeps function-word noise out of the top-N when
    raw-frequency picks pollute the network."""
    # Mostly function words plus a clear content cluster.
    docs = (
        ["the and of to a in the and of to"] * 20
        + ["asylum refugee policy migrant border policy"] * 20
    )
    corpus = _doc_corpus(docs)
    stops = {"the", "and", "of", "to", "a", "in"}
    net = cooccurrence_network(
        corpus, top_n=8, min_count=1, min_cooccur=1, stop_words=stops,
    )
    surfaced = set(net.nodes.index)
    assert surfaced.isdisjoint(stops), (
        f"stop-words leaked into the vocabulary: {surfaced & stops}"
    )
    # Content terms survive.
    assert {"asylum", "refugee", "policy", "migrant"} <= surfaced
