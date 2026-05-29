"""Tests for ``pycorpdiff.viz.scattertext_plot``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd

pytest.importorskip("altair")

from pycorpdiff.viz.scattertext import _percentile_rank, scattertext_plot  # noqa: E402


def _keyness_df() -> pd.DataFrame:
    """A keyness-shaped fixture with both A-leaning and B-leaning terms."""
    a = pd.DataFrame(
        {
            "text": (
                ["healthcare reform"] * 8
                + ["jobs growth economy"] * 4
                + ["border security"] * 2
            )
        }
    )
    b = pd.DataFrame(
        {
            "text": (
                ["border security wall"] * 8
                + ["jobs growth economy"] * 4
                + ["healthcare reform"] * 2
            )
        }
    )
    ca = pcd.from_dataframe(a, text_col="text")
    cb = pcd.from_dataframe(b, text_col="text")
    return pcd.compare(ca, cb).keyness().to_df()


def test_returns_altair_chart() -> None:
    import altair as alt

    chart = scattertext_plot(_keyness_df())
    assert isinstance(chart, alt.LayerChart) or hasattr(chart, "to_dict")


def test_axes_are_rank_percentiles() -> None:
    spec = scattertext_plot(_keyness_df()).to_dict()
    # The chart is a layered spec; the layers carry the encodings.
    layers = spec["layer"]
    # The points/labels layers use percentile_a / percentile_b.
    encoded_xs = {layer.get("encoding", {}).get("x", {}).get("field") for layer in layers}
    encoded_ys = {layer.get("encoding", {}).get("y", {}).get("field") for layer in layers}
    assert "percentile_a" in encoded_xs
    assert "percentile_b" in encoded_ys


def test_axis_domain_is_zero_to_one() -> None:
    """rank-percentile axes should always span [0, 1]."""
    spec = scattertext_plot(_keyness_df()).to_dict()
    # Pull the first layer whose x has a scale.
    for layer in spec["layer"]:
        x_enc = layer.get("encoding", {}).get("x", {})
        if "scale" in x_enc:
            assert x_enc["scale"]["domain"] == [0, 1]
            return
    raise AssertionError("no x-axis scale found in any layer")


def test_label_param_threads_into_axis_titles() -> None:
    spec = scattertext_plot(_keyness_df(), label_a="trump", label_b="biden").to_dict()
    titles = []
    for layer in spec["layer"]:
        x_enc = layer.get("encoding", {}).get("x", {})
        y_enc = layer.get("encoding", {}).get("y", {})
        titles.append(x_enc.get("title", ""))
        titles.append(y_enc.get("title", ""))
    joined = " ".join(t for t in titles if isinstance(t, str))
    assert "trump" in joined
    assert "biden" in joined


def test_tooltip_includes_counts_and_effect() -> None:
    spec = scattertext_plot(_keyness_df()).to_dict()
    # Pull tooltips from any layer that has them.
    tooltip_fields: set[str] = set()
    for layer in spec["layer"]:
        for t in layer.get("encoding", {}).get("tooltip", []) or []:
            if isinstance(t, dict):
                tooltip_fields.add(t.get("field", ""))
    assert {"term", "count_a", "count_b", "g2"}.issubset(tooltip_fields)


def test_includes_diagonal_reference_line() -> None:
    """The chart should layer in a dashed x=y diagonal."""
    spec = scattertext_plot(_keyness_df()).to_dict()
    has_dashed_line = False
    for layer in spec["layer"]:
        mark = layer.get("mark", {})
        if (
            isinstance(mark, dict)
            and mark.get("type") == "line"
            and mark.get("strokeDash") == [4, 4]
        ):
            has_dashed_line = True
            break
    assert has_dashed_line


def test_handles_empty_dataframe() -> None:
    """An empty keyness table should produce a valid (empty) chart, not raise."""
    empty = pd.DataFrame(
        columns=["term", "count_a", "count_b", "g2", "p_value"]
    )
    chart = scattertext_plot(empty)
    spec = chart.to_dict()
    # Bare-points fallback for empty input — should still be a valid spec.
    assert "mark" in spec
    assert spec["datasets"][next(iter(spec["datasets"]))] == []


def test_n_labels_caps_text_layer_size() -> None:
    """The text layer should hold at most 2 * n_labels rows."""
    df = _keyness_df()
    n_labels = 3
    spec = scattertext_plot(df, n_labels=n_labels).to_dict()
    text_layer = None
    for layer in spec["layer"]:
        mark = layer.get("mark", {})
        if isinstance(mark, dict) and mark.get("type") == "text":
            text_layer = layer
            break
    assert text_layer is not None
    # Inline data lives in datasets / data.values; the text layer references
    # the dataset by name.
    data_key = (text_layer.get("data") or {}).get("name")
    assert data_key is not None
    data = spec["datasets"][data_key]
    assert len(data) <= 2 * n_labels


def test_chart_is_interactive() -> None:
    """`.interactive()` was called — selection params should be present."""
    spec = scattertext_plot(_keyness_df()).to_dict()
    # `.interactive()` injects an interval-binding params block at the top.
    assert "params" in spec or any(
        "params" in layer for layer in spec.get("layer", [])
    )


def test_keyness_result_plot_scattertext_routes_here() -> None:
    """KeynessResult.plot(kind='scattertext') should hit this module."""
    a = pd.DataFrame({"text": ["foo bar"] * 5 + ["foo baz"] * 1})
    b = pd.DataFrame({"text": ["foo baz"] * 5 + ["foo bar"] * 1})
    ca = pcd.from_dataframe(a, text_col="text")
    cb = pcd.from_dataframe(b, text_col="text")
    res = pcd.compare(ca, cb).keyness()
    chart = res.plot(kind="scattertext")
    # Same diagonal-line invariant as the unit chart.
    spec = chart.to_dict()
    has_dashed_line = False
    for layer in spec["layer"]:
        mark = layer.get("mark", {})
        if isinstance(mark, dict) and mark.get("strokeDash") == [4, 4]:
            has_dashed_line = True
            break
    assert has_dashed_line


def test_keyness_result_plot_threads_corpus_labels() -> None:
    """KeynessResult should pass its label_a / label_b through to the plot."""
    a = pd.DataFrame({"text": ["foo bar"] * 5, "outlet": ["DEM"] * 5})
    b = pd.DataFrame({"text": ["foo baz"] * 5, "outlet": ["GOP"] * 5})
    ca = pcd.from_dataframe(a, text_col="text", meta_cols=("outlet",)).slice(
        outlet="DEM"
    )
    cb = pcd.from_dataframe(b, text_col="text", meta_cols=("outlet",)).slice(
        outlet="GOP"
    )
    res = pcd.compare(ca, cb).keyness()
    spec = res.plot(kind="scattertext").to_dict()
    titles: list[str] = []
    for layer in spec["layer"]:
        x_enc = layer.get("encoding", {}).get("x", {})
        y_enc = layer.get("encoding", {}).get("y", {})
        titles.append(str(x_enc.get("title", "")))
        titles.append(str(y_enc.get("title", "")))
    joined = " ".join(titles)
    assert "DEM" in joined
    assert "GOP" in joined


def test_unknown_kind_raises_with_scattertext_in_message() -> None:
    a = pd.DataFrame({"text": ["foo"]})
    b = pd.DataFrame({"text": ["bar"]})
    ca = pcd.from_dataframe(a, text_col="text")
    cb = pcd.from_dataframe(b, text_col="text")
    res = pcd.compare(ca, cb).keyness()
    with pytest.raises(ValueError, match="scattertext"):
        res.plot(kind="bogus")


def test_percentile_rank_helper_is_monotonic() -> None:
    """Internal helper: percentile rank rises monotonically with raw count."""
    s = pd.Series([5, 1, 10, 3])
    pct = _percentile_rank(s)
    # The largest value (10) should have the largest percentile.
    assert pct.argmax() == 2
    # The smallest (1) should have the smallest.
    assert pct.argmin() == 1


def test_is_exported_at_module_level() -> None:
    """`from pycorpdiff.viz import scattertext_plot` should work."""
    from pycorpdiff.viz import scattertext_plot as imported

    assert imported is scattertext_plot
