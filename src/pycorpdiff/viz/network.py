"""Network plot for term co-occurrence graphs.

Renders a :class:`pycorpdiff.collocation.NetworkResult` as an altair
chart with circles for nodes, rules for edges, and text labels. Node
positions come from a spring-force layout if ``networkx`` is
installed; otherwise the nodes fall back to a circular layout, which
still gives a structurally faithful (if visually flatter) picture.

altair is intentionally pinned at the ``[viz]`` extra to avoid pulling
heavyweight rendering deps into the base install.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import altair as alt

    from ..collocation.network import NetworkResult


def network_plot(
    result: NetworkResult,
    *,
    width: int = 700,
    height: int = 700,
    max_edges: int = 100,
    label_top_n: int = 30,
    seed: int = 0,
) -> alt.Chart:
    """Plot a :class:`NetworkResult` as a force-directed-style network.

    Parameters
    ----------
    result
        The network to render.
    width, height
        Canvas dimensions in pixels. Square by default.
    max_edges
        Only the top ``max_edges`` edges by ``|weight|`` are drawn,
        preventing dense networks from rendering as a black blob.
    label_top_n
        Inline-label budget. Only the ``label_top_n`` highest-degree
        nodes get text labels next to their dot; the others remain
        bare circles with hover tooltips.
    seed
        Random seed for the spring layout's starting configuration.
    """
    import altair as alt

    nodes = result.nodes.copy()
    edges = result.edges.head(max_edges).copy()

    positions = _layout(nodes, edges, seed=seed)
    nodes_xy = nodes.join(positions, how="left").reset_index(names="term")

    # Edge endpoints — attach source / target coordinates.
    edges_xy = edges.merge(
        positions.rename(columns={"x": "x_src", "y": "y_src"}),
        left_on="source",
        right_index=True,
    ).merge(
        positions.rename(columns={"x": "x_tgt", "y": "y_tgt"}),
        left_on="target",
        right_index=True,
    )

    edge_layer = (
        alt.Chart(edges_xy)
        .mark_rule(opacity=0.35, color="#777")
        .encode(
            x=alt.X("x_src:Q", axis=None, scale=alt.Scale(domain=[-1.1, 1.1])),
            y=alt.Y("y_src:Q", axis=None, scale=alt.Scale(domain=[-1.1, 1.1])),
            x2="x_tgt:Q",
            y2="y_tgt:Q",
            strokeWidth=alt.Size(
                "weight:Q",
                scale=alt.Scale(range=[0.5, 4]),
                legend=alt.Legend(title=f"Edge weight ({result.measure})"),
            ),
            tooltip=["source:N", "target:N", "cooccur_count:Q", "weight:Q"],
        )
    )

    node_layer = (
        alt.Chart(nodes_xy)
        .mark_circle(opacity=0.85, color="#1f77b4")
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.1, 1.1])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.1, 1.1])),
            size=alt.Size(
                "count:Q",
                scale=alt.Scale(range=[80, 600]),
                legend=alt.Legend(title="Term frequency"),
            ),
            tooltip=["term:N", "count:Q", "degree:Q"],
        )
    )

    labelled = nodes_xy.sort_values("degree", ascending=False).head(label_top_n)
    label_layer = (
        alt.Chart(labelled)
        .mark_text(dy=-10, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.1, 1.1])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.1, 1.1])),
            text="term:N",
        )
    )

    chart = (
        (edge_layer + node_layer + label_layer)
        .properties(width=width, height=height)
        .configure_view(strokeWidth=0)
        .interactive()
    )
    return chart  # type: ignore[no-any-return]


def _layout(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute ``(x, y)`` for every node.

    Uses ``networkx.spring_layout`` if available; otherwise falls back
    to a circular layout (still a valid plot, just less informative).
    Returns a DataFrame indexed by term with ``x`` and ``y`` columns
    rescaled to ``[-1, 1]``.
    """
    try:
        import networkx as nx
    except ImportError:
        return _circular_layout(nodes.index.tolist())

    g = nx.Graph()
    for term in nodes.index:
        g.add_node(term)
    for _, row in edges.iterrows():
        g.add_edge(row["source"], row["target"], weight=abs(float(row["weight"])))

    # Kamada-Kawai produces more uniformly-spaced layouts than the
    # default spring algorithm on densely-connected graphs, which is
    # the common case for corpus discourse networks (every top term
    # tends to co-occur with many others). Falls back to a high-k
    # spring layout for disconnected graphs (KK requires connectivity).
    n = max(1, g.number_of_nodes())
    try:
        if nx.is_connected(g):
            pos = nx.kamada_kawai_layout(g)
        else:
            pos = nx.spring_layout(
                g, seed=seed, k=2.5 / math.sqrt(n), iterations=200
            )
    except (nx.NetworkXError, ValueError):
        pos = nx.spring_layout(
            g, seed=seed, k=2.5 / math.sqrt(n), iterations=200
        )
    coords = pd.DataFrame(
        {"x": [pos[t][0] for t in nodes.index], "y": [pos[t][1] for t in nodes.index]},
        index=nodes.index,
    )
    # Rescale to a square in [-1, 1].
    for col in ("x", "y"):
        lo, hi = coords[col].min(), coords[col].max()
        span = hi - lo if hi != lo else 1.0
        coords[col] = 2.0 * (coords[col] - lo) / span - 1.0
    return coords


def _circular_layout(terms: list[str]) -> pd.DataFrame:
    """Fallback layout when ``networkx`` isn't installed."""
    n = len(terms)
    coords = {
        t: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i, t in enumerate(terms)
    }
    return pd.DataFrame(
        {"x": [coords[t][0] for t in terms], "y": [coords[t][1] for t in terms]},
        index=terms,
    )
