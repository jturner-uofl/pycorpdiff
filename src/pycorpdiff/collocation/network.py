"""Term co-occurrence networks.

For exploratory work — "what does the discourse *graph* look like?" —
the natural artefact is a network: nodes are the corpus's most frequent
terms, edges connect terms that co-occur within a window, and edge
weights come from a standard association measure (PMI, t-score, MI³).

This is the term-as-vertex visualisation that gephi-style network tools
have made common in digital humanities; here it lands as a first-class
:class:`pycorpdiff.collocation.NetworkResult` with the same
:meth:`to_df` / :meth:`plot` shape as every other Result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ..corpus import Corpus, CorpusSlice
from ..results import _table_to_html, _table_to_json
from .measures import logdice, mi_three, pmi, t_score

if TYPE_CHECKING:
    import altair as alt

NetworkMeasure = Literal["PMI", "t_score", "MI3", "logDice"]


@dataclass(frozen=True)
class NetworkResult:
    """A term co-occurrence network with nodes, edges, and a plot method.

    The two DataFrames are the canonical "long-format" shape every
    network analytics tool consumes:

    - ``nodes``: index = term, columns = ``count``, ``degree``
    - ``edges``: columns = ``source``, ``target``, ``cooccur_count``,
      ``weight``  (the association score), and ``rank`` (0-based, by
      ``|weight|`` descending).
    """

    nodes: pd.DataFrame
    edges: pd.DataFrame
    measure: NetworkMeasure
    window: int
    label: str = ""
    params: dict[str, object] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        """Return the edges as a flat tidy DataFrame (for round-trips)."""
        return self.edges.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the edge table as HTML (returns the string and,
        optionally, writes to ``path``). Extra kwargs forward to
        :meth:`pandas.DataFrame.to_html`."""
        return _table_to_html(self.edges, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the edge table as JSON (default ``orient="records"``).
        Returns the JSON string and, optionally, writes to ``path``."""
        return _table_to_json(self.edges, path, **kw)

    def summary(self) -> str:
        return (
            f"NetworkResult(measure={self.measure}, window={self.window}, "
            f"nodes={len(self.nodes):,}, edges={len(self.edges):,})"
        )

    def plot(self, **kw: object) -> alt.Chart:
        """Render the network as an altair force-directed-style plot."""
        from ..viz.network import network_plot

        return network_plot(self, **kw)  # type: ignore[arg-type]


def cooccurrence_network(
    corpus: Corpus | CorpusSlice,
    *,
    top_n: int = 50,
    window: int = 5,
    measure: NetworkMeasure = "PMI",
    min_count: int = 3,
    min_cooccur: int = 2,
    smoothing: float = 0.5,
    stop_words: Iterable[str] | None = None,
) -> NetworkResult:
    """Build a term co-occurrence network for the ``top_n`` terms.

    Each pair of distinct terms among the ``top_n`` vocabulary is
    weighted by the chosen association measure on their joint counts
    within ``window`` tokens of each other inside a document.

    Parameters
    ----------
    corpus
        A :class:`Corpus` or :class:`CorpusSlice`.
    top_n
        Vocabulary cap — the ``top_n`` most frequent terms (after
        ``min_count``) become network nodes.
    window
        Symmetric context window for the co-occurrence count.
    measure
        Edge-weight association measure.
    min_count
        Drop terms below this corpus-wide frequency before picking the
        top-N.
    min_cooccur
        Drop edges with joint count below this. Acts as the network's
        noise floor.
    smoothing
        Laplace constant added to joint and marginal counts before
        scoring (mirrors :func:`collocation_shift`'s convention so the
        same measures stay finite on absent pairs).
    stop_words
        Optional iterable of terms to exclude from the vocabulary
        before the ``top_n`` cut. Useful when the raw top-of-frequency
        is dominated by function words (``the``, ``and``, ``of``,
        …) that aren't analytically interesting in the discourse
        graph. Without this, an English corpus's top-30 by frequency
        is almost entirely closed-class function words.

    Returns
    -------
    NetworkResult
    """
    if top_n < 2:
        raise ValueError(f"top_n must be >= 2; got {top_n}")
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    if smoothing <= 0:
        raise ValueError(f"smoothing must be > 0; got {smoothing}")

    vocab = corpus.vocab(min_count=min_count)
    if stop_words is not None:
        stop_set = set(stop_words)
        vocab = vocab[~vocab.index.isin(stop_set)]
    vocab = vocab.head(top_n)
    if len(vocab) < 2:
        raise ValueError(
            f"need at least 2 terms after min_count={min_count} + "
            f"stop_words filter; got {len(vocab)}"
        )

    keep_set = set(vocab.index)
    pair_counts: Counter[tuple[str, str]] = Counter()

    for tokens in corpus.tokens():
        # Pre-filter to in-vocab tokens with original positions.
        positions = [(i, t) for i, t in enumerate(tokens) if t in keep_set]
        for k, (i, t_i) in enumerate(positions):
            for j, t_j in positions[k + 1 :]:
                if j - i > window:
                    break  # positions are sorted; rest are further away
                if t_i == t_j:
                    continue
                pair = (t_i, t_j) if t_i < t_j else (t_j, t_i)
                pair_counts[pair] += 1

    if not pair_counts:
        return NetworkResult(
            nodes=vocab.rename("count").to_frame().assign(degree=0),
            edges=pd.DataFrame(
                columns=["source", "target", "cooccur_count", "weight", "rank"]
            ),
            measure=measure,
            window=window,
            label=_corpus_label(corpus),
            params={"top_n": top_n, "min_count": min_count, "min_cooccur": min_cooccur},
        )

    n_total = corpus.total_tokens()
    rows = []
    for (src, tgt), joint in pair_counts.items():
        if joint < min_cooccur:
            continue
        rows.append(
            {
                "source": src,
                "target": tgt,
                "cooccur_count": joint,
                "f_a": int(vocab[src]),
                "f_b": int(vocab[tgt]),
            }
        )
    if not rows:
        return NetworkResult(
            nodes=vocab.rename("count").to_frame().assign(degree=0),
            edges=pd.DataFrame(
                columns=["source", "target", "cooccur_count", "weight", "rank"]
            ),
            measure=measure,
            window=window,
            label=_corpus_label(corpus),
            params={"top_n": top_n, "min_count": min_count, "min_cooccur": min_cooccur},
        )

    edges = pd.DataFrame(rows)
    f_xy_arr = edges["cooccur_count"].to_numpy(dtype=float) + smoothing
    f_a_arr = edges["f_a"].to_numpy(dtype=float) + smoothing
    f_b_arr = edges["f_b"].to_numpy(dtype=float) + smoothing

    if measure == "PMI":
        weight_arr = np.log2((f_xy_arr * n_total) / (f_a_arr * f_b_arr))
    elif measure == "t_score":
        expected_arr = (f_a_arr * f_b_arr) / n_total
        weight_arr = (f_xy_arr - expected_arr) / np.sqrt(f_xy_arr)
    elif measure == "MI3":
        weight_arr = np.log2((np.power(f_xy_arr, 3) * n_total) / (f_a_arr * f_b_arr))
    elif measure == "logDice":
        weight_arr = 14.0 + np.log2((2.0 * f_xy_arr) / (f_a_arr + f_b_arr))
    else:
        raise ValueError(f"unknown measure={measure!r}")

    # Silence the lint warnings on unused-but-validated helper imports.
    _ = (pmi, t_score, mi_three, logdice)

    edges = edges.drop(columns=["f_a", "f_b"]).assign(weight=weight_arr)
    edges = edges.sort_values("weight", ascending=False, key=lambda s: s.abs())
    edges = edges.reset_index(drop=True).assign(rank=lambda d: d.index.astype(int))

    # Degrees (undirected): how many edges touch each node?
    degree_a = edges.groupby("source").size()
    degree_b = edges.groupby("target").size()
    degrees = degree_a.add(degree_b, fill_value=0).astype(int)

    nodes = vocab.rename("count").to_frame()
    nodes["degree"] = degrees.reindex(nodes.index, fill_value=0).astype(int)

    return NetworkResult(
        nodes=nodes,
        edges=edges,
        measure=measure,
        window=window,
        label=_corpus_label(corpus),
        params={
            "top_n": top_n,
            "min_count": min_count,
            "min_cooccur": min_cooccur,
            "smoothing": smoothing,
        },
    )


def _corpus_label(c: Corpus | CorpusSlice) -> str:
    if isinstance(c, CorpusSlice):
        return c.label
    return "corpus"


# Public type alias for users to depend on if they want.
__all__: Sequence[str] = ["NetworkResult", "cooccurrence_network", "NetworkMeasure"]
