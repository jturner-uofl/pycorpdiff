"""Dispersion plots: *where* in a corpus does a term appear?

A term can have the same overall frequency in two corpora and be
distributed completely differently — even across one corpus, a high-
frequency word can be clustered in a few documents or evenly spread.
The dispersion plot answers "where" by marking each occurrence at the
relevant document index along a horizontal axis.

This is the classic Mosteller / Stubbs / Brezina visualisation for
"how representative is a frequency count" — companion to the
:func:`pycorpdiff.keyness.juilland_d` / ``dispersion_dp`` numerical
measures pycorpdiff already exposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import altair as alt

    from ..corpus import Corpus, CorpusSlice


def dispersion_plot(
    corpus: Corpus | CorpusSlice,
    targets: str | list[str],
    width: int = 600,
    height: int | None = None,
) -> alt.Chart:
    """Visualise *where* each ``target`` appears in the corpus.

    Each occurrence of every target becomes a small vertical tick at
    its document index. Stacked rows show one target at a time, so
    "this term is concentrated in the first third of the corpus" or
    "this term is evenly spread" reads off the plot at a glance.

    Parameters
    ----------
    corpus
        A :class:`Corpus` or :class:`CorpusSlice`.
    targets
        Single target or a list. Each gets its own horizontal row.
    width
        Chart width in pixels.
    height
        Chart height. If ``None``, scales with the number of targets.

    Returns
    -------
    altair.Chart
        Interactive chart with per-occurrence ticks coloured by target.
        Requires the ``[viz]`` extra.

    Example
    -------
    >>> import pycorpdiff as pcd
    >>> corpus = pcd.load_hansard_sample()
    >>> chart = pcd.viz.dispersion_plot(corpus, ['criminal', 'family'])  # doctest: +SKIP
    >>> chart.save('dispersion.svg')                                      # doctest: +SKIP
    """
    import altair as alt

    target_list = [targets] if isinstance(targets, str) else list(targets)
    docs_tokens = corpus.tokens()
    n_docs = len(docs_tokens)

    # Collect (doc_index, target) for every occurrence.
    rows: list[dict[str, Any]] = []
    for doc_idx, tokens in enumerate(docs_tokens):
        for token in tokens:
            if token in target_list:
                rows.append({"doc_index": doc_idx, "target": token})

    if rows:
        points = pd.DataFrame(rows, columns=["doc_index", "target"])
    else:
        # Empty corpus or no matches — emit a typed empty frame so
        # altair can infer column types (it can't infer from empties).
        points = pd.DataFrame(
            {
                "doc_index": pd.Series([], dtype="int64"),
                "target": pd.Series([], dtype="object"),
            }
        )

    if height is None:
        height = max(60, 40 * len(target_list))

    chart = (
        alt.Chart(points)
        .mark_tick(thickness=1.5)
        .encode(
            x=alt.X(
                "doc_index:Q",
                title=None,
                scale=alt.Scale(domain=[0, max(1, n_docs - 1)]),
            ),
            y=alt.Y("target:N", title=None, sort=target_list),
            color=alt.Color("target:N", legend=None, sort=target_list),
            tooltip=[
                alt.Tooltip("doc_index:Q"),
                alt.Tooltip("target:N"),
            ],
        )
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(
                text="Dispersion plot — where each term occurs in the corpus",
                subtitle=f"{n_docs} documents on the x-axis, one tick per occurrence",
            ),
        )
    )
    return chart  # type: ignore[no-any-return]
