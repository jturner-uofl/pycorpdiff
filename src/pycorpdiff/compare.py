"""Public ``compare()`` facade and the :class:`Comparison` class.

This module defines the public API surface. Analytical methods delegate
to the keyness / collocation / semantic subpackages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .corpus import Corpus, CorpusSlice

if TYPE_CHECKING:
    from .results import (
        CollocationShiftResult,
        ConcordanceResult,
        KeynessResult,
        SemanticShiftResult,
    )
    from .semantic.embed import Embedder


KeynessMethod = Literal["log_likelihood", "log_ratio", "bayes_factor", "percent_diff"]
CollocationMeasure = Literal["logDice", "PMI", "t_score", "MI3"]
EmbeddingAlignment = Literal["none", "procrustes"]
MultipleComparisons = Literal["bh", "bonferroni", "none"]
CorpusLike = Corpus | CorpusSlice


@dataclass(frozen=True)
class Comparison:
    """A pairwise comparison of two corpora (or slices).

    Construct via :func:`compare` rather than directly; this keeps the
    surface area small and lets the package add specialised
    constructors (``compare.before_after``, ``compare.over_time``) on
    the function attribute.
    """

    a: CorpusLike
    b: CorpusLike

    def keyness(
        self,
        method: KeynessMethod = "log_likelihood",
        effect_size: bool = True,
        dispersion: bool = False,
        min_count: int = 5,
        multiple_comparisons: MultipleComparisons = "bh",
    ) -> KeynessResult:
        """Compute keyness for every shared-vocabulary item.

        Parameters
        ----------
        method
            Which column to sort the result by. The underlying statistics
            are always computed; this only controls presentation order.
        effect_size
            If True (default), also compute LogRatio (Hardie),
            %DIFF (Gabrielatos), and the BIC-Bayes factor (Wilson).
        dispersion
            If True, compute Juilland's D for both corpora and flag
            terms where ``D < 0.5`` in either — the canonical "this is
            driven by one document" heuristic. Off by default because
            it requires constructing the full doc-term matrices.
        min_count
            Drop terms whose ``count_a + count_b`` is below this
            threshold. Dunning's small-cell unreliability makes the
            default of 5 the standard recommendation.
        multiple_comparisons
            ``"bh"`` (default, Benjamini–Hochberg), ``"bonferroni"``,
            or ``"none"``. The corrected column is named ``p_adjusted``.
        """
        # Imports kept local to break circulars and to keep this module
        # importable without the keyness machinery on hand.
        from .keyness.bayes import bayes_factor as _bayes_factor
        from .keyness.correction import benjamini_hochberg, bonferroni
        from .keyness.dispersion import juilland_d
        from .keyness.effect_sizes import log_ratio as _log_ratio
        from .keyness.effect_sizes import percent_diff as _percent_diff
        from .keyness.loglikelihood import log_likelihood
        from .results import KeynessResult

        dtm_a = self.a.doc_term_counts(min_count=1)
        dtm_b = self.b.doc_term_counts(min_count=1)
        vocab_a = dtm_a.sum(axis=0)
        vocab_b = dtm_b.sum(axis=0)
        n_a = int(vocab_a.sum())
        n_b = int(vocab_b.sum())

        if n_a == 0 or n_b == 0:
            raise ValueError(
                f"both corpora must contain at least one token; got |a|={n_a}, |b|={n_b}"
            )

        all_terms = vocab_a.index.union(vocab_b.index)
        a_aligned = vocab_a.reindex(all_terms, fill_value=0).astype("int64")
        b_aligned = vocab_b.reindex(all_terms, fill_value=0).astype("int64")
        keep = (a_aligned + b_aligned) >= min_count
        a_kept = a_aligned[keep]
        b_kept = b_aligned[keep]

        table = log_likelihood(a_kept, b_kept, n_a, n_b)

        if effect_size:
            table["log_ratio"] = _log_ratio(a_kept, b_kept, n_a, n_b)
            table["percent_diff"] = _percent_diff(a_kept, b_kept, n_a, n_b)
            table["bayes_factor"] = _bayes_factor(a_kept, b_kept, n_a, n_b)

        if dispersion:
            kept_terms = table.index
            disp_a = juilland_d(dtm_a.reindex(columns=kept_terms, fill_value=0))
            disp_b = juilland_d(dtm_b.reindex(columns=kept_terms, fill_value=0))
            table["dispersion_a"] = disp_a
            table["dispersion_b"] = disp_b
            table["dispersion_flag"] = (disp_a < 0.5) | (disp_b < 0.5)

        if multiple_comparisons == "bh":
            table["p_adjusted"] = benjamini_hochberg(table["p_value"].to_numpy())
        elif multiple_comparisons == "bonferroni":
            table["p_adjusted"] = bonferroni(table["p_value"].to_numpy())

        sort_col = {
            "log_likelihood": "g2",
            "log_ratio": "log_ratio",
            "bayes_factor": "bayes_factor",
            "percent_diff": "percent_diff",
        }[method]
        if sort_col not in table.columns:
            # User asked to sort by an effect-size column they disabled.
            raise ValueError(
                f"method={method!r} requires effect_size=True so the column exists"
            )
        # Sort by |signed score| so direction doesn't bury overuse-in-B terms.
        sort_key = table[sort_col].abs()
        table = table.assign(_sort_key=sort_key).sort_values(
            "_sort_key", ascending=False
        ).drop(columns="_sort_key")

        out = table.reset_index().rename(columns={"index": "term"})
        return KeynessResult(
            table=out,
            method=method,
            n_a=n_a,
            n_b=n_b,
            label_a=_corpus_label(self.a),
            label_b=_corpus_label(self.b),
            params={
                "effect_size": effect_size,
                "dispersion": dispersion,
                "min_count": min_count,
                "multiple_comparisons": multiple_comparisons,
            },
            corpus_a=self.a,
            corpus_b=self.b,
        )

    def collocation_shift(
        self,
        target: str,
        window: int = 5,
        measure: CollocationMeasure = "logDice",
        min_count: int = 5,
        smoothing: float = 0.5,
    ) -> CollocationShiftResult:
        """Compute the change in collocates of ``target`` between a and b.

        Window-based co-occurrence with Rychlý logDice (default) or PMI /
        t-score / MI³ as alternatives. Laplace smoothing keeps shifts
        finite for collocates absent on one side.
        """
        from .collocation.shift import collocation_shift as _shift
        from .results import CollocationShiftResult

        table = _shift(
            self.a,
            self.b,
            target=target,
            window=window,
            measure=measure,
            min_count=min_count,
            smoothing=smoothing,
        )
        return CollocationShiftResult(
            target=target,
            table=table.reset_index(),
            measure=measure,
            window=window,
            label_a=_corpus_label(self.a),
            label_b=_corpus_label(self.b),
            corpus_a=self.a,
            corpus_b=self.b,
        )

    def semantic_shift(
        self,
        target: str | list[str],
        embedder: Embedder | None = None,
        window: int = 5,
        align: EmbeddingAlignment = "none",
    ) -> SemanticShiftResult:
        """Compute embedding-space displacement of target term(s).

        Uses *averaged contextual embeddings*: every window around the
        target in each corpus is encoded by ``embedder`` and averaged
        into a corpus-specific centroid. The cosine distance between
        centroids is the reported shift.

        ``embedder`` defaults to :class:`SBERTEmbedder` (requires the
        ``semantic`` extra). For deterministic offline demos pass
        :class:`pycorpdiff.semantic.HashEmbedder`.

        ``align="procrustes"`` is appropriate when the embedder produces
        independent per-corpus spaces (Hamilton-style diachronic
        word2vec). Modern shared-model encoders like SBERT live in a
        common space, so the default is ``"none"``.
        """
        from .results import SemanticShiftResult
        from .semantic.embed import SBERTEmbedder
        from .semantic.shift import semantic_shift as _shift

        effective_embedder = embedder if embedder is not None else SBERTEmbedder()
        table = _shift(
            self.a, self.b, target=target, embedder=effective_embedder,
            window=window, align=align,
        )
        targets = [target] if isinstance(target, str) else list(target)
        return SemanticShiftResult(
            targets=targets,
            table=table,
            alignment=align,
            label_a=_corpus_label(self.a),
            label_b=_corpus_label(self.b),
            corpus_a=self.a,
            corpus_b=self.b,
            embedder=effective_embedder,
            window=window,
        )

    def concordance(
        self, target: str, n: int = 20, window: int = 5
    ) -> ConcordanceResult:
        """Return side-by-side KWIC examples of ``target`` from both corpora.

        Up to ``n`` lines per corpus are returned, concatenated into a
        single :class:`ConcordanceResult` with a ``corpus`` column
        distinguishing the source. Shortcut for
        ``pycorpdiff.explain.kwic_compare(a, b, target, ...)``.
        """
        from .explain import kwic_compare

        return kwic_compare(
            self.a,
            self.b,
            target=target,
            window=window,
            n_per_side=n,
            label_a=_corpus_label(self.a),
            label_b=_corpus_label(self.b),
        )


def compare(a: CorpusLike, b: CorpusLike) -> Comparison:
    """Construct a pairwise :class:`Comparison` of two corpora or slices."""
    return Comparison(a=a, b=b)


def _corpus_label(c: CorpusLike) -> str:
    return c.label if isinstance(c, CorpusSlice) else "corpus"


def _before_after(
    corpus: Corpus,
    event_date: str,
    time_col: str = "date",
) -> Comparison:
    """Construct a before/after Comparison split on ``event_date``.

    The before-slice contains documents with ``time_col < event_date``;
    the after-slice contains documents with ``time_col >= event_date``.
    """
    import pandas as pd

    if time_col not in corpus.docs.columns:
        raise KeyError(f"time_col={time_col!r} not found in corpus columns")
    event = pd.Timestamp(event_date)
    times = pd.to_datetime(corpus.docs[time_col])
    before = CorpusSlice(parent=corpus, mask=times < event, filters={"before": event_date})
    after = CorpusSlice(parent=corpus, mask=times >= event, filters={"after": event_date})
    return Comparison(a=before, b=after)


# Expose the specialised constructor as an attribute of the public ``compare``
# function so users can write ``pcd.compare.before_after(...)`` — matches the
# API shape promised in the README.
compare.before_after = _before_after  # type: ignore[attr-defined]
