"""Public ``compare()`` facade and the :class:`Comparison` class.

This module defines the public API surface. The analytical methods on
:class:`Comparison` delegate to the keyness / collocation / semantic
subpackages; in this scaffolding release they raise
:class:`NotImplementedError` until Phase 1 lands.
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


KeynessMethod = Literal["log_likelihood", "log_ratio", "bayes_factor", "chi_squared"]
CollocationMeasure = Literal["logDice", "PMI", "t_score", "MI", "MI3"]
EmbeddingAlignment = Literal["procrustes", "anchor", "none"]
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
        min_count: int = 5,
    ) -> KeynessResult:
        """Compute keyness for every shared vocabulary item.

        Phase 1 will implement Dunning log-likelihood with LogRatio
        effect sizes (Hardie 2014) and Bayes factor (Wilson, Gabrielatos)
        as alternatives. Dispersion-poor terms are flagged but not
        filtered.
        """
        raise NotImplementedError("keyness() lands in Phase 1")

    def collocation_shift(
        self,
        target: str,
        window: int = 5,
        measure: CollocationMeasure = "logDice",
        min_count: int = 5,
    ) -> CollocationShiftResult:
        """Compute the change in target-word collocates between a and b.

        Phase 2 will implement window-based co-occurrence with logDice
        (Rychlý 2008) as the default measure and PMI / t-score / MI / MI³
        as alternatives.
        """
        raise NotImplementedError("collocation_shift() lands in Phase 2")

    def semantic_shift(
        self,
        target: str | list[str],
        embedder: Embedder | None = None,
        align: EmbeddingAlignment = "procrustes",
    ) -> SemanticShiftResult:
        """Compute embedding-space displacement of target term(s).

        Phase 6 will implement Procrustes-aligned diachronic embeddings
        following Hamilton et al. (2016), with optional anchor-word
        alignment as an alternative.
        """
        raise NotImplementedError("semantic_shift() lands in Phase 6")

    def concordance(self, target: str, n: int = 20) -> ConcordanceResult:
        """Return KWIC examples of ``target`` from both corpora."""
        raise NotImplementedError("concordance() lands in Phase 3")


def compare(a: CorpusLike, b: CorpusLike) -> Comparison:
    """Construct a pairwise :class:`Comparison` of two corpora or slices."""
    return Comparison(a=a, b=b)


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
