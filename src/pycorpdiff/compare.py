"""Public ``compare()`` facade and the :class:`Comparison` class.

This module defines the public API surface. Analytical methods delegate
to the keyness / collocation / semantic subpackages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .corpus import Corpus, CorpusSlice
from .keyness.loglikelihood import LLFormula

if TYPE_CHECKING:
    from .results import (
        CollocationShiftResult,
        ConcordanceResult,
        KeynessResult,
        SemanticShiftResult,
    )
    from .semantic.embed import Embedder


KeynessMethod = Literal[
    "log_likelihood", "log_ratio", "bayes_factor", "percent_diff", "chi_squared",
]
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
        formula: LLFormula = "rayson",
        effect_size: bool = True,
        dispersion: bool = False,
        min_count: int = 5,
        multiple_comparisons: MultipleComparisons = "bh",
        stop_words: set[str] | list[str] | None = None,
        permutation_n: int = 0,
        permutation_seed: int | None = None,
        ci: Literal["none", "bootstrap"] = "none",
        n_boot: int = 999,
        ci_level: float = 0.95,
        simultaneous_ci: bool = False,
        cluster_col: str | None = None,
        bootstrap_seed: int | None = None,
    ) -> KeynessResult:
        """Compute keyness for every shared-vocabulary item.

        Parameters
        ----------
        method
            Which statistic to sort the result by. ``"log_likelihood"``
            (default) sorts by signed Dunning G²; ``"chi_squared"``
            sorts by signed Pearson χ². The other modes
            (``"log_ratio"``, ``"bayes_factor"``, ``"percent_diff"``)
            require ``effect_size=True`` and sort by that column.
        formula
            Which log-likelihood formulation to use for the G² column.
            ``"rayson"`` (default) is the 2-cell shortcut matching
            Rayson's UCREL LL Wizard; ``"dunning"`` is the full 4-cell
            G² matching NLTK's ``BigramAssocMeasures`` and R's
            ``quanteda::textstat_keyness(measure="lr")``. See
            ``docs/statistical-methods.md`` for the math + when they
            diverge.
        effect_size
            If True (default), also compute LogRatio (Hardie),
            %DIFF (Gabrielatos), and the BIC-approximated Bayes factor.
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
        stop_words
            Iterable of terms to exclude before scoring. Useful for
            filtering function-word noise without modifying the source
            corpus. Tokens drop *after* vocabulary union, so the corpus
            totals (used as normalisation denominators) are unaffected.
        permutation_n
            If positive, also compute an empirical permutation *p*-value
            for every retained term and emit it as the ``p_permutation``
            column. Documents are the unit of exchangeability. Useful
            when the asymptotic χ² approximation is suspect (small
            expected counts, very small corpora). ``999`` is the
            conventional value; cost scales linearly. Disabled by
            default — this is the expensive opt-in.
        permutation_seed
            Optional RNG seed for reproducible permutation *p*-values.
        ci
            ``"none"`` (default) or ``"bootstrap"``. When ``"bootstrap"``,
            adds ``g2_ci_lower`` and ``g2_ci_upper`` columns to the
            result table via document-level resampling. CIs that
            straddle zero indicate uncertainty about the *direction* of
            overuse, not just the magnitude — a more honest signal
            than a sub-threshold *p*-value alone. Cost scales linearly
            with ``n_boot``; budget ~30 s per 999 iterations on a
            10k-doc / 5k-vocab corpus.
        n_boot
            Number of bootstrap iterations when ``ci="bootstrap"``.
            999 is the convention.
        ci_level
            Confidence level in (0, 1) when ``ci="bootstrap"``.
            Default 0.95.
        simultaneous_ci
            When ``ci="bootstrap"``: if ``False`` (default), per-term
            percentile CIs (calibrated for any single fixed term);
            if ``True``, Westfall-Young studentized-max simultaneous
            CIs with family-wise (1 - α) coverage across the entire
            vocabulary. Use ``simultaneous_ci=True`` when reporting
            CIs on top-ranked terms (post-selection inference);
            otherwise per-term CIs are anti-conservative on the
            top of a sorted keyness table. See
            ``docs/statistical-methods.md`` § "Simultaneous CIs"
            for the derivation.
        cluster_col
            When ``ci="bootstrap"``: names a metadata column (e.g.
            ``"speaker"`` / ``"member"``) defining clusters of
            non-independent documents. The bootstrap then resamples
            *clusters* (blocks of documents by the same speaker), not
            individual documents. Correct when speeches are nested in
            speakers; IID document resampling understates the CI width
            on hierarchical corpora. ``None`` (default) keeps IID
            resampling.
        bootstrap_seed
            Optional RNG seed for reproducible bootstrap CIs.
        """
        # Imports kept local to break circulars and to keep this module
        # importable without the keyness machinery on hand.
        from .keyness.bayes import bayes_factor as _bayes_factor
        from .keyness.chi_squared import chi_squared as _chi_squared
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
        if stop_words is not None:
            stop_set = set(stop_words)
            keep &= ~a_aligned.index.isin(stop_set)
        a_kept = a_aligned[keep]
        b_kept = b_aligned[keep]

        # G² is always computed (cheap, the default sort column). χ² is
        # computed only when requested — same shape, asymptotically
        # equivalent, no need to pay for both by default.
        table = log_likelihood(a_kept, b_kept, n_a, n_b, formula=formula)
        if method == "chi_squared":
            chi_table = _chi_squared(a_kept, b_kept, n_a, n_b)
            table["chi_squared"] = chi_table["chi_squared"]

        if effect_size:
            table["log_ratio"] = _log_ratio(a_kept, b_kept, n_a, n_b)
            table["percent_diff"] = _percent_diff(a_kept, b_kept, n_a, n_b)
            table["bayes_factor"] = _bayes_factor(
                a_kept, b_kept, n_a, n_b, formula=formula
            )

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

        if permutation_n > 0:
            from .keyness.permutation import permutation_pvalues as _perm

            p_perm = _perm(
                self.a, self.b,
                terms=table.index,
                n_permutations=permutation_n,
                seed=permutation_seed,
            )
            table["p_permutation"] = p_perm.reindex(table.index)

        if ci == "bootstrap":
            from .keyness.bootstrap import bootstrap_g2_ci as _boot

            ci_table = _boot(
                self.a, self.b,
                terms=table.index,
                formula=formula,
                n_boot=n_boot,
                ci_level=ci_level,
                simultaneous=simultaneous_ci,
                cluster_col=cluster_col,
                seed=bootstrap_seed,
            )
            table["g2_ci_lower"] = ci_table["g2_ci_lower"].reindex(table.index)
            table["g2_ci_upper"] = ci_table["g2_ci_upper"].reindex(table.index)
        elif ci != "none":
            raise ValueError(
                f"ci must be 'none' or 'bootstrap'; got {ci!r}"
            )

        sort_col = {
            "log_likelihood": "g2",
            "log_ratio": "log_ratio",
            "bayes_factor": "bayes_factor",
            "percent_diff": "percent_diff",
            "chi_squared": "chi_squared",
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
                "formula": formula,
                "effect_size": effect_size,
                "dispersion": dispersion,
                "min_count": min_count,
                "multiple_comparisons": multiple_comparisons,
                "stop_words": tuple(stop_words) if stop_words else None,
                "permutation_n": permutation_n,
                "permutation_seed": permutation_seed,
                "ci": ci,
                "n_boot": n_boot if ci == "bootstrap" else None,
                "ci_level": ci_level if ci == "bootstrap" else None,
                "simultaneous_ci": simultaneous_ci if ci == "bootstrap" else None,
                "cluster_col": cluster_col if ci == "bootstrap" else None,
                "bootstrap_seed": bootstrap_seed if ci == "bootstrap" else None,
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
