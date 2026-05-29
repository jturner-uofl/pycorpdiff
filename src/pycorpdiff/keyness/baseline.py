"""Keyness against a pre-computed reference-corpus baseline.

Standard keyness compares two corpora the user has on hand. Reference-
corpus keyness compares one corpus against a fixed, externally-defined
*frequency distribution* (the BNC, COCA, a domain reference, a bundled
Gutenberg fiction baseline) — the canonical setup in lexicography and
discourse analysis. The math is identical (Dunning's G² with the same
``formula="rayson"`` / ``"dunning"`` toggle), but the operational
question is different: "what is distinctively *X* about my corpus,
relative to general / domain-typical language?"

The reference side is supplied as a :class:`pycorpdiff.datasets.baselines.Baseline`
— effectively a ``pandas.Series`` of token counts plus a corpus total.
No source documents are needed for the math; this is exactly why
shipping aggregated frequency lists (rather than full reference
corpora) is the right unit of distribution.

Because the reference side has no per-document structure on hand,
analyses that *require* that structure (dispersion, document-level
permutation *p*-values, document-level bootstrap CIs) are unavailable
through :func:`against_baseline`. The function will raise a clear
``ValueError`` if you ask for them. Run a full
:func:`pycorpdiff.compare` against a real :class:`Corpus` if you need
those.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from ..corpus import Corpus, CorpusSlice
from ..datasets.baselines import Baseline, load_baseline
from .loglikelihood import LLFormula

if TYPE_CHECKING:
    from ..results import KeynessResult


def against_baseline(
    corpus: Corpus | CorpusSlice,
    baseline: str | Baseline,
    *,
    formula: LLFormula = "rayson",
    method: Literal[
        "log_likelihood", "chi_squared", "log_ratio", "percent_diff", "bayes_factor"
    ] = "log_likelihood",
    min_count: int = 5,
    effect_size: bool = True,
    multiple_comparisons: Literal["none", "bh", "bonferroni"] = "none",
    stop_words: Iterable[str] | None = None,
) -> KeynessResult:
    """Keyness of ``corpus`` against a fixed reference-corpus baseline.

    Parameters
    ----------
    corpus
        The corpus (or slice) under analysis. Becomes the "A" side
        of the comparison; the baseline is "B".
    baseline
        Either the name of a bundled baseline (e.g.
        ``"gutenberg_fiction"`` — see
        :func:`pycorpdiff.datasets.baselines.list_baselines`) or a
        :class:`Baseline` object produced via
        :func:`pycorpdiff.datasets.baselines.baseline_from_corpus`.
    formula
        Log-likelihood formulation. ``"rayson"`` (default) is the
        2-cell shortcut from Rayson's LL Wizard; ``"dunning"`` is the
        canonical 4-cell G². Same toggle as :meth:`Comparison.keyness`.
    method
        Sort metric for the result table. Same options and meaning as
        :meth:`pycorpdiff.Comparison.keyness`.
    min_count
        Minimum joint frequency across corpus + baseline for a term to
        be included.
    effect_size
        If ``True`` (default), include the ``log_ratio``,
        ``percent_diff``, and ``bayes_factor`` columns.
    multiple_comparisons
        ``"none"`` (default), ``"bh"`` (Benjamini-Hochberg) or
        ``"bonferroni"``.
    stop_words
        Optional iterable of terms to exclude before scoring.

    Returns
    -------
    KeynessResult
        With ``corpus_a=corpus`` and ``corpus_b=None`` (baseline has no
        documents). :meth:`KeynessResult.explain` falls back to KWIC
        lines from ``corpus_a`` only.

    Notes
    -----
    The reference-side counts are not retokenized — the bundled
    baselines were tokenized with the default
    :class:`pycorpdiff.RegexTokenizer` settings (``\\w+``, lowercased,
    NFC-normalised). For meaningful term overlap, ``corpus`` should use
    a compatible tokenization. A radically different tokenizer (e.g.
    a CJK segmenter, a BPE encoder, lemmatised output) will not crash
    but will produce a sparse, hard-to-interpret table because terms
    won't align.

    Examples
    --------
    >>> import pycorpdiff as pcd  # doctest: +SKIP
    >>> corpus = pcd.load_hansard_sample()  # doctest: +SKIP
    >>> result = pcd.against_baseline(corpus, "gutenberg_fiction")  # doctest: +SKIP
    >>> result.table.head()  # doctest: +SKIP
    """
    # Local imports to keep the module-import graph acyclic.
    from ..results import KeynessResult
    from .bayes import bayes_factor as _bayes_factor
    from .chi_squared import chi_squared as _chi_squared
    from .correction import benjamini_hochberg, bonferroni
    from .effect_sizes import log_ratio as _log_ratio
    from .effect_sizes import percent_diff as _percent_diff
    from .loglikelihood import log_likelihood

    baseline_obj = load_baseline(baseline) if isinstance(baseline, str) else baseline

    # Aggregate the corpus side once.  Don't bother with per-document
    # storage — none of the analyses available through this function
    # need it.
    dtm_a = corpus.doc_term_counts(min_count=1)
    if dtm_a.shape[0] == 0:
        raise ValueError("corpus must contain at least one document")
    vocab_a = dtm_a.sum(axis=0).astype("int64")
    n_a = int(vocab_a.sum())
    n_b = int(baseline_obj.total_tokens)
    if n_a == 0:
        raise ValueError(f"corpus has zero tokens; got |corpus|={n_a}")
    if n_b == 0:
        raise ValueError(
            f"baseline {baseline_obj.name!r} has zero tokens; cannot compute keyness"
        )

    vocab_b = baseline_obj.counts.astype("int64")
    all_terms = vocab_a.index.union(vocab_b.index)
    a_aligned = vocab_a.reindex(all_terms, fill_value=0).astype("int64")
    b_aligned = vocab_b.reindex(all_terms, fill_value=0).astype("int64")

    keep = (a_aligned + b_aligned) >= min_count
    if stop_words is not None:
        stop_set = set(stop_words)
        keep &= ~a_aligned.index.isin(stop_set)
    a_kept = a_aligned[keep]
    b_kept = b_aligned[keep]

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

    if multiple_comparisons == "bh":
        table["p_adjusted"] = benjamini_hochberg(table["p_value"].to_numpy())
    elif multiple_comparisons == "bonferroni":
        table["p_adjusted"] = bonferroni(table["p_value"].to_numpy())

    sort_col = {
        "log_likelihood": "g2",
        "log_ratio": "log_ratio",
        "bayes_factor": "bayes_factor",
        "percent_diff": "percent_diff",
        "chi_squared": "chi_squared",
    }[method]
    if sort_col not in table.columns:
        raise ValueError(
            f"method={method!r} requires effect_size=True so the column exists"
        )
    sort_key = table[sort_col].abs()
    table = table.assign(_sort_key=sort_key).sort_values(
        "_sort_key", ascending=False
    ).drop(columns="_sort_key")

    out = table.reset_index().rename(columns={"index": "term"})
    label_a = (
        getattr(corpus, "name", None)
        or getattr(corpus, "label", None)
        or "corpus"
    )
    return KeynessResult(
        table=out,
        method=method,
        n_a=n_a,
        n_b=n_b,
        label_a=str(label_a),
        label_b=f"baseline:{baseline_obj.name}",
        params={
            "formula": formula,
            "effect_size": effect_size,
            "min_count": min_count,
            "multiple_comparisons": multiple_comparisons,
            "stop_words": tuple(stop_words) if stop_words else None,
            "baseline_name": baseline_obj.name,
            "baseline_n_documents": baseline_obj.n_documents,
        },
        corpus_a=corpus,
        corpus_b=None,
    )


__all__ = ["against_baseline"]
