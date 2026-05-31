"""Bootstrap confidence intervals for keyness G².

Most corpus-linguistic tools (quanteda, AntConc, Sketch Engine) report
keyness G² as a point estimate. But G² is computed from random samples
— the documents in each corpus are a sample from some larger
population of speech / writing — so the point estimate has its own
uncertainty. A bootstrap CI on G² quantifies this directly: resample
documents from each corpus *with replacement* preserving the original
counts, recompute G², repeat ``n_boot`` times. The empirical α/2 and
1 - α/2 quantiles across iterations give the CI.

Documents (rather than tokens) are the unit of resampling because
they're the unit of exchangeability for the population-level
inference users actually want — "if we re-collected the corpus from
the same source, what range of G² values would we see?"

**Per-term vs simultaneous CIs.** The default percentile CI is
*per-term* — calibrated for any individual term named in advance.
A user who reads the *top-ranked* term's CI from a ranked keyness
table is implicitly doing post-selection inference: the term was
chosen because it had the largest |G²| in the observed sample, so
its bootstrap CI is anti-conservative under a known null (Type-I
inflation on the order of 30 percentage points in our case study).
The ``simultaneous=True`` option computes Westfall-Young
studentized-max CIs that have family-wise (1 − α) coverage across
the whole vocabulary — slightly wider than per-term CIs, but valid
to report on the top-ranked terms.

**Interpreting the result.** CIs that straddle zero (``g2_ci_lower`` < 0
< ``g2_ci_upper``) indicate uncertainty about the *direction* of
overuse, not just the magnitude. That's a more honest signal than
a sub-threshold *p*-value alone.

References
----------
Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*.
Chapman & Hall. (The percentile-CI construction; § 13.3.)

Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple
Testing: Examples and Methods for p-Value Adjustment*. Wiley.
(The max-T / studentized-max procedure for simultaneous inference.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import xlogy

from ..corpus import Corpus, CorpusSlice
from .loglikelihood import LLFormula


def bootstrap_g2_ci(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    *,
    terms: pd.Index | list[str] | None = None,
    formula: LLFormula = "rayson",
    n_boot: int = 999,
    ci_level: float = 0.95,
    simultaneous: bool = False,
    cluster_col: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Bootstrap CIs on signed G² for every shared term.

    Parameters
    ----------
    a, b
        The two corpora (or slices) under comparison.
    terms
        Optional restriction to a subset of vocabulary terms. If
        ``None`` (default), every term in the union of vocabularies
        is scored.
    formula
        Log-likelihood formulation matching the :func:`log_likelihood`
        choice. ``"rayson"`` (2-cell shortcut, default) or
        ``"dunning"`` (full 4-cell G²). The bootstrap distribution
        adopts whichever formula the point estimate used so the CI is
        self-consistent.
    n_boot
        Number of bootstrap iterations. 999 is the convention.
    ci_level
        Confidence level in (0, 1). Default 0.95.
    simultaneous
        If ``False`` (default), per-term percentile CIs: calibrated
        for any single term specified in advance, but anti-conservative
        when applied to post-hoc-selected terms (e.g., the top-ranked
        row of a sorted keyness table). If ``True``, Westfall-Young
        studentized-max CIs with family-wise (1 - α) coverage across
        the entire vocabulary — slightly wider per-term, but valid
        to report on top-ranked terms.
    cluster_col
        If given, names a metadata column on ``a`` and ``b`` (e.g.
        ``"speaker"`` / ``"member"``) defining clusters of
        non-independent documents. The bootstrap then resamples
        *clusters* with replacement (taking every document in a
        sampled cluster as a block) rather than individual documents.
        This is the correct unit of resampling when speeches are
        nested in speakers nested in parties: IID document resampling
        treats every speech as independent and **understates** the CI
        width (the effective sample size is closer to the number of
        speakers than the number of speeches). Cluster-robust CIs are
        wider and more honest for hierarchical corpora. When ``None``
        (default), documents are resampled IID as before.
    seed
        Optional RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Indexed by term. Always contains ``g2_ci_lower`` and
        ``g2_ci_upper`` — the empirical α/2 and 1 - α/2 quantiles of
        each term's signed-G² bootstrap distribution (per-term
        percentile CI). When ``simultaneous=True``, additionally
        contains ``g2_ci_lower_simultaneous`` and
        ``g2_ci_upper_simultaneous`` — the centred bounds of the
        Westfall-Young studentized-max simultaneous CI with
        family-wise (1 - α) coverage across the vocabulary. The
        simultaneous bounds are wider than per-term and are what to
        report for the *top-ranked* row of a sorted keyness table;
        per-term bounds are what to report for any single term
        named before observing the data.

    Raises
    ------
    ValueError
        If either corpus is empty, ``n_boot < 1``, or
        ``ci_level`` is not in ``(0, 1)``.
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    if formula not in ("rayson", "dunning"):
        raise ValueError(
            f"formula must be 'rayson' or 'dunning'; got {formula!r}"
        )

    dtm_a = a.doc_term_counts(min_count=1)
    dtm_b = b.doc_term_counts(min_count=1)
    all_terms = (
        dtm_a.columns.union(dtm_b.columns) if terms is None else pd.Index(list(terms))
    )

    a_aligned = dtm_a.reindex(columns=all_terms, fill_value=0).astype(np.int64)
    b_aligned = dtm_b.reindex(columns=all_terms, fill_value=0).astype(np.int64)
    a_matrix = sparse.csr_matrix(a_aligned.to_numpy())
    b_matrix = sparse.csr_matrix(b_aligned.to_numpy())

    n_docs_a = a_matrix.shape[0]
    n_docs_b = b_matrix.shape[0]
    if n_docs_a == 0 or n_docs_b == 0:
        raise ValueError(
            "bootstrap_g2_ci needs at least one document on each side; "
            f"got |docs(a)|={n_docs_a}, |docs(b)|={n_docs_b}"
        )

    # Cluster-robust resampling setup. When cluster_col is given, build
    # per-side lists of document-position arrays grouped by cluster id.
    # Doc positions align with the dtm rows because doc_term_counts()
    # iterates a.docs / b.docs in row order.
    clusters_a: list[np.ndarray] | None = None
    clusters_b: list[np.ndarray] | None = None
    if cluster_col is not None:
        clusters_a = _cluster_positions(a, cluster_col, n_docs_a, side="a")
        clusters_b = _cluster_positions(b, cluster_col, n_docs_b, side="b")

    rng = np.random.default_rng(seed)
    n_terms = len(all_terms)
    bootstrap_dist = np.empty((n_boot, n_terms), dtype=np.float64)

    for i in range(n_boot):
        if clusters_a is None or clusters_b is None:
            # IID document resampling, preserving the original counts.
            idx_a = rng.integers(0, n_docs_a, size=n_docs_a)
            idx_b = rng.integers(0, n_docs_b, size=n_docs_b)
        else:
            # Cluster-robust: resample clusters with replacement, take
            # every document in each sampled cluster as a block.
            pick_a = rng.integers(0, len(clusters_a), size=len(clusters_a))
            pick_b = rng.integers(0, len(clusters_b), size=len(clusters_b))
            idx_a = np.concatenate([clusters_a[k] for k in pick_a])
            idx_b = np.concatenate([clusters_b[k] for k in pick_b])
        marg_a = np.asarray(a_matrix[idx_a, :].sum(axis=0)).ravel().astype(np.int64)
        marg_b = np.asarray(b_matrix[idx_b, :].sum(axis=0)).ravel().astype(np.int64)
        n_a_b = int(marg_a.sum())
        n_b_b = int(marg_b.sum())
        if n_a_b == 0 or n_b_b == 0:
            # Degenerate resample (the resampled side has zero tokens).
            # Vanishingly rare on non-pathological corpora; treat as a
            # no-signal iteration so the CI reflects the true variance
            # without inflating from a divide-by-zero NaN.
            bootstrap_dist[i, :] = 0.0
            continue
        bootstrap_dist[i, :] = _g2_signed(
            marg_a, marg_b, n_a_b, n_b_b, formula=formula
        )

    alpha = 1.0 - ci_level

    # Per-term percentile CI (always returned). Calibrated for any
    # single term named in advance; anti-conservative for the
    # top-ranked row of a sorted keyness table.
    per_term_lower = np.quantile(bootstrap_dist, alpha / 2.0, axis=0)
    per_term_upper = np.quantile(bootstrap_dist, 1.0 - alpha / 2.0, axis=0)

    out: dict[str, np.ndarray] = {
        "g2_ci_lower": per_term_lower,
        "g2_ci_upper": per_term_upper,
    }

    if simultaneous:
        # Westfall-Young studentized-max simultaneous CI.
        # For each bootstrap replicate, compute the per-term residual
        # (G²_{t,b} - mean_t) scaled by the per-term bootstrap SD,
        # then take the max absolute studentized residual across
        # terms. The (1 - α) quantile of those maxes is the critical
        # value q. Each term's simultaneous CI is then
        # mean_t ± q · sd_t, which has family-wise (1 - α) coverage
        # across all terms (Westfall & Young 1993). When K = 1 (single
        # term), the max-T distribution degenerates to the per-term
        # distribution and the simultaneous CI equals the per-term CI.
        means = bootstrap_dist.mean(axis=0)
        sds = bootstrap_dist.std(axis=0, ddof=1)
        sds_safe = np.where(sds > 0, sds, 1.0)
        z = (bootstrap_dist - means[None, :]) / sds_safe[None, :]
        max_abs_z = np.abs(z).max(axis=1)
        q_crit = float(np.quantile(max_abs_z, ci_level))
        out["g2_ci_lower_simultaneous"] = means - q_crit * sds_safe
        out["g2_ci_upper_simultaneous"] = means + q_crit * sds_safe

    return pd.DataFrame(out, index=all_terms)


def _cluster_positions(
    corpus: Corpus | CorpusSlice,
    cluster_col: str,
    n_docs: int,
    *,
    side: str,
) -> list[np.ndarray]:
    """Group document positions (0..n_docs-1) by cluster id.

    Document positions align with the dtm rows used in the bootstrap
    because :meth:`doc_term_counts` iterates ``corpus.docs`` in row
    order. Returns one int array of positions per distinct cluster.
    """
    docs = corpus.docs
    if cluster_col not in docs.columns:
        raise ValueError(
            f"cluster_col={cluster_col!r} not found on corpus '{side}'; "
            f"available columns: {list(docs.columns)!r}"
        )
    labels = docs[cluster_col].to_numpy()
    if labels.shape[0] != n_docs:
        raise ValueError(  # pragma: no cover - guards an internal invariant
            f"cluster label count ({labels.shape[0]}) != document count "
            f"({n_docs}) on corpus '{side}'"
        )
    groups: dict[object, list[int]] = {}
    for pos, label in enumerate(labels):
        groups.setdefault(label, []).append(pos)
    return [np.asarray(v, dtype=np.intp) for v in groups.values()]


def _g2_signed(
    a: np.ndarray,
    b: np.ndarray,
    n_a: int,
    n_b: int,
    *,
    formula: LLFormula,
) -> np.ndarray:
    """Vectorised SIGNED G² across all terms.

    Sign convention matches :func:`log_likelihood`: positive when A's
    per-token rate exceeds B's. Used inside the bootstrap loop where
    per-iteration cost dominates total runtime.
    """
    obs_sum = a + b
    total = n_a + n_b
    expected_a = n_a * obs_sum / total
    expected_b = n_b * obs_sum / total
    contrib = (
        xlogy(a, a) - xlogy(a, expected_a) + xlogy(b, b) - xlogy(b, expected_b)
    )
    if formula == "dunning":
        c = n_a - a
        d = n_b - b
        expected_c = n_a - expected_a
        expected_d = n_b - expected_b
        contrib = contrib + (
            xlogy(c, c) - xlogy(c, expected_c) + xlogy(d, d) - xlogy(d, expected_d)
        )
    unsigned: np.ndarray = np.maximum(2.0 * contrib, 0.0)
    a_rate = a / n_a
    b_rate = b / n_b
    sign = np.where(a_rate >= b_rate, 1.0, -1.0)
    signed: np.ndarray = sign * unsigned
    return signed
