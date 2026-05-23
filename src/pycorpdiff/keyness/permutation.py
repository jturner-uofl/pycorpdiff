"""Permutation-test *p*-values for keyness.

The asymptotic χ²-based *p*-value reported by :func:`log_likelihood`
relies on the large-sample approximation Dunning warned about in 1993:
when expected cell counts are small (~< 5), the χ²(df=1) reference
distribution stops being faithful. A permutation test sidesteps this
entirely — it builds the null distribution empirically from the data
itself by repeatedly shuffling document labels and recomputing G².

The standard reference is Westfall & Young (1993), *Resampling-Based
Multiple Testing*. The "+1/+1" small-sample correction in the *p*-value
formula is Phipson & Smyth (2010); without it the smallest reportable
*p* is 0, which is misleading for any finite *B*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import xlogy

from ..corpus import Corpus, CorpusSlice


def permutation_pvalues(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    *,
    terms: pd.Index | list[str] | None = None,
    n_permutations: int = 999,
    seed: int | None = None,
) -> pd.Series:
    """Compute empirical permutation *p*-values for per-term keyness.

    For each permutation, the documents from both corpora are pooled,
    randomly relabelled into two groups preserving the original
    document counts ``|docs(a)|`` and ``|docs(b)|``, and a per-term G²
    is recomputed against the new marginal counts. The reported
    *p*-value for each term is the Phipson–Smyth (2010) empirical
    fraction::

        p = (#perms with |G²_perm| ≥ |G²_observed| + 1) / (B + 1)

    Documents (rather than tokens) are the unit of exchangeability —
    this matches the null "speeches are exchangeable between frames"
    that most corpus-linguistic two-sample questions are really
    testing.

    Parameters
    ----------
    a, b
        The two corpora (or slices) to test.
    terms
        Optional restriction to a subset of vocabulary terms. If
        ``None`` (default) every term in the union of vocabularies is
        scored.
    n_permutations
        Number of label permutations. 999 is the conventional default
        — coarse enough to be fast, fine enough that the smallest
        attainable *p* (= 1/1000 = 0.001) is below the usual 0.01
        screening threshold.
    seed
        Optional random seed for reproducibility.

    Returns
    -------
    pandas.Series
        Indexed by term, named ``p_permutation``. Values in [1/(B+1), 1].
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")

    dtm_a = a.doc_term_counts(min_count=1)
    dtm_b = b.doc_term_counts(min_count=1)

    all_terms = (
        dtm_a.columns.union(dtm_b.columns) if terms is None else pd.Index(list(terms))
    )

    a_aligned = dtm_a.reindex(columns=all_terms, fill_value=0).astype(np.int64)
    b_aligned = dtm_b.reindex(columns=all_terms, fill_value=0).astype(np.int64)

    a_matrix = sparse.csr_matrix(a_aligned.to_numpy())
    b_matrix = sparse.csr_matrix(b_aligned.to_numpy())
    stacked = sparse.vstack([a_matrix, b_matrix]).tocsr()

    n_docs_a = a_matrix.shape[0]
    n_docs_total = stacked.shape[0]

    if n_docs_a == 0 or n_docs_a == n_docs_total:
        raise ValueError(
            "permutation_pvalues needs at least one document on each side; "
            f"got |docs(a)|={n_docs_a}, |docs(b)|={n_docs_total - n_docs_a}"
        )

    doc_lens = np.asarray(stacked.sum(axis=1)).ravel().astype(np.int64)
    full_counts = np.asarray(stacked.sum(axis=0)).ravel().astype(np.int64)
    total_n = int(doc_lens.sum())

    if total_n == 0:
        raise ValueError("permutation_pvalues needs at least one token across both corpora")

    a_marginal = np.asarray(a_matrix.sum(axis=0)).ravel().astype(np.int64)
    b_marginal = full_counts - a_marginal
    n_a_observed = int(a_marginal.sum())
    n_b_observed = total_n - n_a_observed
    observed_g2 = _g2_unsigned(a_marginal, b_marginal, n_a_observed, n_b_observed)
    observed_abs = np.abs(observed_g2)

    rng = np.random.default_rng(seed)
    extreme = np.zeros(len(all_terms), dtype=np.int64)

    for _ in range(n_permutations):
        perm = rng.permutation(n_docs_total)
        idx_a = perm[:n_docs_a]
        # Vectorised marginal sum for the shuffled "a" assignment.
        marg_a_perm = np.asarray(stacked[idx_a, :].sum(axis=0)).ravel().astype(np.int64)
        marg_b_perm = full_counts - marg_a_perm
        n_a_perm = int(doc_lens[idx_a].sum())
        n_b_perm = total_n - n_a_perm
        if n_a_perm == 0 or n_b_perm == 0:
            # Degenerate permutation — all-zero docs landed in one side.
            # Treat as null contribution of zero (no terms exceed observed).
            continue
        g2_perm = _g2_unsigned(marg_a_perm, marg_b_perm, n_a_perm, n_b_perm)
        extreme += np.abs(g2_perm) >= observed_abs

    p_perm = (extreme + 1.0) / (n_permutations + 1.0)
    return pd.Series(p_perm, index=all_terms, name="p_permutation")


def _g2_unsigned(
    a: np.ndarray,
    b: np.ndarray,
    n_a: int,
    n_b: int,
) -> np.ndarray:
    """Vectorised unsigned G² for a vocabulary of terms.

    Mirrors the math in :func:`pycorpdiff.keyness.loglikelihood.log_likelihood`
    but operates on already-aligned numpy arrays and skips the sign
    + p-value bookkeeping. Used internally by the permutation loop
    where the per-iteration cost dominates.
    """
    obs_sum = a + b
    total = n_a + n_b
    expected_a = n_a * obs_sum / total
    expected_b = n_b * obs_sum / total
    unsigned = 2.0 * (
        xlogy(a, a) - xlogy(a, expected_a) + xlogy(b, b) - xlogy(b, expected_b)
    )
    clipped: np.ndarray = np.maximum(unsigned, 0.0)
    return clipped
