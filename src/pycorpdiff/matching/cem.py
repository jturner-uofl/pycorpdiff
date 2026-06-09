"""Coarsened Exact Matching for :class:`pycorpdiff.Corpus` pairs.

See :mod:`pycorpdiff.matching` for the high-level motivation and the
methodology citation (Iacus, King & Porro 2012).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..corpus import Corpus, CorpusSlice


@dataclass(frozen=True)
class MatchResult:
    """The output of :func:`match`: matched slices plus a diagnostic.

    Attributes
    ----------
    a_matched, b_matched
        :class:`CorpusSlice` views of the input corpora containing only
        documents that landed in a matched stratum (and survived the
        within-stratum subsampling). Usable directly with
        :func:`pycorpdiff.compare`, :func:`pycorpdiff.track`, etc.
    on
        The covariates the match was conditioned on.
    strata
        Per-stratum count table with columns ``stratum``, ``n_a_pre``,
        ``n_b_pre``, ``n_kept``, ``matched``. One row per stratum
        encountered (matched or not).
    imbalance
        DataFrame indexed by covariate with ``l1_pre`` and ``l1_post``
        columns. L1 imbalance is the standard CEM diagnostic: half the
        sum of absolute differences between the two sides' empirical
        marginal distributions on the coarsened covariate. Range [0, 1];
        lower is more balanced. ``l1_post`` should be ≪ ``l1_pre`` for
        every matched covariate — that's exactly what matching buys you.
    n_a_pre, n_a_post, n_b_pre, n_b_post
        Document counts before and after matching, per side.
    seed
        The RNG seed used for within-stratum subsampling. Saved so the
        match is fully reproducible.
    """

    a_matched: CorpusSlice
    b_matched: CorpusSlice
    on: tuple[str, ...]
    strata: pd.DataFrame
    imbalance: pd.DataFrame
    n_a_pre: int
    n_a_post: int
    n_b_pre: int
    n_b_post: int
    seed: int | None
    cuts_used: Mapping[str, list[float] | str] = field(default_factory=dict)

    def summary(self) -> str:
        """One-paragraph human-readable match report."""
        n_strata_total = len(self.strata)
        n_strata_matched = int(self.strata["matched"].sum())
        a_pct = 100.0 * self.n_a_post / max(self.n_a_pre, 1)
        b_pct = 100.0 * self.n_b_post / max(self.n_b_pre, 1)
        imb_pre = self.imbalance["l1_pre"].mean()
        imb_post = self.imbalance["l1_post"].mean()
        return (
            f"CEM match on {list(self.on)}: "
            f"{n_strata_matched}/{n_strata_total} strata matched, "
            f"|a|: {self.n_a_pre} → {self.n_a_post} ({a_pct:.0f}%), "
            f"|b|: {self.n_b_pre} → {self.n_b_post} ({b_pct:.0f}%). "
            f"Mean L1 imbalance: {imb_pre:.3f} → {imb_post:.3f}."
        )

    def __repr__(self) -> str:
        return f"MatchResult(on={list(self.on)}, |a|={self.n_a_post}, |b|={self.n_b_post})"


def _coerce_to_parent_and_mask(
    corpus: Corpus | CorpusSlice,
) -> tuple[Corpus, pd.Series]:
    """Normalise to ``(parent_corpus, boolean_mask_over_parent_docs)``."""
    if isinstance(corpus, CorpusSlice):
        return corpus.parent, corpus.mask.copy()
    mask = pd.Series(True, index=corpus.docs.index)
    return corpus, mask


def _coarsen(
    series_a: pd.Series,
    series_b: pd.Series,
    spec: int | Sequence[float] | None,
) -> tuple[pd.Series, pd.Series, list[float] | str]:
    """Coarsen one covariate.

    Numeric covariates are binned by quantile (default 5 bins).
    Categorical covariates are returned as-is.

    ``spec`` either gives the number of quantile bins (int) or a list
    of explicit edges (Sequence[float]). ``None`` triggers the default.
    """
    combined = pd.concat([series_a, series_b], ignore_index=True)
    if pd.api.types.is_numeric_dtype(combined):
        if isinstance(spec, Sequence) and not isinstance(spec, str):
            edges = list(spec)
        else:
            n_bins = spec if isinstance(spec, int) and spec > 0 else 5
            edges_arr = np.unique(
                np.quantile(combined.dropna(), np.linspace(0, 1, n_bins + 1))
            )
            edges = edges_arr.tolist()
        if len(edges) < 2:
            # Constant covariate — bin everything into one stratum-bin.
            edges = [combined.min() - 1e-9, combined.max() + 1e-9]
        # ``include_lowest`` matches pd.cut's expectation that the left
        # edge is inclusive on the lowest bin.
        binned_a = pd.cut(series_a, bins=edges, include_lowest=True, duplicates="drop")
        binned_b = pd.cut(series_b, bins=edges, include_lowest=True, duplicates="drop")
        return binned_a.astype(str), binned_b.astype(str), edges
    # Categorical / object dtype — use values as-is.
    return series_a.astype(str), series_b.astype(str), "categorical"


def _l1_imbalance(coarsened_a: pd.Series, coarsened_b: pd.Series) -> float:
    """L1 imbalance: ½ Σ |p_a(stratum) - p_b(stratum)| on the coarsened
    covariate. Range [0, 1]; 0 = perfect balance.

    Source: Iacus, King & Porro (2008); the standard CEM diagnostic.
    """
    n_a = len(coarsened_a)
    n_b = len(coarsened_b)
    if n_a == 0 or n_b == 0:
        return 1.0
    p_a = coarsened_a.value_counts(normalize=True)
    p_b = coarsened_b.value_counts(normalize=True)
    all_keys = p_a.index.union(p_b.index)
    return 0.5 * float(
        (p_a.reindex(all_keys, fill_value=0.0)
         - p_b.reindex(all_keys, fill_value=0.0)).abs().sum()
    )


def match(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    *,
    on: Sequence[str],
    cuts: Mapping[str, int | Sequence[float]] | None = None,
    seed: int | None = None,
    subsample: bool = True,
) -> MatchResult:
    """Coarsened-exact-match two corpora on document-level covariates.

    Parameters
    ----------
    a, b
        The two corpora (or slices) to balance.
    on
        Column names (in both ``a.docs`` and ``b.docs``) to match on.
        Numeric columns are quantile-binned (see ``cuts``); object /
        categorical columns are used verbatim.
    cuts
        Per-covariate coarsening spec. Two forms:

        - ``cuts={"year": 5}`` — number of quantile bins for that
          numeric covariate (default ``5`` when unspecified).
        - ``cuts={"year": [2000, 2010, 2016, 2020]}`` — explicit bin
          edges (left edge inclusive on the lowest bin).

        Ignored for non-numeric covariates.
    seed
        Optional RNG seed for the within-stratum subsampling. Match
        results are fully reproducible under a fixed seed.
    subsample
        If ``True`` (default), within each matched stratum the
        over-represented side is randomly subsampled to ``min(n_a, n_b)``
        ("k-to-k" matching, the Iacus-King-Porro default). If ``False``,
        all documents in matched strata are kept; counts per stratum may
        be uneven, which leaves residual within-stratum imbalance.

    Returns
    -------
    MatchResult
        Containing matched slices and the per-covariate L1 imbalance
        before vs after.

    Raises
    ------
    ValueError
        If ``on`` is empty, a covariate is missing from either side, or
        no stratum contains documents from both sides (no matched pairs
        possible).

    Examples
    --------
    >>> import pycorpdiff as pcd  # doctest: +SKIP
    >>> corpus = pcd.load_hansard_sample()  # doctest: +SKIP
    >>> human = corpus.slice(frame="humanising")  # doctest: +SKIP
    >>> criminal = corpus.slice(frame="criminalising")  # doctest: +SKIP
    >>> m = pcd.match(human, criminal, on=["year", "party"], seed=0)  # doctest: +SKIP
    >>> print(m.summary())  # doctest: +SKIP
    >>> keyness = pcd.compare(m.a_matched, m.b_matched).keyness()  # doctest: +SKIP
    """
    on_list = list(on)
    if not on_list:
        raise ValueError("at least one covariate must be supplied via `on=`")

    parent_a, base_mask_a = _coerce_to_parent_and_mask(a)
    parent_b, base_mask_b = _coerce_to_parent_and_mask(b)

    df_a = parent_a.docs.loc[base_mask_a]
    df_b = parent_b.docs.loc[base_mask_b]
    n_a_pre = len(df_a)
    n_b_pre = len(df_b)
    if n_a_pre == 0 or n_b_pre == 0:
        raise ValueError(
            "match() needs at least one document on each side; got "
            f"|a|={n_a_pre}, |b|={n_b_pre}"
        )
    for col in on_list:
        if col not in df_a.columns:
            raise ValueError(f"covariate {col!r} missing from corpus a")
        if col not in df_b.columns:
            raise ValueError(f"covariate {col!r} missing from corpus b")

    # Drop rows with NaN in any matching covariate — keeps the stratum
    # set well-defined; NaN-stratum matching is an explicit choice we
    # don't surface here.
    keep_a = df_a[on_list].notna().all(axis=1)
    keep_b = df_b[on_list].notna().all(axis=1)
    df_a = df_a.loc[keep_a]
    df_b = df_b.loc[keep_b]

    # Coarsen each covariate, recording the edges used per side so
    # both sides land in the same bins.
    coarsened_a: dict[str, pd.Series] = {}
    coarsened_b: dict[str, pd.Series] = {}
    cuts_used: dict[str, list[float] | str] = {}
    for col in on_list:
        spec = cuts.get(col) if cuts else None
        ca, cb, edges = _coarsen(df_a[col], df_b[col], spec)
        # _coarsen drops NaN through pd.cut; restore index alignment.
        coarsened_a[col] = ca
        coarsened_b[col] = cb
        cuts_used[col] = edges

    # Per-covariate L1 imbalance *before* matching.
    l1_pre = {col: _l1_imbalance(coarsened_a[col], coarsened_b[col]) for col in on_list}

    # Build the joint stratum key by string-joining the coarsened columns.
    stratum_a = pd.Series(
        ["|".join(str(coarsened_a[col].loc[idx]) for col in on_list) for idx in df_a.index],
        index=df_a.index,
        name="stratum",
    )
    stratum_b = pd.Series(
        ["|".join(str(coarsened_b[col].loc[idx]) for col in on_list) for idx in df_b.index],
        index=df_b.index,
        name="stratum",
    )

    # Stratum-level diagnostics.
    counts_a_pre = stratum_a.value_counts()
    counts_b_pre = stratum_b.value_counts()
    all_strata = counts_a_pre.index.union(counts_b_pre.index)
    matched_strata = counts_a_pre.index.intersection(counts_b_pre.index)
    if len(matched_strata) == 0:
        raise ValueError(
            "no stratum contains documents from both sides; no matches possible. "
            "Try coarser cuts (fewer bins per numeric covariate) or fewer covariates."
        )

    rng = np.random.default_rng(seed)
    keep_idx_a: list[Any] = []
    keep_idx_b: list[Any] = []
    n_kept_per_stratum: dict[str, int] = {}

    for s in matched_strata:
        idx_a = stratum_a.index[stratum_a == s].to_list()
        idx_b = stratum_b.index[stratum_b == s].to_list()
        if subsample:
            k = min(len(idx_a), len(idx_b))
            sel_a = (
                rng.choice(idx_a, size=k, replace=False).tolist()
                if len(idx_a) > k
                else list(idx_a)
            )
            sel_b = (
                rng.choice(idx_b, size=k, replace=False).tolist()
                if len(idx_b) > k
                else list(idx_b)
            )
            n_kept_per_stratum[s] = k
        else:
            sel_a = idx_a
            sel_b = idx_b
            n_kept_per_stratum[s] = len(idx_a) + len(idx_b)
        keep_idx_a.extend(sel_a)
        keep_idx_b.extend(sel_b)

    # Construct refined masks over the parent corpora.
    matched_mask_a = pd.Series(False, index=parent_a.docs.index)
    matched_mask_b = pd.Series(False, index=parent_b.docs.index)
    matched_mask_a.loc[keep_idx_a] = True
    matched_mask_b.loc[keep_idx_b] = True

    # Post-match L1 imbalance.
    coarsened_a_post = {
        col: coarsened_a[col].loc[keep_idx_a] for col in on_list
    }
    coarsened_b_post = {
        col: coarsened_b[col].loc[keep_idx_b] for col in on_list
    }
    l1_post = {
        col: _l1_imbalance(coarsened_a_post[col], coarsened_b_post[col])
        for col in on_list
    }

    a_matched = CorpusSlice(parent=parent_a, mask=matched_mask_a, filters={"matched": True})
    b_matched = CorpusSlice(parent=parent_b, mask=matched_mask_b, filters={"matched": True})

    strata_rows = []
    for s in all_strata:
        n_a = int(counts_a_pre.get(s, 0))
        n_b = int(counts_b_pre.get(s, 0))
        matched = s in matched_strata
        n_kept = (2 * n_kept_per_stratum[s]) if (matched and subsample) else (
            n_kept_per_stratum[s] if matched else 0
        )
        strata_rows.append(
            {
                "stratum": s,
                "n_a_pre": n_a,
                "n_b_pre": n_b,
                "n_kept": n_kept,
                "matched": matched,
            }
        )
    strata_df = pd.DataFrame(strata_rows).set_index("stratum")

    imbalance_df = pd.DataFrame(
        {
            "l1_pre": [l1_pre[col] for col in on_list],
            "l1_post": [l1_post[col] for col in on_list],
        },
        index=on_list,
    )
    imbalance_df.index.name = "covariate"

    return MatchResult(
        a_matched=a_matched,
        b_matched=b_matched,
        on=tuple(on_list),
        strata=strata_df,
        imbalance=imbalance_df,
        n_a_pre=n_a_pre,
        n_a_post=int(matched_mask_a.sum()),
        n_b_pre=n_b_pre,
        n_b_post=int(matched_mask_b.sum()),
        seed=seed,
        cuts_used=cuts_used,
    )
