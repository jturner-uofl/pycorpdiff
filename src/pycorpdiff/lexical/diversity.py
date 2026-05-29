"""Lexical-diversity metrics and the public ``lexical_diversity`` verb.

See :mod:`pycorpdiff.lexical` for the high-level overview. Each metric
is implemented as a pure function over a token list so they're
trivially composable and individually testable; the public
:func:`lexical_diversity` wraps them with pooling, optional per-period
slicing, and optional bootstrap CIs on per-period values.

References
----------
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot:
  the moving-average type-token ratio (MATTR). *Journal of
  Quantitative Linguistics*, 17(2), 94–100.
- McCarthy, P. M., & Jarvis, S. (2007). vocd: A theoretical and
  empirical evaluation. *Language Testing*, 24(4), 459–488.
- McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A
  validation study of sophisticated approaches to lexical diversity
  assessment. *Behavior Research Methods*, 42(2), 381–392.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import lgamma
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pandas as pd

from ..corpus import Corpus, CorpusSlice

if TYPE_CHECKING:
    import altair as alt


# ----------------------------------------------------------------------
# Pure-math metrics
# ----------------------------------------------------------------------


def ttr(tokens: Sequence[str]) -> float:
    """Type-token ratio (uncorrected).

    Length-dependent: short texts trend toward 1.0, long texts toward
    0. Kept for familiarity and as a sanity check; for honest
    cross-text comparison use :func:`mattr`, :func:`mtld`, or
    :func:`hdd` instead.
    """
    n = len(tokens)
    if n == 0:
        return float("nan")
    return len(set(tokens)) / n


def mattr(tokens: Sequence[str], window: int = 100) -> float:
    """Moving-average TTR (Covington & McFall 2010).

    Walk a window of ``window`` tokens across the stream and average
    the per-window TTRs. Length-robust because every window is the same
    size — the TTR–length confound cancels out.

    For texts with fewer than ``window`` tokens, falls back to plain
    TTR. Default window 100 follows the literature; small windows
    (< 50) tend to overshoot, large windows (> 500) tend to undershoot.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1; got {window}")
    n = len(tokens)
    if n == 0:
        return float("nan")
    if n < window:
        return ttr(tokens)
    ratios: list[float] = []
    for start in range(0, n - window + 1):
        chunk = tokens[start : start + window]
        ratios.append(len(set(chunk)) / window)
    return float(np.mean(ratios))


def _mtld_one_pass(tokens: Sequence[str], threshold: float) -> float:
    """One direction of the MTLD walk.

    Walks tokens, tracking the running TTR. Each time the running TTR
    drops to ``threshold``, records a "factor" (its token length) and
    resets. Partial factors at the end are scaled by how far they got.
    """
    factors_completed: float = 0.0
    factor_lengths_sum = 0
    types_seen: set[str] = set()
    tokens_in_factor = 0
    for tok in tokens:
        tokens_in_factor += 1
        types_seen.add(tok)
        ratio = len(types_seen) / tokens_in_factor
        if ratio <= threshold:
            factors_completed += 1.0
            factor_lengths_sum += tokens_in_factor
            types_seen = set()
            tokens_in_factor = 0
    # Partial factor at the end of the stream — scale by how far the
    # running TTR got toward the threshold (McCarthy & Jarvis 2010,
    # equation 1).
    if tokens_in_factor > 0:
        residual_ratio = len(types_seen) / tokens_in_factor
        if residual_ratio < 1.0:
            partial = (1.0 - residual_ratio) / (1.0 - threshold)
            factors_completed += partial
            factor_lengths_sum += tokens_in_factor
        else:
            # Residual reached 1.0 (every token unique) → undefined
            # partial; ignore.
            pass
    if factors_completed == 0:
        return float("nan")
    return factor_lengths_sum / factors_completed


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).

    Walks the token stream forward and backward, recording each "factor"
    — the run of tokens until the running TTR drops to ``threshold``.
    Returns the mean factor length, forward + backward averaged.

    The ``0.72`` threshold is the empirically-derived stable point from
    McCarthy 2005; deviating from it is discouraged unless you have
    very specific calibration needs.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")
    n = len(tokens)
    if n == 0:
        return float("nan")
    forward = _mtld_one_pass(tokens, threshold)
    backward = _mtld_one_pass(list(reversed(tokens)), threshold)
    if np.isnan(forward) and np.isnan(backward):
        return float("nan")
    if np.isnan(forward):
        return backward
    if np.isnan(backward):
        return forward
    return (forward + backward) / 2.0


def _log_choose(n: int, k: int) -> float:
    """log(n choose k) via lgamma — stable for the large counts HD-D produces."""
    if k < 0 or k > n:
        return float("-inf")
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def hdd(tokens: Sequence[str], sample_size: int = 42) -> float:
    """HD-D: hypergeometric-based expected vocabulary in a 42-token sample
    (McCarthy & Jarvis 2007).

    For each unique type *t* with count :math:`c_t` in a corpus of
    *N* tokens, the probability of drawing at least one token of type
    *t* in a uniform random sample of ``sample_size`` tokens is
    :math:`1 - \\binom{N - c_t}{s} / \\binom{N}{s}`. HD-D is the sum
    of those probabilities across all types — interpretable as the
    expected number of unique types in a size-:math:`s` random
    sample. ``sample_size=42`` is the published convention; range is
    roughly :math:`[0, \\text{sample\\_size}]`.

    For texts shorter than ``sample_size``, returns the count of unique
    types directly (the sample-size cap is moot).
    """
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1; got {sample_size}")
    n = len(tokens)
    if n == 0:
        return float("nan")
    if n <= sample_size:
        # Drawing all tokens trivially gets every type; expected unique
        # types = actual unique types.
        return float(len(set(tokens)))
    counts = pd.Series(tokens).value_counts().to_numpy()
    log_total = _log_choose(n, sample_size)
    total = 0.0
    for c in counts:
        # P(zero occurrences of this type in the sample):
        log_zero = _log_choose(n - int(c), sample_size) - log_total
        p_at_least_one = 1.0 - np.exp(log_zero)
        total += p_at_least_one
    return float(total)


# ----------------------------------------------------------------------
# Result dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LexicalDiversityResult:
    """Pooled lexical-diversity metrics over a whole corpus / slice.

    Attributes
    ----------
    ttr, mattr, mtld, hdd
        The four metric values on the pooled token stream.
    n_tokens, n_types
        Total tokens and unique types observed.
    label
        Short identifier for the corpus this was computed on.
    per_doc
        Optional per-document metric breakdown, present when
        ``per_doc=True`` was passed to
        :func:`lexical_diversity`. Columns: ``doc_id``, ``n_tokens``,
        ``ttr``, ``mattr``, ``mtld``, ``hdd``.
    params
        The kwargs the metrics were computed with (window, threshold, …).
    """

    ttr: float
    mattr: float
    mtld: float
    hdd: float
    n_tokens: int
    n_types: int
    label: str = "corpus"
    per_doc: pd.DataFrame | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"metric": "TTR", "value": self.ttr},
                {"metric": "MATTR", "value": self.mattr},
                {"metric": "MTLD", "value": self.mtld},
                {"metric": "HD-D", "value": self.hdd},
            ]
        )

    def summary(self) -> str:
        return (
            f"LexicalDiversityResult(label={self.label!r}, "
            f"|tokens|={self.n_tokens:,}, |types|={self.n_types:,}, "
            f"TTR={self.ttr:.3f}, MATTR={self.mattr:.3f}, "
            f"MTLD={self.mtld:.1f}, HD-D={self.hdd:.2f})"
        )

    def __repr__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class LexicalDiversityTrajectory:
    """Per-period lexical diversity over a time-indexed corpus.

    Attributes
    ----------
    table
        Long-form DataFrame with one row per ``(period, metric)`` pair.
        Columns: ``period``, ``metric``, ``value``, ``n_tokens``,
        ``n_types`` — and when bootstrap CIs were requested,
        ``ci_lower`` and ``ci_upper``.
    freq
        pandas offset alias the periods were bucketed by ("Y", "Q",
        "M", …).
    label
        Short identifier for the source corpus.
    params
        kwargs the metrics were computed with.
    """

    table: pd.DataFrame
    freq: str
    label: str = "corpus"
    params: dict[str, Any] = field(default_factory=dict)

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def plot(self, **kw: Any) -> alt.Chart:
        """Multi-line trajectory plot, one line per metric over time."""
        from ..viz.lexical_diversity import lexical_diversity_trajectory_plot

        return lexical_diversity_trajectory_plot(self.table, **kw)

    def summary(self) -> str:
        n_periods = self.table["period"].nunique()
        metrics = self.table["metric"].unique().tolist()
        return (
            f"LexicalDiversityTrajectory(label={self.label!r}, "
            f"freq={self.freq!r}, periods={n_periods}, "
            f"metrics={metrics})"
        )

    def __repr__(self) -> str:
        return self.summary()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

_METRICS = ("ttr", "mattr", "mtld", "hdd")


def _all_tokens(corpus: Corpus | CorpusSlice) -> list[str]:
    """Pool every token from every document into one flat list."""
    return [tok for doc in corpus.tokens() for tok in doc]


def _compute_pooled(
    tokens: Sequence[str],
    *,
    window: int,
    threshold: float,
    sample_size: int,
) -> dict[str, float]:
    """Run all four metrics on a flat token stream."""
    return {
        "ttr": ttr(tokens),
        "mattr": mattr(tokens, window=window),
        "mtld": mtld(tokens, threshold=threshold),
        "hdd": hdd(tokens, sample_size=sample_size),
    }


def _bootstrap_ci_per_period(
    docs_tokens: list[list[str]],
    *,
    window: int,
    threshold: float,
    sample_size: int,
    n_boot: int,
    ci_level: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    """Document-level bootstrap percentile CI on each metric for one period.

    Resample documents with replacement, repool, recompute metrics;
    take the ``alpha/2`` and ``1 - alpha/2`` quantiles across
    iterations.
    """
    n_docs = len(docs_tokens)
    boot = {m: np.empty(n_boot, dtype=np.float64) for m in _METRICS}
    for i in range(n_boot):
        idx = rng.integers(0, n_docs, size=n_docs)
        pooled = [tok for j in idx for tok in docs_tokens[j]]
        if not pooled:
            for m in _METRICS:
                boot[m][i] = float("nan")
            continue
        values = _compute_pooled(
            pooled, window=window, threshold=threshold, sample_size=sample_size
        )
        for m in _METRICS:
            boot[m][i] = values[m]
    alpha = 1.0 - ci_level
    out: dict[str, tuple[float, float]] = {}
    for m in _METRICS:
        vals = boot[m][np.isfinite(boot[m])]
        if vals.size == 0:
            out[m] = (float("nan"), float("nan"))
        else:
            out[m] = (
                float(np.quantile(vals, alpha / 2.0)),
                float(np.quantile(vals, 1.0 - alpha / 2.0)),
            )
    return out


# ----------------------------------------------------------------------
# Public verb
# ----------------------------------------------------------------------


@overload
def lexical_diversity(
    corpus: Corpus | CorpusSlice,
    *,
    freq: None = ...,
    time_col: str = ...,
    per_doc: bool = ...,
    window: int = ...,
    threshold: float = ...,
    sample_size: int = ...,
    ci: Literal["none", "bootstrap"] = ...,
    n_boot: int = ...,
    ci_level: float = ...,
    seed: int | None = ...,
    metrics: Iterable[str] | None = ...,
) -> LexicalDiversityResult: ...


@overload
def lexical_diversity(
    corpus: Corpus | CorpusSlice,
    *,
    freq: str,
    time_col: str = ...,
    per_doc: bool = ...,
    window: int = ...,
    threshold: float = ...,
    sample_size: int = ...,
    ci: Literal["none", "bootstrap"] = ...,
    n_boot: int = ...,
    ci_level: float = ...,
    seed: int | None = ...,
    metrics: Iterable[str] | None = ...,
) -> LexicalDiversityTrajectory: ...


def lexical_diversity(
    corpus: Corpus | CorpusSlice,
    *,
    freq: str | None = None,
    time_col: str = "date",
    per_doc: bool = False,
    window: int = 100,
    threshold: float = 0.72,
    sample_size: int = 42,
    ci: Literal["none", "bootstrap"] = "none",
    n_boot: int = 199,
    ci_level: float = 0.95,
    seed: int | None = None,
    metrics: Iterable[str] | None = None,
) -> LexicalDiversityResult | LexicalDiversityTrajectory:
    """Compute lexical-diversity metrics on a corpus.

    Without ``freq``: returns a single :class:`LexicalDiversityResult`
    pooled over every document.

    With ``freq`` (a pandas offset alias like ``"Y"``, ``"Q"``,
    ``"M"``): slices the corpus by period and computes metrics within
    each period, returning a :class:`LexicalDiversityTrajectory`.

    Parameters
    ----------
    corpus
        Source corpus or slice.
    freq
        Optional period-binning alias. When supplied, ``corpus.docs``
        must contain ``time_col``.
    time_col
        Column carrying timestamps. Default ``"date"``.
    per_doc
        If ``True`` (and ``freq=None``), include a per-document
        breakdown DataFrame on the result. Ignored in the temporal
        path — per-document detail is rarely useful at that
        granularity.
    window
        MATTR window length. Default ``100`` matches the literature.
    threshold
        MTLD running-TTR threshold. Default ``0.72`` from McCarthy 2005.
    sample_size
        HD-D random-sample size. Default ``42`` is the published
        convention.
    ci
        Only meaningful in the temporal path. ``"bootstrap"`` adds
        ``ci_lower`` and ``ci_upper`` columns on the trajectory table.
        Document-level resampling within each period. *Caveat:* MTLD
        and MATTR are path-dependent (order-sensitive) walks, so
        document-level bootstrap can produce mildly biased bands —
        the point estimate occasionally falls outside the percentile
        CI. The *width* remains a useful stability signal, but treat
        MTLD / MATTR CIs as approximate. TTR and HD-D, being
        order-independent aggregates over tokens, give clean
        percentile CIs.
    n_boot, ci_level, seed
        Bootstrap parameters when ``ci="bootstrap"``. Default
        ``n_boot=199`` keeps per-period cost manageable across long
        trajectories; raise it (≥ 999) when you need tight bounds.
    metrics
        Optional subset of the four metric names to compute, e.g.
        ``("mtld", "hdd")`` to skip TTR + MATTR. Default ``None``
        computes all four.

    Returns
    -------
    LexicalDiversityResult | LexicalDiversityTrajectory

    Raises
    ------
    ValueError
        On empty corpora, missing ``time_col``, unknown metric names,
        or out-of-range ``ci_level``.
    """
    metrics_set = set(metrics) if metrics is not None else set(_METRICS)
    unknown = metrics_set - set(_METRICS)
    if unknown:
        raise ValueError(
            f"unknown metric(s): {sorted(unknown)}; "
            f"valid metrics are {list(_METRICS)}"
        )
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")

    label = _corpus_label(corpus)

    if freq is None:
        return _lexical_diversity_pooled(
            corpus,
            per_doc=per_doc,
            window=window,
            threshold=threshold,
            sample_size=sample_size,
            metrics_set=metrics_set,
            label=label,
        )
    return _lexical_diversity_trajectory(
        corpus,
        freq=freq,
        time_col=time_col,
        window=window,
        threshold=threshold,
        sample_size=sample_size,
        ci=ci,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        metrics_set=metrics_set,
        label=label,
    )


def _corpus_label(corpus: Corpus | CorpusSlice) -> str:
    if isinstance(corpus, CorpusSlice):
        return corpus.label
    return "corpus"


def _lexical_diversity_pooled(
    corpus: Corpus | CorpusSlice,
    *,
    per_doc: bool,
    window: int,
    threshold: float,
    sample_size: int,
    metrics_set: set[str],
    label: str,
) -> LexicalDiversityResult:
    docs_tokens = corpus.tokens()
    if not docs_tokens:
        raise ValueError("lexical_diversity needs at least one document")
    all_tokens = [tok for doc in docs_tokens for tok in doc]
    if not all_tokens:
        raise ValueError("lexical_diversity needs at least one token across the corpus")

    values = _compute_pooled(
        all_tokens, window=window, threshold=threshold, sample_size=sample_size
    )
    per_doc_table: pd.DataFrame | None = None
    if per_doc:
        rows = []
        for doc_id, doc_tokens in enumerate(docs_tokens):
            if not doc_tokens:
                continue
            doc_vals = _compute_pooled(
                doc_tokens,
                window=window,
                threshold=threshold,
                sample_size=sample_size,
            )
            rows.append(
                {
                    "doc_id": doc_id,
                    "n_tokens": len(doc_tokens),
                    **doc_vals,
                }
            )
        per_doc_table = pd.DataFrame(rows)

    return LexicalDiversityResult(
        ttr=values["ttr"] if "ttr" in metrics_set else float("nan"),
        mattr=values["mattr"] if "mattr" in metrics_set else float("nan"),
        mtld=values["mtld"] if "mtld" in metrics_set else float("nan"),
        hdd=values["hdd"] if "hdd" in metrics_set else float("nan"),
        n_tokens=len(all_tokens),
        n_types=len(set(all_tokens)),
        label=label,
        per_doc=per_doc_table,
        params={
            "window": window,
            "threshold": threshold,
            "sample_size": sample_size,
            "metrics": tuple(sorted(metrics_set)),
        },
    )


def _lexical_diversity_trajectory(
    corpus: Corpus | CorpusSlice,
    *,
    freq: str,
    time_col: str,
    window: int,
    threshold: float,
    sample_size: int,
    ci: Literal["none", "bootstrap"],
    n_boot: int,
    ci_level: float,
    seed: int | None,
    metrics_set: set[str],
    label: str,
) -> LexicalDiversityTrajectory:
    # CorpusSlice has a .by_time method through delegation to parent,
    # but the cleanest path is to construct the TemporalCorpus directly
    # over the slice's effective docs frame; .by_time on the slice would
    # try to slice the *parent*. We work over the slice's docs directly.
    if time_col not in corpus.docs.columns:
        raise ValueError(
            f"time_col={time_col!r} missing from corpus.docs; "
            f"available: {list(corpus.docs.columns)}"
        )
    if ci not in ("none", "bootstrap"):
        raise ValueError(f"ci must be 'none' or 'bootstrap'; got {ci!r}")

    times = pd.to_datetime(corpus.docs[time_col])
    period_series = times.dt.to_period(freq)
    docs_frame = corpus.docs.assign(_period=period_series)

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    all_tokens_by_doc = corpus.tokens()
    # Align tokens with the docs frame's index order so the period
    # grouping matches.
    doc_index_to_pos = {idx: pos for pos, idx in enumerate(corpus.docs.index)}

    for period, group_df in docs_frame.groupby("_period", sort=True):
        positions = [doc_index_to_pos[idx] for idx in group_df.index]
        period_docs_tokens = [all_tokens_by_doc[p] for p in positions]
        pooled = [tok for doc in period_docs_tokens for tok in doc]
        n_tokens = len(pooled)
        n_types = len(set(pooled))
        if n_tokens == 0:
            continue
        if n_tokens < 25:
            warnings.warn(
                f"period {period} has only {n_tokens} tokens; lexical-"
                "diversity metrics will be noisy at this scale",
                UserWarning,
                stacklevel=3,
            )
        values = _compute_pooled(
            pooled, window=window, threshold=threshold, sample_size=sample_size
        )
        ci_bounds = (
            _bootstrap_ci_per_period(
                period_docs_tokens,
                window=window,
                threshold=threshold,
                sample_size=sample_size,
                n_boot=n_boot,
                ci_level=ci_level,
                rng=rng,
            )
            if ci == "bootstrap"
            else None
        )
        for metric_key, metric_label in (
            ("ttr", "TTR"),
            ("mattr", "MATTR"),
            ("mtld", "MTLD"),
            ("hdd", "HD-D"),
        ):
            if metric_key not in metrics_set:
                continue
            row: dict[str, Any] = {
                "period": period,
                "metric": metric_label,
                "value": values[metric_key],
                "n_tokens": n_tokens,
                "n_types": n_types,
            }
            if ci_bounds is not None:
                lo, hi = ci_bounds[metric_key]
                row["ci_lower"] = lo
                row["ci_upper"] = hi
            rows.append(row)

    if not rows:
        raise ValueError(
            "no period produced tokens; check that "
            f"{time_col!r} parses and the corpus is non-empty"
        )
    table = pd.DataFrame(rows).sort_values(["metric", "period"]).reset_index(drop=True)
    return LexicalDiversityTrajectory(
        table=table,
        freq=freq,
        label=label,
        params={
            "window": window,
            "threshold": threshold,
            "sample_size": sample_size,
            "metrics": tuple(sorted(metrics_set)),
            "ci": ci,
            "n_boot": n_boot if ci == "bootstrap" else None,
            "ci_level": ci_level if ci == "bootstrap" else None,
            "seed": seed if ci == "bootstrap" else None,
        },
    )
