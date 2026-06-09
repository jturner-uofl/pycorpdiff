"""Sense drift detection: when does a corpus's sense distribution change, and how.

This is the diachronic, *explainable* counterpart to :func:`induce_senses`.
Where :func:`induce_senses` partitions a single snapshot into senses, and
:func:`semantic_trajectory` tracks one term's centroid over time,
:func:`sense_drift` answers two questions across a time-ordered corpus:

1. **Detection** --- *when* does the sense distribution of a token (or a
   record set) drift away from a reference period?
2. **Explanation** --- *what kind* of drift is it: a genuinely new sense
   **emerging**, an existing sense **shifting** in frequency, or the
   sense distribution **broadening**?

The design borrows from two literatures and fuses them:

- **Concept-drift detection** (Sethi & Kantardzic 2017's Margin Density
  Drift Detection): watch the density of a classifier's *uncertainty
  region* over time rather than the raw feature distribution. We adapt
  this by fitting a sense model on a reference period and monitoring how
  many later records fall *outside* every known sense.
- **Lexical semantic change detection** (Giulianelli et al. 2020;
  Montariol et al. 2021; Schlechtweg et al. 2020 / SemEval-2020 Task 1):
  cluster contextual embeddings into usage types and track the
  distribution over time, quantified with Jensen--Shannon divergence.

Concretely, we:

1. Fit *k* sense centroids on the reference period and estimate a tied
   Gaussian covariance, giving each record a **Mahalanobis** novelty
   score = distance to the nearest sense (Lee et al. 2018's
   out-of-distribution score).
2. For each later period, compute the **margin density** (fraction of
   records whose novelty exceeds a reference-calibrated cutoff) and the
   **Jensen--Shannon divergence** between that period's sense
   distribution --- the *k* known senses plus a single "novel" bin ---
   and the reference distribution.
3. Flag a period as drifting when its margin density exceeds a
   control-chart threshold (reference mean + ``k_sigma`` standard
   deviations), in the spirit of Rabanser et al. (2019)'s two-sample
   shift tests.
4. **Explain** each flagged period by re-clustering its novel records:
   a coherent novel cluster -> *emergence*; high JSD driven by
   re-weighted known senses -> *frequency shift*; diffuse novelty with
   no coherent cluster -> *broadening*.

Embeddings are bring-your-own (never computed internally); clustering
and covariance use ``scikit-learn`` (the ``[semantic]`` extra), imported
lazily.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..results import _table_to_html, _table_to_json

if TYPE_CHECKING:
    import altair as alt

FloatArray = npt.NDArray[np.float64]


def _l2_normalize(x: FloatArray) -> FloatArray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return np.asarray(x / n, dtype=np.float64)


def _fit_reference_model(
    x_ref: FloatArray, k: int, random_state: int
) -> tuple[npt.NDArray[np.intp], FloatArray, FloatArray]:
    """Cluster the reference embeddings; return labels, centroids, and the
    inverse of a tied (shared) covariance for Mahalanobis scoring."""
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(x_ref)
    d = x_ref.shape[1]
    centroids = np.vstack([x_ref[labels == c].mean(axis=0) for c in range(k)])
    # Tied within-class covariance (Lee et al. 2018), pooled across clusters.
    cov = np.zeros((d, d), dtype=np.float64)
    for c in range(k):
        diff = x_ref[labels == c] - centroids[c]
        cov += diff.T @ diff
    cov /= max(x_ref.shape[0], 1)
    cov += 1e-6 * np.eye(d)  # regularise for invertibility
    cov_inv = np.linalg.pinv(cov)
    return (np.asarray(labels, dtype=np.intp),
            np.asarray(centroids, dtype=np.float64),
            np.asarray(cov_inv, dtype=np.float64))


def _mahalanobis_to_nearest(
    x: FloatArray, centroids: FloatArray, cov_inv: FloatArray
) -> tuple[FloatArray, npt.NDArray[np.intp]]:
    """Per-row min Mahalanobis distance to any centroid, and the argmin."""
    k = centroids.shape[0]
    dists = np.empty((x.shape[0], k), dtype=np.float64)
    for c in range(k):
        diff = x - centroids[c]
        dists[:, c] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, cov_inv, diff), 0.0))
    nearest = dists.argmin(axis=1)
    return dists[np.arange(x.shape[0]), nearest], nearest


def _control_threshold(
    ref_vals: list[float], k_sigma: float, *, single_n: int, single_p: float
) -> float:
    """Reference mean + k_sigma standard deviations. With < 2 reference
    periods, use the binomial standard error of the single estimate."""
    if len(ref_vals) >= 2:
        return float(np.mean(ref_vals)) + k_sigma * float(np.std(ref_vals, ddof=1))
    std = float(np.sqrt(max(single_p * (1 - single_p), 1e-9) / max(single_n, 1)))
    return single_p + k_sigma * std


def _jsd(p: FloatArray, q: FloatArray) -> float:
    """Jensen--Shannon divergence (base 2, in [0, 1]) between two
    distributions over the same bins."""
    eps = 1e-12
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a: FloatArray, b: FloatArray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _score_novelty(
    x: FloatArray, ref_mask: npt.NDArray[np.bool_], k: int,
    novelty: str, cutoff_pctile: float, random_state: int,
) -> tuple[npt.NDArray[np.intp], FloatArray, npt.NDArray[np.bool_]]:
    """Fit the reference sense model and score every row: return the nearest
    sense, the novelty score, and a boolean ``is_novel`` (above the
    reference-calibrated cutoff). Called once for the real data and once per
    permutation, so it carries the full out-of-sample bias the null exposes."""
    _, centroids, cov_inv = _fit_reference_model(x[ref_mask], k, random_state)
    if novelty == "mahalanobis":
        nov, nearest = _mahalanobis_to_nearest(x, centroids, cov_inv)
    else:
        sims = x @ centroids.T
        nearest = sims.argmax(axis=1).astype(np.intp)
        nov = 1.0 - sims.max(axis=1)
    cutoff = float(np.percentile(nov[ref_mask], cutoff_pctile))
    return nearest, nov, nov > cutoff


def _period_md_jsd(
    periods_col: npt.NDArray[Any], nearest: npt.NDArray[np.intp],
    is_novel: npt.NDArray[np.bool_], ref_mask: npt.NDArray[np.bool_], k: int,
) -> tuple[list[Any], dict[Any, float], dict[Any, float], FloatArray]:
    """Per-period margin density and Jensen--Shannon divergence from the
    reference sense distribution (numpy-only; no DataFrame, for speed in the
    permutation loop)."""
    def _dist(mask: npt.NDArray[np.bool_]) -> FloatArray:
        counts = np.zeros(k + 1, dtype=np.float64)
        known = mask & ~is_novel
        for c in range(k):
            counts[c] = float(np.sum(known & (nearest == c)))
        counts[k] = float(np.sum(mask & is_novel))
        return counts

    ref_dist = _dist(ref_mask)
    uniq = sorted({p for p in periods_col.tolist() if p == p})  # drop NaN
    md: dict[Any, float] = {}
    jsd: dict[Any, float] = {}
    for p in uniq:
        m = periods_col == p
        md[p] = float(is_novel[m].mean()) if m.any() else 0.0
        jsd[p] = _jsd(_dist(m), ref_dist) if m.any() else 0.0
    return uniq, md, jsd, ref_dist


@dataclass(frozen=True)
class SenseDriftResult:
    """Per-period sense-drift detection with a change-type explanation.

    Attributes
    ----------
    table
        Per-period: ``period``, ``n``, ``margin_density`` (fraction of
        records outside every reference sense), ``jsd`` (Jensen--Shannon
        divergence of the period's sense distribution from the
        reference), and ``drift`` (bool, control-chart flag).
    change_type
        Across the flagged periods, one of ``"emergence"``,
        ``"frequency_shift"``, or ``"broadening"`` --- ``None`` if no
        period drifts.
    drift_terms
        Terms most distinctive (by log-ratio vs the reference) of the
        novel material driving the drift, for *any* change type --- the
        "what" behind the "when".
    reference, k
        The reference period label(s) and number of senses fit there.
    threshold
        The margin-density flag threshold --- null-calibrated (a high
        percentile of the label-shuffle null) when ``n_permutations > 0``,
        otherwise the in-sample control chart.
    p_value
        Permutation p-value for the overall drift (real max margin density
        vs the label-shuffle null max); ``None`` unless
        ``n_permutations > 0``. The in-sample chart over-flags out-of-sample
        periods, so this is the honest significance test.
    """

    table: pd.DataFrame
    change_type: str | None
    drift_terms: list[str]
    reference: list[Any]
    k: int
    threshold: float
    p_value: float | None
    embedding_meta: dict[str, Any]
    _records: pd.DataFrame = field(repr=False)

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_json(self.table, path, **kw)

    def summary(self) -> str:
        flagged = self.table[self.table["drift"]]
        if not len(flagged):
            return (f"No sense drift detected across {len(self.table)} periods "
                    f"(reference {self.reference}, {self.k} senses).")
        first = flagged.iloc[0]
        ct = self.change_type or "drift"
        msg = (f"Sense drift detected from {first['period']} "
               f"({len(flagged)} period(s) flagged); change type: {ct} "
               f"(margin density {first['margin_density']:.3f}, "
               f"JSD {first['jsd']:.3f}).")
        if self.p_value is not None:
            msg += f" Permutation p={self.p_value:.3f}."
        if self.drift_terms:
            msg += " Distinctive terms: " + ", ".join(self.drift_terms[:8]) + "."
        return msg

    def flagged_records(self, period: Any | None = None) -> pd.DataFrame:
        """Novel (uncertainty-region) records driving the drift, for
        inspection. Restrict to one ``period`` or return all flagged."""
        recs = self._records[self._records["novel"]]
        if period is not None:
            recs = recs[recs["period"] == period]
        elif self.table["drift"].any():
            flagged = set(self.table.loc[self.table["drift"], "period"])
            recs = recs[recs["period"].isin(flagged)]
        return recs.reset_index(drop=True)

    def _cluster_terms(self, c: int, top: int = 8) -> list[str]:
        recs = self._records
        in_c = recs.loc[(recs["nearest_sense"] == c) & ~recs["novel"], "text"].tolist()
        out_c = recs.loc[(recs["nearest_sense"] != c) & ~recs["novel"], "text"].tolist()
        return _distinctive_terms(in_c, out_c, top=top)

    def sense_trajectories(self) -> pd.DataFrame:
        """Per-reference-sense prevalence over time. For each period and each
        of the ``k`` reference senses, the count and share of confidently
        assigned (non-novel) records. This is the data behind *decline*: the
        mirror of the novelty signal that drives emergence/broadening."""
        recs = self._records
        periods = sorted({p for p in recs["period"].tolist() if p == p})
        rows = []
        for p in periods:
            sub = recs[recs["period"] == p]
            n_total = len(sub)
            conf = sub[~sub["novel"]]
            for c in range(self.k):
                cnt = int((conf["nearest_sense"] == c).sum())
                rows.append({"period": p, "sense": c, "n": cnt,
                             "share": cnt / n_total if n_total else 0.0})
        return pd.DataFrame(rows)

    def decline_report(
        self, min_share: float = 0.05, rel_change: float = 0.30, late_periods: int = 3,
    ) -> pd.DataFrame:
        """Classify each reference sense's trajectory --- the *fall-off* hunt,
        the mirror of emergence detection.

        A sense whose share drops by at least ``rel_change`` (relative) from
        the reference window to the last ``late_periods`` is *declining*; the
        decline is split into **obsolescence** (its absolute count also falls)
        vs **dilution** (count is stable or rising, share falls only because
        the rest of the corpus grew). Senses below ``min_share`` early are
        marked ``minor`` and not judged. Returns one row per sense with
        early/late share and count, the verdict, and distinctive terms,
        sorted from steepest decline to steepest rise.
        """
        traj = self.sense_trajectories()
        ref_set = set(self.reference)
        periods = sorted(traj["period"].unique())
        ref_periods = [p for p in periods if p in ref_set]
        late = [p for p in periods if p not in ref_set][-late_periods:]
        rows = []
        for c in range(self.k):
            tc = traj[traj["sense"] == c]
            early_share = float(tc[tc["period"].isin(ref_periods)]["share"].mean())
            late_share = float(tc[tc["period"].isin(late)]["share"].mean())
            early_cnt = float(tc[tc["period"].isin(ref_periods)]["n"].mean())
            late_cnt = float(tc[tc["period"].isin(late)]["n"].mean())
            rel = (late_share - early_share) / early_share if early_share > 0 else 0.0
            if early_share < min_share:
                verdict = "minor"
            elif rel <= -rel_change:
                verdict = ("obsolescence" if late_cnt < early_cnt * (1 - rel_change)
                           else "dilution")
            elif rel >= rel_change:
                verdict = "rising"
            else:
                verdict = "stable"
            rows.append({
                "sense": c, "early_share": early_share, "late_share": late_share,
                "early_count": early_cnt, "late_count": late_cnt,
                "rel_share_change": rel, "verdict": verdict,
                "terms": ", ".join(self._cluster_terms(c)),
            })
        return (pd.DataFrame(rows)
                .sort_values("rel_share_change")
                .reset_index(drop=True))

    def plot(self, **kw: Any) -> alt.Chart:
        """Margin density and JSD over time, with drift-flagged periods."""
        import altair as alt

        t = self.table.assign(period_str=lambda d: d["period"].astype(str))
        base = alt.Chart(t).encode(x=alt.X("period_str:O", title="period"))
        md = base.mark_line(point=True, color="#e76f51").encode(
            y=alt.Y("margin_density:Q", title="margin density"))
        flags = base.transform_filter(alt.datum.drift).mark_point(
            size=120, color="#d00", shape="triangle-up").encode(y="margin_density:Q")
        return (md + flags).properties(**kw)  # type: ignore[no-any-return]


def sense_drift(
    items: pd.DataFrame,
    embeddings: FloatArray,
    time_col: str,
    *,
    reference: Any | Sequence[Any],
    k: int = 3,
    novelty: str = "mahalanobis",
    cutoff_pctile: float = 95.0,
    k_sigma: float = 3.0,
    min_run: int = 2,
    n_permutations: int = 0,
    null_pctile: float = 95.0,
    normalize: bool = True,
    random_state: int = 42,
    text_col: str = "text",
    embedding_meta: dict[str, Any] | None = None,
) -> SenseDriftResult:
    """Detect and explain drift in a corpus's sense distribution over time.

    Parameters
    ----------
    items
        DataFrame with a time column and (optionally) ``text_col`` for
        cluster labelling. Row order aligns with ``embeddings``.
    embeddings
        Bring-your-own ``(n, d)`` matrix.
    time_col
        Column holding the period (e.g. an integer year). Periods are
        used verbatim and sorted; bin a date column yourself first if
        needed.
    reference
        Period label or list of labels whose records define the *known*
        senses. The drift of every period is measured against this.
    k
        Number of senses to fit on the reference period.
    novelty
        ``"mahalanobis"`` (Lee et al. 2018; recommended) or ``"cosine"``
        (1 - max centroid cosine similarity).
    cutoff_pctile
        A record is *novel* (in the uncertainty region) if its novelty
        score exceeds this percentile of the reference records' scores.
    k_sigma
        In-sample control-chart sensitivity (used only when
        ``n_permutations == 0``): a period drifts if its margin density
        exceeds the reference mean by ``k_sigma`` standard deviations.
    n_permutations
        If ``> 0``, calibrate the flag threshold against a label-shuffle
        null of this many permutations and report a permutation
        ``p_value``. **Recommended for inference**: the in-sample control
        chart over-flags because out-of-sample periods look novel relative
        to a reference fitted on themselves; the shuffle null removes that
        bias. Costs one model re-fit per permutation. ``0`` (default) uses
        the fast in-sample chart, fine for exploration.
    null_pctile
        Percentile of the label-shuffle null margin-density (and JSD)
        distribution used as the flag threshold when ``n_permutations > 0``.
    normalize
        L2-normalise embeddings before fitting (default ``True``).

    Returns
    -------
    SenseDriftResult
    """
    frame = items.reset_index(drop=True)
    x = np.asarray(embeddings, dtype=np.float64)
    if x.shape[0] != len(frame):
        raise ValueError(
            f"embeddings has {x.shape[0]} rows but items has {len(frame)}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or inf")
    if time_col not in frame.columns:
        raise ValueError(f"time_col {time_col!r} not in items")
    if novelty not in {"mahalanobis", "cosine"}:
        raise ValueError("novelty must be 'mahalanobis' or 'cosine'")
    if normalize:
        x = _l2_normalize(x)

    ref_labels_set = list(reference) if isinstance(reference, (list, tuple, set)) else [reference]
    periods_col = frame[time_col].to_numpy()
    ref_mask = frame[time_col].isin(ref_labels_set).to_numpy()
    if ref_mask.sum() < k * 5:
        raise ValueError(
            f"reference period has only {int(ref_mask.sum())} records; "
            f"need >= {k * 5} for k={k} senses")

    nearest, nov, is_novel = _score_novelty(
        x, ref_mask, k, novelty, cutoff_pctile, random_state)
    periods, md_real, jsd_real, ref_dist = _period_md_jsd(
        periods_col, nearest, is_novel, ref_mask, k)
    not_ref_periods = [p for p in periods if p not in ref_labels_set]

    # --- Flag thresholds ---------------------------------------------------
    # Margin density catches novelty-driven drift (emergence, broadening);
    # JSD also catches a re-weighting of known senses (frequency shift), so a
    # period drifts if *either* exceeds its threshold.
    p_value: float | None = None
    if n_permutations > 0:
        # Null-calibrated thresholds. The in-sample control chart is biased:
        # out-of-sample periods look novel relative to a reference fitted on
        # themselves, so it over-flags. Permuting the period labels destroys
        # the temporal structure but preserves that bias, giving the correct
        # null; a period drifts only if it beats a high percentile of it. The
        # permutation p-value compares the real maximum margin density against
        # the per-shuffle null maxima.
        rng = np.random.default_rng(random_state)
        md_pool: list[float] = []
        jsd_pool: list[float] = []
        md_maxes: list[float] = []
        for _ in range(n_permutations):
            perm = rng.permutation(periods_col)
            pm = np.isin(perm, ref_labels_set)
            if pm.sum() < k * 5:
                continue
            n2, _, novel2 = _score_novelty(
                x, pm, k, novelty, cutoff_pctile, random_state)
            _, md_b, jsd_b, _ = _period_md_jsd(perm, n2, novel2, pm, k)
            vals = [md_b[p] for p in md_b if p not in ref_labels_set]
            md_pool.extend(vals)
            jsd_pool.extend(jsd_b[p] for p in jsd_b if p not in ref_labels_set)
            if vals:
                md_maxes.append(max(vals))
        md_threshold = float(np.percentile(md_pool, null_pctile)) if md_pool else np.inf
        jsd_threshold = float(np.percentile(jsd_pool, null_pctile)) if jsd_pool else np.inf
        real_max = max((md_real[p] for p in not_ref_periods), default=0.0)
        if md_maxes:
            p_value = (int(np.sum(np.asarray(md_maxes) >= real_max)) + 1) / (len(md_maxes) + 1)
    else:
        md_ref = [md_real[p] for p in periods if p in ref_labels_set]
        jsd_ref = [jsd_real[p] for p in periods if p in ref_labels_set]
        md_threshold = _control_threshold(
            md_ref, k_sigma, single_n=int(ref_mask.sum()),
            single_p=float(is_novel[ref_mask].mean()))
        jsd_threshold = (
            float(np.mean(jsd_ref)) + k_sigma * float(np.std(jsd_ref, ddof=1))
            if len(jsd_ref) >= 2 else np.inf)
    threshold = md_threshold

    table = pd.DataFrame({
        "period": periods,
        "n": [int((periods_col == p).sum()) for p in periods],
        "margin_density": [md_real[p] for p in periods],
        "jsd": [jsd_real[p] for p in periods],
    })
    not_ref = ~table["period"].isin(ref_labels_set)
    raw = (((table["margin_density"] > md_threshold)
            | (table["jsd"] > jsd_threshold)) & not_ref).to_numpy()

    # Sustained-run confirmation: a period counts as drift only if it is part
    # of a run of >= min_run consecutive raw exceedances. Isolated single-
    # period spikes are treated as noise (false-alarm control).
    confirmed = np.zeros(len(raw), dtype=bool)
    i = 0
    while i < len(raw):
        if raw[i]:
            j = i
            while j < len(raw) and raw[j]:
                j += 1
            if j - i >= min_run:
                confirmed[i:j] = True
            i = j
        else:
            i += 1
    table["drift"] = confirmed

    # Records (for the explanation layer and .flagged_records()).
    texts = ([str(v) for v in frame[text_col]] if text_col in frame.columns
             else [""] * len(frame))
    recs = pd.DataFrame({
        "period": periods_col,
        "nearest_sense": nearest,
        "novelty": nov,
        "novel": is_novel,
        "text": texts,
    })

    # --- Explanation layer -------------------------------------------------
    change_type: str | None = None
    drift_terms: list[str] = []
    flagged_periods = list(table.loc[table["drift"], "period"])
    if flagged_periods:
        flag_mask = np.isin(periods_col, flagged_periods) & is_novel
        novel_emb = x[flag_mask]
        ref_texts = [texts[i] for i in range(len(texts)) if ref_mask[i]]
        novel_texts = [texts[i] for i in np.nonzero(flag_mask)[0]]
        fmask = np.isin(periods_col, flagged_periods)
        drift_dist = np.zeros(k + 1, dtype=np.float64)
        known = fmask & ~is_novel
        for c in range(k):
            drift_dist[c] = float(np.sum(known & (nearest == c)))
        drift_dist[k] = float(np.sum(fmask & is_novel))
        change_type, drift_terms = _explain(
            novel_texts, ref_texts, novel_emb, ref_dist, drift_dist, k)

    return SenseDriftResult(
        table=table,
        change_type=change_type,
        drift_terms=drift_terms,
        reference=ref_labels_set,
        k=k,
        threshold=threshold,
        p_value=p_value,
        embedding_meta=dict(embedding_meta or {}),
        _records=recs,
    )


_EXPLAIN_STOP = frozenset(
    """the and for with this that are was from which were have has had been
    they their them then than into over under between about can will would
    these those such using used study studies results methods method data
    analysis effects effect compared associated significant however other
    most more high also after both there when within across among may might
    each our your""".split()  # noqa: SIM905 - readable stop list
)
_EXPLAIN_WORD = re.compile(r"[a-z][a-z-]{3,}")


def _distinctive_terms(
    target_texts: list[str], other_texts: list[str], top: int = 12, min_df: int = 3,
) -> list[str]:
    """Terms most distinctive of ``target_texts`` vs ``other_texts`` by
    log-ratio of document frequency (a keyness-style score; words common to
    both wash out because their ratio is ~1)."""
    def _docfreq(texts: list[str]) -> tuple[dict[str, int], int]:
        cnt: dict[str, int] = {}
        for t in texts:
            for w in set(_EXPLAIN_WORD.findall(str(t).lower())):
                if w not in _EXPLAIN_STOP:
                    cnt[w] = cnt.get(w, 0) + 1
        return cnt, max(len(texts), 1)

    tc, tn = _docfreq(target_texts)
    oc, on = _docfreq(other_texts)
    scored = []
    for w, cn in tc.items():
        if cn < min_df:
            continue
        target_rate = cn / tn
        other_rate = (oc.get(w, 0) + 0.5) / on
        scored.append((w, float(np.log2(target_rate / other_rate))))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [w for w, _ in scored[:top]]


def _explain(
    novel_texts: list[str],
    ref_texts: list[str],
    novel_emb: FloatArray,
    ref_dist: FloatArray,
    period_dist: FloatArray,
    k: int,
) -> tuple[str, list[str]]:
    """Classify the drift's change type and always surface the terms most
    distinctive of the novel material (log-ratio vs the reference)."""
    eps = 1e-12
    rp = ref_dist / (ref_dist.sum() + eps)
    pp = period_dist / (period_dist.sum() + eps)
    novel_share_growth = pp[k] - rp[k]
    known_reweight = float(np.abs(pp[:k] - rp[:k]).sum())

    # Coherence: is the novel material one tight cluster (emergence) or
    # diffuse (broadening)? Mean cosine of each novel vector to the novel
    # centroid (embeddings are L2-normalised, so dot == cosine).
    coherent = False
    if len(novel_emb) >= 5:
        c = novel_emb.mean(axis=0)
        c = c / (np.linalg.norm(c) + eps)
        coherent = float((novel_emb @ c).mean()) >= 0.5

    if novel_share_growth >= 0.05 and coherent:
        change_type = "emergence"
    elif known_reweight >= 0.10 and novel_share_growth < 0.05:
        change_type = "frequency_shift"
    else:
        change_type = "broadening"

    # Distinctive terms via log-ratio of novel vs reference document
    # frequency (a keyness-style score).
    return change_type, _distinctive_terms(novel_texts, ref_texts, top=12)
