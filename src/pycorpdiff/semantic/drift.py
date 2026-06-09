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
    """

    table: pd.DataFrame
    change_type: str | None
    drift_terms: list[str]
    reference: list[Any]
    k: int
    threshold: float
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
        Control-chart sensitivity: a period drifts if its margin density
        exceeds the reference mean by ``k_sigma`` standard deviations.
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
    periods = sorted(frame[time_col].dropna().unique())
    ref_mask = frame[time_col].isin(ref_labels_set).to_numpy()
    if ref_mask.sum() < k * 5:
        raise ValueError(
            f"reference period has only {int(ref_mask.sum())} records; "
            f"need >= {k * 5} for k={k} senses")

    x_ref = x[ref_mask]
    ref_labels, centroids, cov_inv = _fit_reference_model(x_ref, k, random_state)

    # Novelty score for every record.
    if novelty == "mahalanobis":
        nov, nearest = _mahalanobis_to_nearest(x, centroids, cov_inv)
    else:
        sims = x @ centroids.T
        nearest = sims.argmax(axis=1)
        nov = 1.0 - sims.max(axis=1)

    cutoff = float(np.percentile(nov[ref_mask], cutoff_pctile))
    is_novel = nov > cutoff

    texts = ([str(v) for v in frame[text_col]] if text_col in frame.columns
             else [""] * len(frame))
    recs = pd.DataFrame({
        "period": frame[time_col].to_numpy(),
        "nearest_sense": nearest,
        "novelty": nov,
        "novel": is_novel,
        "text": texts,
    })

    # Reference sense distribution: k known senses + a novel bin.
    def _dist(sub: pd.DataFrame) -> FloatArray:
        counts = np.zeros(k + 1, dtype=np.float64)
        known = sub[~sub["novel"]]
        for c in range(k):
            counts[c] = int((known["nearest_sense"] == c).sum())
        counts[k] = int(sub["novel"].sum())
        return counts

    ref_dist = _dist(recs[recs["period"].isin(ref_labels_set)])

    rows = []
    md_ref: list[float] = []
    jsd_ref: list[float] = []
    for p in periods:
        sub = recs[recs["period"] == p]
        n = len(sub)
        md = float(sub["novel"].mean()) if n else 0.0
        jsd = _jsd(_dist(sub), ref_dist) if n else 0.0
        rows.append({"period": p, "n": n, "margin_density": md, "jsd": jsd})
        if p in ref_labels_set:
            md_ref.append(md)
            jsd_ref.append(jsd)

    # Control-chart thresholds from the reference periods. Margin density
    # catches novelty-driven drift (emergence, broadening); JSD also
    # catches a re-weighting of known senses (frequency shift), so a
    # period drifts if *either* exceeds its threshold. A reference
    # spanning >= 2 periods gives the JSD chart its variance; with a
    # single reference period we fall back to the margin-density binomial.
    md_threshold = _control_threshold(
        md_ref, k_sigma,
        single_n=int(ref_mask.sum()),
        single_p=float(recs.loc[recs["period"].isin(ref_labels_set), "novel"].mean()),
    )
    table = pd.DataFrame(rows)
    not_ref = ~table["period"].isin(ref_labels_set)
    raw = ((table["margin_density"] > md_threshold) & not_ref).to_numpy()
    threshold = md_threshold
    if len(jsd_ref) >= 2:
        jsd_threshold = float(np.mean(jsd_ref)) + k_sigma * float(np.std(jsd_ref, ddof=1))
        raw = raw | ((table["jsd"] > jsd_threshold) & not_ref).to_numpy()
    # Sustained-run confirmation: a period counts as drift only if it is
    # part of a run of >= min_run consecutive raw exceedances. Isolated
    # single-period spikes are treated as noise (false-alarm control).
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

    # --- Explanation layer -------------------------------------------------
    change_type: str | None = None
    drift_terms: list[str] = []
    flagged_periods = list(table.loc[table["drift"], "period"])
    if flagged_periods:
        flag_mask = (recs["period"].isin(flagged_periods) & recs["novel"]).to_numpy()
        novel_emb = x[flag_mask]
        ref_texts = recs.loc[recs["period"].isin(ref_labels_set), "text"].tolist()
        drift_dist = _dist(recs[recs["period"].isin(flagged_periods)])
        change_type, drift_terms = _explain(
            recs.loc[flag_mask, "text"].tolist(), ref_texts, novel_emb,
            ref_dist, drift_dist, k)

    return SenseDriftResult(
        table=table,
        change_type=change_type,
        drift_terms=drift_terms,
        reference=ref_labels_set,
        k=k,
        threshold=threshold,
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
    # frequency (a keyness-style score; generic words common to both wash
    # out because their ratio is ~1).
    def _docfreq(texts: list[str]) -> tuple[dict[str, int], int]:
        cnt: dict[str, int] = {}
        for t in texts:
            for w in set(_EXPLAIN_WORD.findall(str(t).lower())):
                if w not in _EXPLAIN_STOP:
                    cnt[w] = cnt.get(w, 0) + 1
        return cnt, max(len(texts), 1)

    nc, nn = _docfreq(novel_texts)
    rc, rn = _docfreq(ref_texts)
    scored = []
    for w, cn in nc.items():
        if cn < 3:
            continue
        novel_rate = cn / nn
        ref_rate = (rc.get(w, 0) + 0.5) / rn
        scored.append((w, float(np.log2(novel_rate / ref_rate))))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    terms = [w for w, _ in scored[:12]]
    return change_type, terms
