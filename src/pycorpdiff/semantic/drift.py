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

    from ..annotate import Annotator, SenseNamingResult

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

    def name_senses(
        self,
        annotator: Annotator,
        *,
        n_examples: int = 8,
        include_novel: bool = True,
        cache: dict[str, str] | None = None,
    ) -> SenseNamingResult:
        """Attach human-readable labels + glosses to each fitted sense (and the
        emergent bin) with an LLM :class:`~pycorpdiff.Annotator`, grounded in
        *cited* exemplars.

        The annotator only **names**: it is handed this result's own most-central
        exemplars per sense (lowest-novelty non-novel records) plus their
        distinctive terms, and returns a label and a one-line gloss. Output comes
        back in a *separate* :class:`~pycorpdiff.SenseNamingResult` and never flows
        into this result's numeric table --- the honest division of labour
        (vectors and counts quantify; the LLM interprets; never the reverse). The
        veracity of any sense is not asked or answered here.

        Parameters
        ----------
        annotator
            Anything satisfying the :class:`~pycorpdiff.Annotator` protocol
            (e.g. :class:`~pycorpdiff.OllamaAnnotator`, or
            :class:`~pycorpdiff.EchoAnnotator` for offline use).
        n_examples
            Number of cited exemplars shown to the model per sense.
        include_novel
            Also name the residual *novel / emergent* bin (the material driving
            emergence/broadening), using :attr:`drift_terms`.
        cache
            Optional dict reused across calls to avoid re-querying identical
            prompts (keyed by ``model_id`` + prompt hash). Updated in place.

        Returns
        -------
        SenseNamingResult
        """
        from ..annotate import _name_senses

        return _name_senses(self, annotator, n_examples=n_examples,
                            include_novel=include_novel, cache=cache)

    def _sense_labels(self, top: int = 2) -> dict[int, str]:
        return {c: ", ".join(self._cluster_terms(c, top=top)) or f"sense {c}"
                for c in range(self.k)}

    def plot(self, **kw: Any) -> alt.Chart:
        """Margin density over time with the flag threshold and drift markers.

        The dashed rule is the calibrated threshold (a high percentile of the
        label-shuffle null when ``n_permutations > 0``); periods above it that
        survive the sustained-run check are flagged in red. Makes the
        significance visible, not just tabulated."""
        import altair as alt

        t = self.table.assign(period_str=lambda d: d["period"].astype(str))
        base = alt.Chart(t).encode(x=alt.X("period_str:O", title="period"))
        area = base.mark_area(opacity=0.25, color="#e76f51").encode(
            y=alt.Y("margin_density:Q", title="margin density"))
        line = base.mark_line(color="#e76f51", point=True).encode(y="margin_density:Q")
        rule = (alt.Chart(pd.DataFrame({"y": [self.threshold]}))
                .mark_rule(strokeDash=[5, 4], color="#444")
                .encode(y="y:Q"))
        flags = base.transform_filter(alt.datum.drift).mark_point(
            size=140, color="#d00", shape="triangle-up", filled=True).encode(
            y="margin_density:Q")
        sub = (f"permutation p = {self.p_value:.3f}; " if self.p_value is not None else "")
        title = alt.TitleParams(
            "Sense-drift margin density", subtitle=f"{sub}change type: {self.change_type}")
        return (area + line + rule + flags).properties(title=title, **kw)  # type: ignore[no-any-return]

    def plot_composition(self, **kw: Any) -> alt.Chart:
        """Stacked-area sense composition over time --- the headline drift
        figure. Each reference sense is a band labelled by its distinctive
        terms; the residual ``novel / emergent`` band is the uncertainty
        region that drives emergence and broadening. The takeover (one band
        swelling while others shrink) is visible at a glance."""
        import altair as alt

        traj = self.sense_trajectories()
        per = traj.groupby("period")["share"].sum().reset_index(name="known")
        novel = per.assign(sense=self.k, share=(1.0 - per["known"]).clip(lower=0.0))
        full = pd.concat([traj[["period", "sense", "share"]],
                          novel[["period", "sense", "share"]]], ignore_index=True)
        labels = self._sense_labels()
        labels[self.k] = "novel / emergent"
        full = full.assign(
            sense_label=full["sense"].map(labels),
            period_str=full["period"].astype(str))
        order = [labels[c] for c in range(self.k)] + [labels[self.k]]
        chart = alt.Chart(full).mark_area().encode(
            x=alt.X("period_str:O", title="period"),
            y=alt.Y("share:Q", stack="normalize", title="sense share"),
            color=alt.Color("sense_label:N", title="sense (top terms)",
                            sort=order, scale=alt.Scale(scheme="tableau10")),
            order=alt.Order("sense:Q"),
            tooltip=["period_str", "sense_label", alt.Tooltip("share:Q", format=".2f")],
        ).properties(title="Sense composition over time", **kw)
        return chart  # type: ignore[no-any-return]

    def plot_decline(self, **kw: Any) -> alt.Chart:
        """Slopegraph of each reference sense's share from the reference window
        to the late period, coloured by verdict (obsolescence / dilution /
        rising / stable). The fall-off, seen: obsolescent senses slope down in
        red, rising ones up in green."""
        import altair as alt

        rep = self.decline_report()
        labels = self._sense_labels()
        long = pd.concat([
            rep.assign(when="early", share=rep["early_share"]),
            rep.assign(when="late", share=rep["late_share"]),
        ], ignore_index=True)
        long = long.assign(sense_label=long["sense"].map(labels))
        scale = alt.Scale(
            domain=["obsolescence", "dilution", "rising", "stable", "minor"],
            range=["#d62728", "#ff9896", "#2ca02c", "#999999", "#dddddd"])
        base = alt.Chart(long).encode(
            x=alt.X("when:O", sort=["early", "late"], title=None),
            y=alt.Y("share:Q", title="sense share"),
            color=alt.Color("verdict:N", scale=scale, title="verdict"),
            detail="sense:N",
            tooltip=["sense_label", "verdict",
                     alt.Tooltip("early_share:Q", format=".2f"),
                     alt.Tooltip("late_share:Q", format=".2f")])
        chart = base.mark_line(point=True, strokeWidth=3).properties(
            title="Sense decline (early to late share)", **kw)
        return chart  # type: ignore[no-any-return]


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
    background_embeddings: FloatArray | None = None,
    background_time: Sequence[Any] | None = None,
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
        Choose by *regime* (this is a real trade-off, not a strict ranking):

        - ``"mahalanobis"`` (Lee et al. 2018; **default**) — sensitive to
          genuinely novel senses, which drives emergence/broadening detection.
          Best for **register-stable diachronic monitoring** (one corpus over
          time), the typical use case. On CBD-in-PubMed it recovers the
          broadening and its drivers (dravet, clobazam, legalization).
        - ``"cosine"`` (1 - max centroid cosine similarity) — a **bounded**
          alternative for **cross-register / cross-era** corpora where the
          *whole* embedding distribution shifts (archaic vs modern text, one
          platform vs another). Mahalanobis is designed to amplify distribution
          shift, so it **saturates** there (every later record looks novel and
          margin density pins near 1.0); cosine still ranks. Validated on
          SemEval-2020 LSCD (cosine recovers signal Mahalanobis saturates on),
          but on register-stable CBD it under-detects the broadening Mahalanobis
          catches.
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
    background_embeddings, background_time
        Optional **nuisance-drift correction**. Pass an ``(m, d)`` matrix of
        control/background records (e.g. a random non-target sample from the
        same corpus) and their parallel period labels. Each non-reference
        period's mean background shift from the reference is subtracted from
        that period's records before scoring, so drift is measured *beyond* the
        corpus-wide drift. Useful for cross-register/era corpora where the whole
        embedding cloud shifts; a near no-op on register-stable corpora.

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
    ref_labels_set = list(reference) if isinstance(reference, (list, tuple, set)) else [reference]
    periods_col = frame[time_col].to_numpy()
    ref_mask = frame[time_col].isin(ref_labels_set).to_numpy()
    if ref_mask.sum() < k * 5:
        raise ValueError(
            f"reference period has only {int(ref_mask.sum())} records; "
            f"need >= {k * 5} for k={k} senses")

    if background_embeddings is not None:
        # M1 nuisance-drift correction: subtract each period's mean shift of a
        # background/control corpus (e.g. random non-target records over the
        # same periods) so drift is measured *beyond* the corpus-wide drift.
        # For cross-register/era data where the whole cloud shifts (see the
        # ``novelty`` note); a no-op stabiliser on register-stable corpora.
        if background_time is None:
            raise ValueError(
                "background_time is required when background_embeddings is given")
        bg = np.asarray(background_embeddings, dtype=np.float64)  # raw space, like x
        bg_periods = np.asarray(list(background_time))
        if bg.shape[0] != bg_periods.shape[0]:
            raise ValueError(
                f"background_embeddings has {bg.shape[0]} rows but "
                f"background_time has {bg_periods.shape[0]}")
        bg_ref = bg[np.isin(bg_periods, ref_labels_set)]
        if bg_ref.shape[0] == 0:
            raise ValueError("no background records in the reference period(s)")
        bg_ref_mean = bg_ref.mean(axis=0)
        x = x.copy()
        for p in np.unique(periods_col):
            if p in ref_labels_set:
                continue
            bp = bg[bg_periods == p]
            if bp.shape[0] == 0:
                continue
            x[periods_col == p] -= bp.mean(axis=0) - bg_ref_mean

    if normalize:
        x = _l2_normalize(x)

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


# =====================================================================
# k-NN-density drift --- a sense-free, streaming-capable sibling detector
# =====================================================================
# Where :func:`sense_drift` fits *k* reference senses and watches the density of
# the region *outside* them (margin density, after Sethi & Kantardzic 2017),
# :func:`knn_density_drift` drops the sense model entirely: a record is novel if
# it sits far from its nearest neighbours *in the past*. Identical control-chart
# flagging and sustained-run logic, so the two are directly comparable (running
# both and checking they agree is a robustness test). Crucially this is the
# formulation a vector store serves natively (time-filtered nearest-neighbour),
# so ``mode="cumulative"`` is incremental/streaming-ready for an online monitor.


def _knn_density_novelty(
    x_query: FloatArray, x_prior: FloatArray, k: int,
    self_cols: npt.NDArray[np.intp] | None = None,
) -> FloatArray:
    """``1 - mean cosine`` to the ``k`` nearest *prior* embeddings (rows are
    L2-normalised, so dot == cosine). A record in empty space --- no close prior
    neighbour --- scores high. If the query rows are a subset of the prior, pass
    ``self_cols`` (each query row's column within the prior) to drop the
    self-match. This is exactly the retrieval-density a vector DB returns."""
    n_prior = x_prior.shape[0]
    if n_prior == 0:
        return np.zeros(x_query.shape[0], dtype=np.float64)
    sims = x_query @ x_prior.T
    if self_cols is not None:
        sims[np.arange(x_query.shape[0]), self_cols] = -np.inf
    prior_eff = n_prior - (1 if self_cols is not None else 0)
    kk = max(1, min(k, prior_eff))
    idx = np.argpartition(sims, -kk, axis=1)[:, -kk:]
    topk = np.take_along_axis(sims, idx, axis=1)
    return np.asarray(1.0 - topk.mean(axis=1), dtype=np.float64)


def _knn_novelty_all(
    x: FloatArray, periods_col: npt.NDArray[Any], ref_mask: npt.NDArray[np.bool_],
    k: int, mode: str,
) -> tuple[FloatArray, FloatArray]:
    """Per-record k-NN novelty + the reference-internal novelty (the calibration
    baseline). ``reference`` mode scores every record against the fixed reference
    set; ``cumulative`` mode scores each record against all *strictly earlier*
    periods --- the streaming / emergence formulation."""
    nov = np.zeros(x.shape[0], dtype=np.float64)
    n_ref = int(ref_mask.sum())
    nov_ref = _knn_density_novelty(
        x[ref_mask], x[ref_mask], k,
        self_cols=np.arange(n_ref, dtype=np.intp))
    if mode == "reference":
        nov[ref_mask] = nov_ref
        rest = ~ref_mask
        if rest.any():
            nov[rest] = _knn_density_novelty(x[rest], x[ref_mask], k)
    else:  # cumulative: novelty vs all strictly-earlier periods
        order = sorted({p for p in periods_col.tolist() if p == p})
        for p in order:
            qm = periods_col == p
            earlier = [q for q in order if q < p]
            prior = x[np.isin(periods_col, earlier)] if earlier else x[:0]
            nov[qm] = _knn_density_novelty(x[qm], prior, k)
    return nov, nov_ref


def _density_by_period(
    periods_col: npt.NDArray[Any], is_novel: npt.NDArray[np.bool_],
) -> tuple[list[Any], dict[Any, float]]:
    uniq = sorted({p for p in periods_col.tolist() if p == p})
    return uniq, {p: (float(is_novel[periods_col == p].mean())
                      if (periods_col == p).any() else 0.0) for p in uniq}


@dataclass(frozen=True)
class KNNDensityDriftResult:
    """Per-period k-NN-density drift detection (sense-free, streaming-capable).

    Attributes
    ----------
    table
        Per-period: ``period``, ``n``, ``novelty_density`` (fraction of records
        whose distance to their nearest prior neighbours exceeds the
        reference-calibrated cutoff), and ``drift`` (bool, control-chart flag).
    mode
        ``"reference"`` (novelty vs the fixed reference window) or
        ``"cumulative"`` (novelty vs all strictly-earlier periods --- the
        streaming / emergence formulation).
    reference, k
        Reference period label(s) and the number of nearest neighbours.
    threshold
        Novelty-density flag threshold (null-calibrated when
        ``n_permutations > 0``, else the in-sample control chart).
    p_value
        Permutation p-value (``None`` unless ``n_permutations > 0``;
        ``reference`` mode only).
    """

    table: pd.DataFrame
    mode: str
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
            return (f"No k-NN-density drift detected across {len(self.table)} "
                    f"periods (reference {self.reference}, k={self.k}, mode={self.mode}).")
        first = flagged.iloc[0]
        msg = (f"k-NN-density drift detected from {first['period']} "
               f"({len(flagged)} period(s) flagged; novelty density "
               f"{first['novelty_density']:.3f}, mode={self.mode}).")
        if self.p_value is not None:
            msg += f" Permutation p={self.p_value:.3f}."
        return msg

    def flagged_records(self, period: Any | None = None) -> pd.DataFrame:
        """Novel records in the flagged periods (or one ``period``), for
        inspection --- the emergent material driving the drift."""
        recs = self._records[self._records["novel"]]
        if period is not None:
            recs = recs[recs["period"] == period]
        elif self.table["drift"].any():
            flagged = set(self.table.loc[self.table["drift"], "period"])
            recs = recs[recs["period"].isin(flagged)]
        return recs.reset_index(drop=True)

    def exemplars(self, period: Any | None = None, top: int = 8) -> pd.DataFrame:
        """The ``top`` *most novel* records (highest distance from their prior
        neighbours) --- cited exemplars of what newly entered the space."""
        recs = self.flagged_records(period)
        return recs.sort_values("novelty", ascending=False).head(top).reset_index(drop=True)

    def plot(self, **kw: Any) -> alt.Chart:
        """Novelty density over time with the flag threshold and drift markers."""
        import altair as alt

        t = self.table.assign(period_str=lambda d: d["period"].astype(str))
        base = alt.Chart(t).encode(x=alt.X("period_str:O", title="period"))
        area = base.mark_area(opacity=0.25, color="#b91c1c").encode(
            y=alt.Y("novelty_density:Q", title="novelty density"))
        line = base.mark_line(color="#b91c1c", point=True).encode(y="novelty_density:Q")
        rule = (alt.Chart(pd.DataFrame({"y": [self.threshold]}))
                .mark_rule(strokeDash=[5, 4], color="#444").encode(y="y:Q"))
        flags = base.transform_filter(alt.datum.drift).mark_point(
            size=140, color="#d00", shape="triangle-up", filled=True).encode(
            y="novelty_density:Q")
        sub = (f"permutation p = {self.p_value:.3f}; " if self.p_value is not None else "")
        title = alt.TitleParams("k-NN-density drift", subtitle=f"{sub}mode: {self.mode}")
        return (area + line + rule + flags).properties(title=title, **kw)  # type: ignore[no-any-return]


def knn_density_drift(
    items: pd.DataFrame,
    embeddings: FloatArray,
    time_col: str,
    *,
    reference: Any | Sequence[Any],
    k: int = 10,
    cutoff_pctile: float = 95.0,
    k_sigma: float = 3.0,
    min_run: int = 2,
    mode: str = "reference",
    n_permutations: int = 0,
    null_pctile: float = 95.0,
    normalize: bool = True,
    random_state: int = 42,
    text_col: str = "text",
    embedding_meta: dict[str, Any] | None = None,
) -> KNNDensityDriftResult:
    """Detect drift as a rise in **k-nearest-neighbour novelty density** --- the
    sense-free, streaming-capable sibling of :func:`sense_drift`.

    A record is *novel* when it sits far (``1 - mean cosine`` to its ``k`` nearest
    prior neighbours) from what came before; a period drifts when its fraction of
    novel records clears a reference-calibrated control-chart threshold. No sense
    model is fit, so it is robust to the cross-era saturation that pins
    Mahalanobis margin density near 1.0, and it maps one-to-one onto a vector
    store's time-filtered nearest-neighbour query (so ``mode="cumulative"`` is an
    online/streaming monitor). Sharing :func:`sense_drift`'s flagging means the
    two can be cross-checked: agreement across both formulations is a robustness
    result, not a coincidence.

    Parameters
    ----------
    items, embeddings, time_col, reference
        As in :func:`sense_drift`. Row order of ``items`` aligns with
        ``embeddings``; ``reference`` defines the known/baseline period(s).
    k
        Number of nearest neighbours for the novelty score (default 10).
    cutoff_pctile
        A record is novel if its novelty exceeds this percentile of the
        reference records' *internal* novelty.
    k_sigma, min_run
        In-sample control-chart sensitivity and the sustained-run length a
        flag must persist for (false-alarm control), as in :func:`sense_drift`.
    mode
        ``"reference"`` (novelty vs the fixed reference window; **default**) or
        ``"cumulative"`` (novelty vs all strictly-earlier periods --- the
        streaming / emergence formulation; a vector-DB serves this incrementally).
    n_permutations, null_pctile
        Optional label-shuffle null and its threshold percentile, reporting a
        permutation ``p_value`` (``reference`` mode only).
    normalize, random_state, text_col, embedding_meta
        As in :func:`sense_drift`.

    Returns
    -------
    KNNDensityDriftResult
    """
    frame = items.reset_index(drop=True)
    x = np.asarray(embeddings, dtype=np.float64)
    if x.shape[0] != len(frame):
        raise ValueError(f"embeddings has {x.shape[0]} rows but items has {len(frame)}")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or inf")
    if time_col not in frame.columns:
        raise ValueError(f"time_col {time_col!r} not in items")
    if mode not in {"reference", "cumulative"}:
        raise ValueError("mode must be 'reference' or 'cumulative'")
    if mode == "cumulative" and n_permutations > 0:
        raise ValueError("permutation null is only defined for mode='reference'")
    ref_labels_set = list(reference) if isinstance(reference, (list, tuple, set)) else [reference]
    periods_col = frame[time_col].to_numpy()
    ref_mask = frame[time_col].isin(ref_labels_set).to_numpy()
    need = max(k + 1, 5)
    if ref_mask.sum() < need:
        raise ValueError(
            f"reference period has only {int(ref_mask.sum())} records; "
            f"need >= {need} for k={k} neighbours")

    if normalize:
        x = _l2_normalize(x)

    nov, nov_ref = _knn_novelty_all(x, periods_col, ref_mask, k, mode)
    cutoff = float(np.percentile(nov_ref, cutoff_pctile))
    is_novel = nov > cutoff
    periods, dens = _density_by_period(periods_col, is_novel)
    not_ref_periods = [p for p in periods if p not in ref_labels_set]

    p_value: float | None = None
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        pool: list[float] = []
        maxes: list[float] = []
        for _ in range(n_permutations):
            perm = rng.permutation(periods_col)
            pm = np.isin(perm, ref_labels_set)
            if pm.sum() < need:
                continue
            nov_b, nov_ref_b = _knn_novelty_all(x, perm, pm, k, "reference")
            isn_b = nov_b > float(np.percentile(nov_ref_b, cutoff_pctile))
            _, dens_b = _density_by_period(perm, isn_b)
            vals = [dens_b[p] for p in dens_b if p not in ref_labels_set]
            pool.extend(vals)
            if vals:
                maxes.append(max(vals))
        threshold = float(np.percentile(pool, null_pctile)) if pool else np.inf
        real_max = max((dens[p] for p in not_ref_periods), default=0.0)
        if maxes:
            p_value = (int(np.sum(np.asarray(maxes) >= real_max)) + 1) / (len(maxes) + 1)
    else:
        dens_ref = [dens[p] for p in periods if p in ref_labels_set]
        threshold = _control_threshold(
            dens_ref, k_sigma, single_n=int(ref_mask.sum()),
            single_p=float(is_novel[ref_mask].mean()))

    table = pd.DataFrame({
        "period": periods,
        "n": [int((periods_col == p).sum()) for p in periods],
        "novelty_density": [dens[p] for p in periods],
    })
    not_ref = ~table["period"].isin(ref_labels_set)
    raw = ((table["novelty_density"] > threshold) & not_ref).to_numpy()

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

    texts = ([str(v) for v in frame[text_col]] if text_col in frame.columns
             else [""] * len(frame))
    recs = pd.DataFrame({
        "period": periods_col,
        "novelty": nov,
        "novel": is_novel,
        "text": texts,
    })

    return KNNDensityDriftResult(
        table=table,
        mode=mode,
        reference=ref_labels_set,
        k=k,
        threshold=float(threshold),
        p_value=p_value,
        embedding_meta=dict(embedding_meta or {}),
        _records=recs,
    )
