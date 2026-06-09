"""Embedding-based word-sense induction (WSI) and reference-label auditing.

Where :func:`semantic_trajectory` tracks how *one* term's meaning drifts
over time, :func:`induce_senses` answers a different question: *how many
distinct senses are present in a set of occurrences, and which occurrence
belongs to which sense* — discovered unsupervised, from the geometry of
the embeddings alone.

The motivating use case is auditing a hand-built sense classifier. A
regex or keyword classifier that assigns "cannabidiol" vs "common bile
duct" vs "corticobasal degeneration" to PubMed records is fast and
transparent, but a reviewer can fairly ask whether the buckets were
tuned to produce a desired answer. :func:`induce_senses` provides an
independent second opinion:

1. **Induce** senses by clustering the embeddings (:func:`induce_senses`).
2. **Cross-check** the induced partition against the hand-built labels
   (:meth:`SenseInductionResult.agreement_with` → adjusted Rand index,
   V-measure, contingency table).
3. **Localise leakage** — surface the individual records whose embedding
   geometry disagrees with their assigned reference label
   (:meth:`SenseInductionResult.leakage_audit`).

Design contract (see ``docs/design.md``):

- **Bring-your-own embeddings.** :func:`induce_senses` takes a
  precomputed ``(n_items, d)`` matrix, never an embedder. The base
  install stays light; model choice, pinning and caching are the
  caller's, which is what keeps a run reproducible.
- **Deterministic by default.** Clustering is k-means with a fixed
  seed (or agglomerative, which is seed-free); UMAP / HDBSCAN are
  deliberately *not* used, so the audit is byte-stable across runs.
- **Counts stay primary.** This is a complementary lens, not a
  replacement for the count-based keyness core.

``scikit-learn`` is required and lives in the ``[semantic]`` extra; it
is imported lazily so importing :mod:`pycorpdiff` never pulls it.
"""

from __future__ import annotations

import re
import warnings
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

# A deliberately small, conventional English stop list. Cluster labelling
# (top distinctive terms) is a readability aid, not an inferential step,
# so an exhaustive list isn't warranted — these remove the worst noise.
_STOPWORDS = frozenset(
    """
    a an the and or but of to in on at for with without from by as is are was were
    be been being this that these those it its their his her our your my we you they
    he she i them us not no nor can will would could should may might must do does did
    have has had having than then so such into over under between about against during
    which who whom whose what when where why how all any both each few more most other
    some only own same too very s t can also one two three new used using use study
    """.split()  # noqa: SIM905 - readable stop list, not worth a 120-item literal
)

_WORD_RE = re.compile(r"[a-z][a-z-]{2,}")


def _l2_normalize(x: FloatArray) -> FloatArray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.asarray(x / norms, dtype=np.float64)


def _silhouette_select(
    x: FloatArray,
    k_range: tuple[int, int],
    method: str,
    random_state: int,
) -> tuple[int, npt.NDArray[np.intp], float, pd.DataFrame]:
    """Pick k by maximising the mean silhouette over ``k_range``.

    Silhouette is O(n^2) in memory, so for large n it is estimated on a
    seeded sub-sample (``sample_size``) — deterministic given
    ``random_state``. Returns the chosen k, its labels, its score, and
    a per-k score table for transparency.
    """
    from sklearn.metrics import silhouette_score

    n = x.shape[0]
    lo, hi = k_range
    hi = min(hi, n - 1)
    if hi < lo:
        raise ValueError(
            f"k_range {k_range} leaves no valid k for n_items={n}; "
            "supply an explicit k= or more items"
        )
    sample_size = None if n <= 2000 else 2000
    rows: list[dict[str, Any]] = []
    best: tuple[int, npt.NDArray[np.intp], float] | None = None
    for k in range(lo, hi + 1):
        # Probing k larger than the number of natural clusters is the
        # whole point of the search; sklearn's ConvergenceWarning
        # ("distinct clusters < n_clusters") is expected noise here and
        # is handled below by scoring such partitions as NaN.
        with warnings.catch_warnings():
            from sklearn.exceptions import ConvergenceWarning

            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = _fit_predict(x, k, method, random_state)
            if len(np.unique(labels)) < 2 or len(np.unique(labels)) < k:
                score = float("nan")
            else:
                score = float(
                    silhouette_score(
                        x, labels, sample_size=sample_size, random_state=random_state
                    )
                )
        rows.append({"k": k, "silhouette": score})
        if best is None or (not np.isnan(score) and score > best[2]):
            best = (k, labels, score)
    assert best is not None
    return best[0], best[1], best[2], pd.DataFrame(rows)


def _fit_predict(
    x: FloatArray, k: int, method: str, random_state: int
) -> npt.NDArray[np.intp]:
    if method == "kmeans":
        from sklearn.cluster import KMeans

        labels = KMeans(
            n_clusters=k, n_init=10, random_state=random_state
        ).fit_predict(x)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        labels = AgglomerativeClustering(n_clusters=k).fit_predict(x)
    else:
        raise ValueError(
            f"method must be 'kmeans' or 'agglomerative', got {method!r}"
        )
    return np.asarray(labels, dtype=np.intp)


def _top_terms_per_cluster(
    texts: Sequence[str], labels: npt.NDArray[np.int_], n_terms: int = 8
) -> dict[int, str]:
    """Label each cluster by its most *distinctive* terms.

    Distinctiveness = within-cluster relative frequency minus overall
    relative frequency (a cheap keyness proxy, kept local so the module
    has no dependency on the keyness package). Purely a readability aid.
    """
    tokens_per_item = [
        [t for t in _WORD_RE.findall(str(s).lower()) if t not in _STOPWORDS]
        for s in texts
    ]
    overall: dict[str, int] = {}
    for toks in tokens_per_item:
        for t in toks:
            overall[t] = overall.get(t, 0) + 1
    overall_total = sum(overall.values()) or 1

    out: dict[int, str] = {}
    for c in np.unique(labels):
        in_c = [tokens_per_item[i] for i in np.flatnonzero(labels == c)]
        counts: dict[str, int] = {}
        for toks in in_c:
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
        cluster_total = sum(counts.values()) or 1
        scored = [
            (t, n / cluster_total - overall.get(t, 0) / overall_total)
            for t, n in counts.items()
            if n >= 2
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        out[int(c)] = ", ".join(t for t, _ in scored[:n_terms])
    return out


@dataclass(frozen=True)
class SenseAgreement:
    """Agreement between an induced partition and a reference labelling.

    Returned by :meth:`SenseInductionResult.agreement_with`. All three
    cluster-comparison metrics are label-permutation invariant, so they
    are meaningful even though induced cluster ids (0, 1, 2, …) have no
    relation to the reference sense names.
    """

    ari: float
    """Adjusted Rand index: 1.0 = identical partitions, ~0 = chance."""
    v_measure: float
    """Harmonic mean of homogeneity and completeness (0..1)."""
    homogeneity: float
    completeness: float
    contingency: pd.DataFrame
    """rows = reference sense, cols = induced cluster, cells = counts."""

    def to_df(self) -> pd.DataFrame:
        return self.contingency.copy()

    def summary(self) -> str:
        return (
            f"ARI={self.ari:.3f}  V-measure={self.v_measure:.3f}  "
            f"(homogeneity={self.homogeneity:.3f}, "
            f"completeness={self.completeness:.3f})"
        )

    def __str__(self) -> str:  # pragma: no cover - thin
        return self.summary()


@dataclass(frozen=True)
class SenseInductionResult:
    """Unsupervised sense partition over a set of occurrences.

    Attributes
    ----------
    clusters
        Per-sense summary: ``sense``, ``size``, ``share``, ``top_terms``.
        This is what :meth:`to_df` returns.
    assignments
        Per-item: ``doc_id``, ``sense`` (and ``text`` when available).
    labels
        Integer induced-sense label per input item, in input order.
    k, method, silhouette
        The chosen number of senses, clustering method, and the mean
        silhouette of the chosen partition (``None`` if an explicit k
        was supplied and silhouette wasn't computed).
    k_scores
        Per-k silhouette table when k was selected automatically, else
        ``None``.
    embedding_meta
        Caller-supplied provenance (model, revision, vector hash),
        echoed verbatim for the reproducibility manifest.
    """

    clusters: pd.DataFrame
    assignments: pd.DataFrame
    labels: npt.NDArray[np.int_]
    k: int
    method: str
    silhouette: float | None
    k_scores: pd.DataFrame | None
    embedding_meta: dict[str, Any]
    unit: str
    _embeddings: FloatArray = field(repr=False)
    _doc_ids: npt.NDArray[Any] = field(repr=False)
    _texts: list[str] | None = field(repr=False, default=None)
    _time: pd.Series | None = field(repr=False, default=None)

    # ---- six-method contract -------------------------------------------------
    def to_df(self) -> pd.DataFrame:
        return self.clusters.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_html(self.clusters, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_json(self.clusters, path, **kw)

    def summary(self) -> str:
        n = len(self.labels)
        sil = f"{self.silhouette:.3f}" if self.silhouette is not None else "n/a"
        top = self.clusters.sort_values("share", ascending=False).iloc[0]
        return (
            f"{self.k} senses induced over {n:,} {self.unit}s "
            f"({self.method}, silhouette={sil}); "
            f"largest sense {top['share']:.1%}"
        )

    def plot(self, **kw: Any) -> alt.Chart:
        """Horizontal bar chart of per-sense record share."""
        import altair as alt

        data = self.clusters.assign(
            label=lambda d: d["sense"].astype(str) + "  " + d["top_terms"].str.slice(0, 40)
        )
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X("share:Q", title="share of records", axis=alt.Axis(format="%")),
                y=alt.Y("label:N", title=None, sort="-x"),
                tooltip=["sense", "size", alt.Tooltip("share:Q", format=".1%"), "top_terms"],
            )
            .properties(**kw)
        )
        return chart  # type: ignore[no-any-return]

    # ---- audit operations ----------------------------------------------------
    def agreement_with(self, reference_labels: Sequence[Any]) -> SenseAgreement:
        """Quantify agreement between the induced partition and a
        reference labelling (e.g. the hand-built regex buckets).

        ``reference_labels`` must be aligned with the input items (one
        label per item, input order). Returns a :class:`SenseAgreement`
        with adjusted Rand index, V-measure and a contingency table.
        """
        from sklearn.metrics import (
            adjusted_rand_score,
            homogeneity_completeness_v_measure,
        )

        ref = np.asarray(list(reference_labels))
        if len(ref) != len(self.labels):
            raise ValueError(
                f"reference_labels has length {len(ref)} but result has "
                f"{len(self.labels)} items; they must align by position"
            )
        ari = float(adjusted_rand_score(ref, self.labels))
        homog, compl, vmeas = (
            float(v) for v in homogeneity_completeness_v_measure(ref, self.labels)
        )
        contingency = pd.crosstab(
            pd.Series(ref, name="reference"),
            pd.Series(self.labels, name="induced"),
        )
        return SenseAgreement(ari, vmeas, homog, compl, contingency)

    def leakage_audit(
        self, reference_labels: Sequence[Any], k: int = 20
    ) -> pd.DataFrame:
        """Surface records whose embedding geometry disputes their
        reference label — i.e. likely reference-classifier leakage.

        For each reference class a centroid is computed from the
        embeddings. A record is *suspect* when it sits closer to a
        different class's centroid than to its own. Suspects are ranked
        by margin (how much closer the rival centroid is). This is the
        deterministic, targeted counterpart to a random spot-check:
        instead of sampling and hoping to hit a leaked record, it goes
        straight to the most geometrically misplaced ones.

        Returns up to ``k`` rows: ``doc_id``, ``reference_sense``,
        ``nearest_other_sense``, ``margin`` (positive = suspect), and
        ``text`` when available.
        """
        ref = np.asarray(list(reference_labels))
        if len(ref) != len(self.labels):
            raise ValueError(
                f"reference_labels has length {len(ref)} but result has "
                f"{len(self.labels)} items; they must align by position"
            )
        x = self._embeddings
        classes = list(pd.unique(ref))
        if len(classes) < 2:
            raise ValueError("leakage_audit needs >= 2 reference classes")
        centroids = np.vstack(
            [x[ref == c].mean(axis=0) for c in classes]
        )  # (n_classes, d)
        # Squared Euclidean distance from every item to every centroid.
        dists = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        own_idx = np.array([class_to_idx[c] for c in ref])
        own_dist = dists[np.arange(len(ref)), own_idx]
        masked = dists.copy()
        masked[np.arange(len(ref)), own_idx] = np.inf
        nearest_other_idx = masked.argmin(axis=1)
        nearest_other_dist = masked[np.arange(len(ref)), nearest_other_idx]
        margin = own_dist - nearest_other_dist  # > 0 => closer to a rival
        suspect = margin > 0
        order = np.argsort(margin[suspect])[::-1]
        sel = np.flatnonzero(suspect)[order][:k]
        out = pd.DataFrame(
            {
                "doc_id": self._doc_ids[sel],
                "reference_sense": ref[sel],
                "nearest_other_sense": [classes[i] for i in nearest_other_idx[sel]],
                "margin": margin[sel],
            }
        )
        if self._texts is not None:
            out["text"] = [self._texts[i][:160] for i in sel]
        return out.reset_index(drop=True)

    def share_over_time(
        self, freq: str = "Y", time: Sequence[Any] | None = None
    ) -> pd.DataFrame:
        """Induced-sense share per time period — the computed counterpart
        to a hand-built sense-fraction trajectory.

        Uses the time values captured at induction (via ``time_col``) or
        an explicit ``time`` override. Returns tidy rows: ``period``,
        ``sense``, ``count``, ``share`` (share within period).
        """
        tvals = self._time if time is None else pd.Series(list(time))
        if tvals is None:
            raise ValueError(
                "no time values available; pass time_col= to induce_senses "
                "or a time= sequence here"
            )
        period = pd.PeriodIndex(pd.to_datetime(tvals, errors="coerce"), freq=freq)
        df = pd.DataFrame({"period": period, "sense": self.labels})
        df = df.dropna(subset=["period"])
        counts = (
            df.groupby(["period", "sense"]).size().rename("count").reset_index()
        )
        totals = counts.groupby("period")["count"].transform("sum")
        counts["share"] = counts["count"] / totals
        counts["period"] = counts["period"].astype(str)
        return counts


def induce_senses(
    items: pd.DataFrame | Sequence[str],
    embeddings: FloatArray,
    *,
    k: int | None = None,
    unit: str = "document",
    item_to_doc: str | Sequence[Any] | None = None,
    method: str = "kmeans",
    random_state: int = 42,
    text_col: str = "text",
    time_col: str | None = None,
    normalize: bool = True,
    k_range: tuple[int, int] = (2, 10),
    embedding_meta: dict[str, Any] | None = None,
) -> SenseInductionResult:
    """Induce word senses by clustering bring-your-own embeddings.

    Parameters
    ----------
    items
        Either a :class:`pandas.DataFrame` (one row per text unit) or a
        plain sequence of strings. Row order must align with
        ``embeddings``.
    embeddings
        Precomputed ``(n_items, d)`` matrix — one vector per item. This
        function never embeds text itself: produce the vectors with any
        embedder you like (e.g. :class:`SBERTEmbedder`, or your own
        contextual model for token-level WSI) and pass them in. That
        keeps the base install light and puts model pinning / caching
        in your hands.
    k
        Number of senses. ``None`` (default) selects k by maximising the
        mean silhouette over ``k_range``. For auditing a hand-built
        classifier, pass ``k = n_reference_buckets`` so the contingency
        table is square.
    unit
        ``"document"`` (each row is a document) or ``"token"`` (each row
        is a single token-occurrence; supply ``item_to_doc`` to map
        occurrences back to documents).
    item_to_doc
        Required when ``unit="token"``: a column name in ``items`` or an
        array aligning each occurrence to its document id.
    method
        ``"kmeans"`` (default, deterministic via ``random_state``) or
        ``"agglomerative"`` (deterministic, seed-free).
    text_col, time_col
        Column names in ``items`` for the text (used for cluster
        labelling) and, optionally, a timestamp (enables
        :meth:`SenseInductionResult.share_over_time`).
    normalize
        L2-normalise embeddings before clustering so k-means's Euclidean
        objective approximates cosine. Default ``True``.
    k_range
        Inclusive ``(min_k, max_k)`` searched when ``k is None``.
    embedding_meta
        Free-form provenance dict (model name, revision, vector hash)
        echoed onto the result for your reproducibility manifest.

    Returns
    -------
    SenseInductionResult
    """
    if isinstance(items, pd.DataFrame):
        frame = items.reset_index(drop=True)
    else:
        frame = pd.DataFrame({text_col: list(items)})

    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n_items, d); got shape {x.shape}")
    if x.shape[0] != len(frame):
        raise ValueError(
            f"embeddings has {x.shape[0]} rows but items has {len(frame)}; "
            "they must align by position"
        )
    if x.shape[0] < 2:
        raise ValueError("need at least 2 items to induce senses")
    if not np.isfinite(x).all():
        raise ValueError("embeddings contain NaN or inf")
    if unit not in {"document", "token"}:
        raise ValueError(f"unit must be 'document' or 'token', got {unit!r}")

    if normalize:
        x = _l2_normalize(x)

    # Resolve doc ids.
    if unit == "token" and item_to_doc is None:
        raise ValueError("unit='token' requires item_to_doc")
    if item_to_doc is None:
        doc_ids: npt.NDArray[Any] = np.arange(len(frame))
    elif isinstance(item_to_doc, str):
        if item_to_doc not in frame.columns:
            raise ValueError(f"item_to_doc column {item_to_doc!r} not in items")
        doc_ids = frame[item_to_doc].to_numpy()
    else:
        doc_ids = np.asarray(list(item_to_doc))
        if len(doc_ids) != len(frame):
            raise ValueError("item_to_doc length must match items")

    # Cluster.
    if k is not None:
        if k < 2:
            raise ValueError("k must be >= 2")
        if k >= len(frame):
            raise ValueError(f"k={k} must be < n_items={len(frame)}")
        labels = _fit_predict(x, k, method, random_state)
        chosen_k = k
        k_scores = None
        silhouette: float | None = None
    else:
        chosen_k, labels, sil, k_scores = _silhouette_select(
            x, k_range, method, random_state
        )
        silhouette = sil

    labels = np.asarray(labels, dtype=int)

    texts: list[str] | None = None
    if text_col in frame.columns:
        texts = [str(v) for v in frame[text_col].tolist()]

    top_terms = (
        _top_terms_per_cluster(texts, labels) if texts is not None else {}
    )

    sizes = pd.Series(labels).value_counts().sort_index()
    clusters = pd.DataFrame(
        {
            "sense": sizes.index.astype(int),
            "size": sizes.to_numpy(),
            "share": (sizes / sizes.sum()).to_numpy(),
            "top_terms": [top_terms.get(int(s), "") for s in sizes.index],
        }
    ).reset_index(drop=True)

    assignments = pd.DataFrame({"doc_id": doc_ids, "sense": labels})
    if texts is not None:
        assignments["text"] = [t[:200] for t in texts]

    time_series: pd.Series | None = None
    if time_col is not None and time_col in frame.columns:
        time_series = frame[time_col].reset_index(drop=True)

    return SenseInductionResult(
        clusters=clusters,
        assignments=assignments,
        labels=labels,
        k=chosen_k,
        method=method,
        silhouette=silhouette,
        k_scores=k_scores,
        embedding_meta=dict(embedding_meta or {}),
        unit=unit,
        _embeddings=x,
        _doc_ids=doc_ids,
        _texts=texts,
        _time=time_series,
    )
