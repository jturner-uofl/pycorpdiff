"""Semantic flow fields — velocity, acceleration, and divergence of meaning.

Where :func:`semantic_trajectory` tracks *one* word's distance from a
baseline and :func:`induce_senses` clusters a *single* snapshot, this
module treats a whole vocabulary's diachronic embeddings as a
**time-varying vector field** and borrows the analysis vocabulary that
fluid dynamics and single-cell genomics (RNA velocity) built for exactly
that object.

Given a word's vector in an ordered sequence of aligned time slices
(e.g. HistWords' Procrustes-aligned per-decade word2vec), we compute:

- **velocity** — the finite-difference displacement between consecutive
  slices. Its magnitude (**speed**) is the rate of semantic change
  (Hamilton, Leskovec & Jurafsky 2016's "law of semantic change").
- **acceleration** — the change in velocity; spikes flag *semantic
  shocks* (a meaning that suddenly starts moving).
- **divergence** — the local flux of the velocity field over a word's
  nearest neighbours. Positive divergence = the neighbourhood is
  flowing apart (**semantic broadening**, e.g. *holiday*: holy day ->
  any day off); negative = converging (**narrowing**, e.g. *meat*: any
  food -> animal flesh). This gives the 19th-century broadening /
  narrowing typology a computable, sign-carrying instrument.

The result plots as a velocity field: a 2-D projection of the
vocabulary with an arrow per word from its first-slice to its
last-slice position, coloured by divergence — the RNA-velocity picture,
applied to language.

This is *complementary*, not load-bearing: the embeddings are
caller-supplied and must already be aligned across slices (HistWords
ships pre-aligned; otherwise run :func:`procrustes_align` first).
``scikit-learn`` (the ``[semantic]`` extra) is required and imported
lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ..results import _table_to_html, _table_to_json

if TYPE_CHECKING:
    import altair as alt
    import pandas as pd

FloatArray = npt.NDArray[np.float64]


def _normalise_eras(
    eras: Mapping[Any, Mapping[str, Any]] | Sequence[tuple[Any, Mapping[str, Any]]],
) -> list[tuple[Any, dict[str, FloatArray]]]:
    """Coerce the eras argument to an ordered list of (label, {word: vec})."""
    items = list(eras.items()) if isinstance(eras, Mapping) else list(eras)
    if len(items) < 2:
        raise ValueError("semantic_flow needs at least 2 time slices")
    out: list[tuple[Any, dict[str, FloatArray]]] = []
    for label, mapping in items:
        if not isinstance(mapping, Mapping):
            raise ValueError(
                f"era {label!r} must map word -> vector, got {type(mapping).__name__}"
            )
        out.append((label, {w: np.asarray(v, dtype=np.float64) for w, v in mapping.items()}))
    return out


def _local_divergence(
    positions: FloatArray, velocities: FloatArray, knn: int
) -> FloatArray:
    """Discrete divergence of the velocity field at each point.

    For point i with velocity v_i, over its k nearest neighbours j the
    estimator averages the radial component of relative velocity,
    ``(v_j - v_i) . (p_j - p_i) / |p_j - p_i|^2``. This is the discrete
    flux through a small neighbourhood (a sign-preserving divergence
    estimate valid in any dimension): positive => neighbours stream
    away (a source / broadening), negative => converge (a sink /
    narrowing).
    """
    from sklearn.neighbors import NearestNeighbors

    n = positions.shape[0]
    k = min(knn, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(positions)
    _, idx = nn.kneighbors(positions)
    div = np.empty(n, dtype=np.float64)
    eps = 1e-12
    for i in range(n):
        neigh = idx[i][1:]  # drop self
        dp = positions[neigh] - positions[i]
        dv = velocities[neigh] - velocities[i]
        dist2 = np.einsum("ij,ij->i", dp, dp) + eps
        radial = np.einsum("ij,ij->i", dv, dp) / dist2
        div[i] = float(radial.mean())
    return div


@dataclass(frozen=True)
class SemanticFlowResult:
    """Velocity-field summary of a vocabulary's diachronic embeddings.

    Attributes
    ----------
    table
        Per-word: ``word``, ``speed`` (mean per-step velocity
        magnitude), ``displacement`` (first->last distance),
        ``acceleration`` (max per-step velocity change; ``NaN`` with
        only two slices), ``divergence`` (local field divergence at the
        first slice; >0 broadening, <0 narrowing).
    eras
        The ordered slice labels.
    projection
        ``(n_words, 2)`` 2-D coordinates of the *first* slice (for
        plotting), plus ``projection_end`` for the last slice.
    """

    table: pd.DataFrame
    eras: list[Any]
    projection: FloatArray = field(repr=False)
    projection_end: FloatArray = field(repr=False)

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        return _table_to_json(self.table, path, **kw)

    def summary(self) -> str:
        t = self.table
        fastest = t.sort_values("speed", ascending=False).iloc[0]
        broad = t.sort_values("divergence", ascending=False).iloc[0]
        narrow = t.sort_values("divergence").iloc[0]
        return (
            f"{len(t):,} words across {len(self.eras)} slices "
            f"({self.eras[0]}..{self.eras[-1]}). "
            f"Fastest: {fastest['word']!r} (speed {fastest['speed']:.3f}). "
            f"Most broadening: {broad['word']!r} (div {broad['divergence']:+.3f}); "
            f"most narrowing: {narrow['word']!r} (div {narrow['divergence']:+.3f})."
        )

    def plot(self, *, top: int = 40, label: Sequence[str] | None = None, **kw: Any) -> alt.Chart:
        """RNA-velocity-style flow field: arrows from first->last slice
        position in a 2-D projection, coloured by divergence.

        ``top`` draws arrows for the fastest-moving ``top`` words (plus
        any explicitly named in ``label``) to keep the field readable.
        """
        import altair as alt
        import pandas as pd

        t = self.table.reset_index(drop=True)
        keep = set(t.sort_values("speed", ascending=False).head(top)["word"])
        if label:
            keep |= set(label)
        mask = t["word"].isin(keep).to_numpy()
        df = pd.DataFrame(
            {
                "word": t.loc[mask, "word"].to_numpy(),
                "x0": self.projection[mask, 0],
                "y0": self.projection[mask, 1],
                "x1": self.projection_end[mask, 0],
                "y1": self.projection_end[mask, 1],
                "speed": t.loc[mask, "speed"].to_numpy(),
                "divergence": t.loc[mask, "divergence"].to_numpy(),
            }
        )
        arrows = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("x0:Q", title="semantic PC1"),
                y=alt.Y("y0:Q", title="semantic PC2"),
                x2="x1:Q",
                y2="y1:Q",
                color=alt.Color(
                    "divergence:Q",
                    title="divergence (+broaden / -narrow)",
                    scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
                ),
                detail="word:N",
                tooltip=["word", alt.Tooltip("speed:Q", format=".3f"),
                         alt.Tooltip("divergence:Q", format="+.3f")],
            )
        )
        heads = (
            alt.Chart(df)
            .mark_point(shape="triangle", filled=True, size=45)
            .encode(x="x1:Q", y="y1:Q",
                    color=alt.Color("divergence:Q",
                                    scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
                                    legend=None))
        )
        text = (
            alt.Chart(df)
            .mark_text(align="left", dx=5, dy=-3, fontSize=9)
            .encode(x="x1:Q", y="y1:Q", text="word:N")
        )
        chart = (arrows + heads + text).properties(**kw)
        return chart  # type: ignore[no-any-return]


def semantic_flow(
    eras: Mapping[Any, Mapping[str, Any]] | Sequence[tuple[Any, Mapping[str, Any]]],
    *,
    words: Sequence[str] | None = None,
    knn: int = 15,
    n_components: int = 2,
    random_state: int = 42,
) -> SemanticFlowResult:
    """Treat a vocabulary's diachronic embeddings as a velocity field.

    Parameters
    ----------
    eras
        Ordered mapping ``{slice_label: {word: vector}}`` (or a list of
        ``(slice_label, mapping)`` pairs). Vectors **must already be
        aligned across slices** — HistWords ships per-decade
        Procrustes-aligned vectors; otherwise align with
        :func:`procrustes_align` first. Insertion / list order is the
        time order.
    words
        Vocabulary to track. Defaults to the words present in *every*
        slice (the intersection), so every tracked word has a complete
        trajectory.
    knn
        Neighbours used for the local divergence estimate.
    n_components
        Projection dimensionality for the plot (2 by default).
    random_state
        Seed for the PCA projection.

    Returns
    -------
    SemanticFlowResult
    """
    import pandas as pd
    from sklearn.decomposition import PCA

    slices = _normalise_eras(eras)
    labels = [lab for lab, _ in slices]

    if words is not None:
        vocab = [w for w in words]
        missing = [w for w in vocab if not all(w in m for _, m in slices)]
        if missing:
            raise ValueError(
                f"{len(missing)} requested word(s) absent from some slice, "
                f"e.g. {missing[:5]}"
            )
    else:
        common: set[str] | None = None
        for _, m in slices:
            common = set(m) if common is None else (common & set(m))
        vocab = sorted(common or set())
    if len(vocab) < 2:
        raise ValueError("fewer than 2 words common to all slices")

    d = next(iter(slices[0][1].values())).shape[0]
    n_words = len(vocab)
    n_slices = len(slices)

    # Trajectory tensor: (n_words, n_slices, d)
    traj = np.empty((n_words, n_slices, d), dtype=np.float64)
    for wi, w in enumerate(vocab):
        for si, (_, m) in enumerate(slices):
            traj[wi, si] = m[w]

    # Velocity (n_words, n_slices-1, d) and derived scalars.
    velocity = np.diff(traj, axis=1)
    step_speed = np.linalg.norm(velocity, axis=2)          # (n_words, n_slices-1)
    speed = step_speed.mean(axis=1)
    displacement = np.linalg.norm(traj[:, -1] - traj[:, 0], axis=1)
    if n_slices >= 3:
        accel = np.linalg.norm(np.diff(velocity, axis=1), axis=2).max(axis=1)
    else:
        accel = np.full(n_words, np.nan)

    # Divergence of the velocity field at the first slice.
    divergence = _local_divergence(traj[:, 0], velocity[:, 0], knn)

    table = pd.DataFrame(
        {
            "word": vocab,
            "speed": speed,
            "displacement": displacement,
            "acceleration": accel,
            "divergence": divergence,
        }
    ).sort_values("speed", ascending=False).reset_index(drop=True)

    # 2-D projection fit on the union of all slice vectors.
    stacked = traj.reshape(n_words * n_slices, d)
    proj = PCA(n_components=n_components, random_state=random_state).fit(stacked)
    start_xy = proj.transform(traj[:, 0])[:, :2]
    end_xy = proj.transform(traj[:, -1])[:, :2]
    # Re-order projections to match the (speed-sorted) table.
    order = {w: i for i, w in enumerate(vocab)}
    perm = [order[w] for w in table["word"]]

    return SemanticFlowResult(
        table=table,
        eras=labels,
        projection=start_xy[perm],
        projection_end=end_xy[perm],
    )
