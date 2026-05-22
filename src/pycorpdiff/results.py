"""Result dataclasses returned by every public analytical verb.

Every Result implements the same informal contract:

- ``.to_df()`` returns a tidy :class:`pandas.DataFrame`.
- ``.plot(**kw)`` returns an :class:`altair.Chart`.
- ``.explain(term, n)`` returns a :class:`ConcordanceResult` with
  evidence for one row of the result.
- ``.summary()`` returns a short human-readable string.

This contract is intentionally a duck-typing convention rather than an
abstract base class — it keeps Results lightweight and lets them be
constructed from a plain DataFrame without inheritance gymnastics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import altair as alt

    from .corpus import Corpus, CorpusSlice


def _table_to_html(table: pd.DataFrame, path: str | Path | None, **kw: Any) -> str:
    """Render ``table`` as HTML; optionally write to ``path``."""
    html: str = str(table.to_html(**kw))
    if path is not None:
        Path(path).write_text(html, encoding="utf-8")
    return html


def _table_to_json(
    table: pd.DataFrame, path: str | Path | None, **kw: Any
) -> str:
    """Render ``table`` as JSON (records orientation by default); optionally
    write to ``path``.

    Coerces any object-dtype columns containing ``pd.Period`` values to
    strings before serialisation — pandas's JSON writer doesn't know
    how to represent Period and would raise OverflowError. The string
    form (``"2020"``, ``"2020Q1"``, …) round-trips back to Period
    cleanly via :func:`pandas.Period`.
    """
    serialisable = table.copy()
    for col in serialisable.columns:
        col_dtype = serialisable[col].dtype
        if isinstance(col_dtype, pd.PeriodDtype):
            serialisable[col] = serialisable[col].astype(str)
        elif col_dtype == object:  # noqa: E721
            sample = next(
                (v for v in serialisable[col] if v is not None and not pd.isna(v)),
                None,
            )
            if isinstance(sample, pd.Period):
                serialisable[col] = serialisable[col].astype(str)
    kw.setdefault("orient", "records")
    json_str: str = str(serialisable.to_json(**kw))
    if path is not None:
        Path(path).write_text(json_str, encoding="utf-8")
    return json_str


@dataclass(frozen=True)
class KeynessResult:
    """Per-term keyness scores for two corpora.

    The ``table`` DataFrame has one row per shared vocabulary item with
    columns including ``term``, ``count_a``, ``count_b``, ``score``,
    ``effect_size``, ``p_value``, ``dispersion_a``, ``dispersion_b``,
    and a boolean ``dispersion_flag``.
    """

    table: pd.DataFrame
    method: str
    n_a: int
    n_b: int
    label_a: str = "a"
    label_b: str = "b"
    params: dict[str, Any] = field(default_factory=dict)
    corpus_a: Corpus | CorpusSlice | None = None
    corpus_b: Corpus | CorpusSlice | None = None

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as HTML (returns the string and,
        optionally, writes to ``path``). Extra kwargs forward to
        :meth:`pandas.DataFrame.to_html`."""
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as JSON (default ``orient="records"``).
        Returns the JSON string and, optionally, writes to ``path``."""
        return _table_to_json(self.table, path, **kw)

    def plot(self, kind: str = "volcano", **kw: Any) -> alt.Chart:
        """Return an altair chart of the keyness result.

        ``kind="volcano"`` (default) returns a volcano-style scatter of
        effect size against −log₁₀(*p*); ``kind="bar"`` returns a top-N
        horizontal bar chart; ``kind="scattertext"`` returns the
        Scattertext-style interactive rank-percentile scatter (Kessler
        2017). Extra keyword arguments are forwarded to the underlying
        viz function (``n_labels``, ``n``, ``width``, ``height``).
        """
        from .viz.keyness import keyness_top_n_bar, keyness_volcano
        from .viz.scattertext import scattertext_plot

        if kind == "volcano":
            return keyness_volcano(self.table, **kw)
        if kind == "bar":
            return keyness_top_n_bar(self.table, **kw)
        if kind == "scattertext":
            return scattertext_plot(
                self.table, label_a=self.label_a, label_b=self.label_b, **kw
            )
        raise ValueError(
            f"unknown kind={kind!r}; expected 'volcano', 'bar', or 'scattertext'"
        )

    def explain(self, term: str, n: int = 5, window: int = 5) -> ConcordanceResult:
        """Show KWIC examples of ``term`` from both source corpora.

        Returns up to ``n`` lines per corpus. Requires that the result
        was built via :meth:`pycorpdiff.Comparison.keyness` (which
        populates the corpus references); building a ``KeynessResult``
        from a bare DataFrame will raise.
        """
        if self.corpus_a is None or self.corpus_b is None:
            raise ValueError(
                "explain() requires source corpora; this KeynessResult was "
                "constructed without them"
            )
        from .explain import kwic_compare

        return kwic_compare(
            self.corpus_a,
            self.corpus_b,
            target=term,
            window=window,
            n_per_side=n,
            label_a=self.label_a,
            label_b=self.label_b,
        )

    def summary(self) -> str:
        return (
            f"KeynessResult({self.method}, |a|={self.n_a:,}, |b|={self.n_b:,}, "
            f"terms={len(self.table):,})"
        )


@dataclass(frozen=True)
class CollocationShiftResult:
    """Change in collocates of a target term between two corpora."""

    target: str
    table: pd.DataFrame
    measure: str
    window: int
    label_a: str = "a"
    label_b: str = "b"
    corpus_a: Corpus | CorpusSlice | None = None
    corpus_b: Corpus | CorpusSlice | None = None

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as HTML (returns the string and,
        optionally, writes to ``path``). Extra kwargs forward to
        :meth:`pandas.DataFrame.to_html`."""
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as JSON (default ``orient="records"``).
        Returns the JSON string and, optionally, writes to ``path``."""
        return _table_to_json(self.table, path, **kw)

    def plot(self, **kw: Any) -> alt.Chart:
        """Return a diverging horizontal bar chart of the top collocate shifts."""
        from .viz.collocation import collocation_diverging_bar

        return collocation_diverging_bar(self.table, **kw)

    def explain(self, collocate: str, n: int = 5) -> ConcordanceResult:
        """Show KWIC windows where ``target`` co-occurs with ``collocate``.

        Returns up to ``n`` lines per corpus, restricted to contexts in
        which both the target and ``collocate`` appear within the same
        window. This is the per-row evidence behind a shift score.
        """
        if self.corpus_a is None or self.corpus_b is None:
            raise ValueError(
                "explain() requires source corpora; this CollocationShiftResult "
                "was constructed without them"
            )
        from .explain import kwic_compare

        return kwic_compare(
            self.corpus_a,
            self.corpus_b,
            target=self.target,
            window=self.window,
            n_per_side=n,
            collocate=collocate,
            label_a=self.label_a,
            label_b=self.label_b,
        )

    def summary(self) -> str:
        return (
            f"CollocationShiftResult(target={self.target!r}, measure={self.measure}, "
            f"window={self.window}, collocates={len(self.table):,})"
        )


@dataclass(frozen=True)
class SemanticShiftResult:
    """Embedding-space displacement of a target term between corpora."""

    targets: list[str]
    table: pd.DataFrame
    alignment: str
    label_a: str = "a"
    label_b: str = "b"
    corpus_a: Corpus | CorpusSlice | None = None
    corpus_b: Corpus | CorpusSlice | None = None
    embedder: Any | None = None
    window: int = 5

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as HTML (returns the string and,
        optionally, writes to ``path``). Extra kwargs forward to
        :meth:`pandas.DataFrame.to_html`."""
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as JSON (default ``orient="records"``).
        Returns the JSON string and, optionally, writes to ``path``."""
        return _table_to_json(self.table, path, **kw)

    def plot(self, **kw: Any) -> alt.Chart:
        raise NotImplementedError("SemanticShiftResult.plot() lands in Phase 6")

    def neighbors_before(
        self, target: str | None = None, n: int = 10
    ) -> pd.DataFrame:
        """Top-n contextual neighbours of ``target`` in corpus A.

        Returns the rows of :func:`pycorpdiff.semantic.neighborhood_drift`
        with a non-null ``sim_a`` (i.e. terms that appeared in A's
        top-k), sorted by ``sim_a`` descending. Requires the result was
        built via :meth:`Comparison.semantic_shift` so the source
        corpora and embedder are attached.
        """
        return self._neighborhood(target=target, n=n, side="a")

    def neighbors_after(
        self, target: str | None = None, n: int = 10
    ) -> pd.DataFrame:
        """Top-n contextual neighbours of ``target`` in corpus B."""
        return self._neighborhood(target=target, n=n, side="b")

    def _neighborhood(
        self, target: str | None, n: int, side: str
    ) -> pd.DataFrame:
        if self.corpus_a is None or self.corpus_b is None:
            raise ValueError(
                "neighbors_before / neighbors_after require source corpora; "
                "this SemanticShiftResult was constructed without them"
            )
        if target is None:
            if len(self.targets) != 1:
                raise ValueError(
                    f"result carries {len(self.targets)} targets; pass target= to pick one"
                )
            target = self.targets[0]
        if target not in self.targets:
            raise ValueError(
                f"target={target!r} not in result targets {self.targets!r}"
            )
        from .semantic.shift import neighborhood_drift

        full = neighborhood_drift(
            self.corpus_a,
            self.corpus_b,
            target=target,
            k=n,
            embedder=self.embedder,
            window=self.window,
        )
        sim_col = "sim_a" if side == "a" else "sim_b"
        return (
            full.dropna(subset=[sim_col])
            .sort_values(sim_col, ascending=False, kind="stable")
            .head(n)
            .reset_index(drop=True)
        )

    def summary(self) -> str:
        return (
            f"SemanticShiftResult(targets={self.targets!r}, alignment={self.alignment})"
        )


@dataclass(frozen=True)
class TemporalTrajectory:
    """A time-indexed series for one or more target terms.

    ``table`` has columns ``period``, ``term``, ``count``, ``relfreq``,
    ``ci_lower``, ``ci_upper``.
    """

    table: pd.DataFrame
    targets: list[str]
    freq: str

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as HTML (returns the string and,
        optionally, writes to ``path``). Extra kwargs forward to
        :meth:`pandas.DataFrame.to_html`."""
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as JSON (default ``orient="records"``).
        Returns the JSON string and, optionally, writes to ``path``."""
        return _table_to_json(self.table, path, **kw)

    def plot(self, **kw: Any) -> alt.Chart:
        """Return a line plot with Wilson CI bands per term."""
        from .viz.trajectory import trajectory_with_ci

        return trajectory_with_ci(self.table, **kw)

    def changepoints(
        self,
        target: str | None = None,
        method: str = "pelt",
        penalty: float | None = None,
    ) -> pd.DataFrame:
        """Run changepoint detection on a target's relative-frequency series.

        Requires the ``[temporal]`` extra (ruptures). When the
        trajectory holds multiple targets, supply ``target`` to pick one;
        a single-target trajectory uses it automatically.
        """
        from .temporal.changepoint import detect_changepoints

        if target is None:
            if len(self.targets) != 1:
                raise ValueError(
                    f"trajectory carries {len(self.targets)} targets; "
                    "pass target= to pick one"
                )
            target = self.targets[0]
        if target not in self.targets:
            raise ValueError(f"target={target!r} not in trajectory targets {self.targets!r}")

        sub = self.table[self.table["term"] == target].set_index("period")["relfreq"]
        return detect_changepoints(sub, method=method, penalty=penalty)  # type: ignore[arg-type]

    def interrupted_time_series(
        self,
        event_date: str,
        target: str | None = None,
    ) -> pd.DataFrame:
        """Fit a segmented-regression ITS model around ``event_date``.

        Requires the ``[temporal]`` extra (statsmodels). Returns level
        and slope-change estimates with confidence intervals.
        """
        from .temporal.its import interrupted_time_series

        if target is None:
            if len(self.targets) != 1:
                raise ValueError(
                    f"trajectory carries {len(self.targets)} targets; "
                    "pass target= to pick one"
                )
            target = self.targets[0]
        if target not in self.targets:
            raise ValueError(f"target={target!r} not in trajectory targets {self.targets!r}")

        sub = self.table[self.table["term"] == target].set_index("period")["relfreq"]
        return interrupted_time_series(sub, event_date=event_date)

    def summary(self) -> str:
        return (
            f"TemporalTrajectory(targets={self.targets!r}, freq={self.freq!r}, "
            f"periods={self.table['period'].nunique() if 'period' in self.table else 0:,})"
        )


@dataclass(frozen=True)
class ConcordanceResult:
    """KWIC (keyword-in-context) lines for a target term."""

    target: str
    table: pd.DataFrame
    window: int

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def to_html(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as HTML (returns the string and,
        optionally, writes to ``path``). Extra kwargs forward to
        :meth:`pandas.DataFrame.to_html`."""
        return _table_to_html(self.table, path, **kw)

    def to_json(self, path: str | Path | None = None, **kw: Any) -> str:
        """Render the underlying table as JSON (default ``orient="records"``).
        Returns the JSON string and, optionally, writes to ``path``."""
        return _table_to_json(self.table, path, **kw)

    def summary(self) -> str:
        return f"ConcordanceResult(target={self.target!r}, lines={len(self.table):,})"
