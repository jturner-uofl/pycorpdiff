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
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import altair as alt

    from .corpus import Corpus, CorpusSlice


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

    def plot(self, kind: str = "volcano", **kw: Any) -> alt.Chart:
        """Return an altair chart of the keyness result.

        ``kind="volcano"`` (default) returns a volcano-style scatter of
        effect size against −log₁₀(*p*); ``kind="bar"`` returns a top-N
        horizontal bar chart. Extra keyword arguments are forwarded to
        the underlying viz function (``n_labels``, ``n``, ``width``,
        ``height``).
        """
        from .viz.keyness import keyness_top_n_bar, keyness_volcano

        if kind == "volcano":
            return keyness_volcano(self.table, **kw)
        if kind == "bar":
            return keyness_top_n_bar(self.table, **kw)
        raise ValueError(f"unknown kind={kind!r}; expected 'volcano' or 'bar'")

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

    def to_df(self) -> pd.DataFrame:
        return self.table.copy()

    def plot(self, **kw: Any) -> alt.Chart:
        raise NotImplementedError("SemanticShiftResult.plot() lands in Phase 6")

    def neighbors_before(self, target: str | None = None, n: int = 10) -> pd.DataFrame:
        raise NotImplementedError("SemanticShiftResult.neighbors_before() lands in Phase 6")

    def neighbors_after(self, target: str | None = None, n: int = 10) -> pd.DataFrame:
        raise NotImplementedError("SemanticShiftResult.neighbors_after() lands in Phase 6")

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

    def plot(self, **kw: Any) -> alt.Chart:
        """Return a line plot with Wilson CI bands per term."""
        from .viz.trajectory import trajectory_with_ci

        return trajectory_with_ci(self.table, **kw)

    def changepoints(self, **kw: Any) -> pd.DataFrame:
        raise NotImplementedError("TemporalTrajectory.changepoints() lands in Phase 7")

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

    def summary(self) -> str:
        return f"ConcordanceResult(target={self.target!r}, lines={len(self.table):,})"
