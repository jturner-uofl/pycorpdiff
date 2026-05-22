"""Temporal slicing primitives — ``TemporalCorpus``, ``track``, ``Tracker``."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..corpus import Corpus, CorpusSlice
from ..results import TemporalTrajectory
from ..stats import wilson_ci


@dataclass(frozen=True)
class TemporalCorpus:
    """A corpus indexed by time period for diachronic analysis.

    Constructed via :meth:`pycorpdiff.Corpus.by_time`; bucketing of the
    parent corpus's ``time_col`` follows the pandas offset alias
    ``freq``. Periods with no documents are skipped — there's no
    silent-zero entry in :meth:`periods` or :meth:`iter_slices`.
    """

    parent: Corpus
    time_col: str
    freq: str = "Y"

    def __len__(self) -> int:
        return len(self.parent)

    def _period_series(self) -> pd.Series:
        """Per-document Period values, indexed like the parent's docs frame."""
        times = pd.to_datetime(self.parent.docs[self.time_col])
        return times.dt.to_period(self.freq)

    def periods(self) -> list[pd.Period]:
        """Sorted list of populated periods."""
        return sorted(self._period_series().unique())

    def slice(self, period: pd.Period | str) -> CorpusSlice:
        """Return the :class:`CorpusSlice` for one period.

        ``period`` may be a :class:`pandas.Period` or any string pandas
        can parse to one (e.g. ``"2020"``, ``"2020Q1"``, ``"2020-03"``).
        """
        idx = self._period_series()
        period_obj = pd.Period(period, freq=self.freq) if isinstance(period, str) else period
        mask = pd.Series(idx.values == period_obj, index=self.parent.docs.index)
        return CorpusSlice(
            parent=self.parent,
            mask=mask,
            filters={"period": str(period_obj)},
        )

    def iter_slices(self) -> Iterator[tuple[pd.Period, CorpusSlice]]:
        """Yield ``(period, CorpusSlice)`` pairs in chronological order."""
        idx = self._period_series()
        for period in self.periods():
            mask = pd.Series(idx.values == period, index=self.parent.docs.index)
            yield period, CorpusSlice(
                parent=self.parent, mask=mask, filters={"period": str(period)}
            )


@dataclass(frozen=True)
class Tracker:
    """A diachronic tracker over one or more target terms."""

    corpus: Corpus | CorpusSlice
    targets: list[str]

    def over_time(
        self,
        freq: str = "Y",
        time_col: str = "date",
        confidence: float = 0.95,
    ) -> TemporalTrajectory:
        """Return a :class:`TemporalTrajectory` of relative frequencies.

        For every populated period × target, computes the raw count,
        period token total, relative frequency, and a Wilson score
        interval at ``confidence`` (default 95%). The output frame has
        one row per (period, term) pair, sorted by term then period.
        """
        temporal = self.corpus.by_time(time_col, freq)
        rows: list[dict[str, object]] = []
        for period, slice_ in temporal.iter_slices():
            tokens_per_doc = slice_.tokens()
            all_tokens: list[str] = [tok for doc in tokens_per_doc for tok in doc]
            total = len(all_tokens)
            counter = pd.Series(all_tokens).value_counts() if total else pd.Series(dtype=int)
            for target in self.targets:
                count = int(counter.get(target, 0))
                relfreq = (count / total) if total > 0 else float("nan")
                lo, hi = wilson_ci(
                    np.array([count], dtype=np.int64),
                    np.array([total], dtype=np.int64),
                    confidence=confidence,
                )
                rows.append(
                    {
                        "period": period,
                        "term": target,
                        "count": count,
                        "total": total,
                        "relfreq": relfreq,
                        "ci_lower": float(lo[0]),
                        "ci_upper": float(hi[0]),
                    }
                )
        table = (
            pd.DataFrame(rows)
            .sort_values(["term", "period"], kind="stable")
            .reset_index(drop=True)
        )
        return TemporalTrajectory(table=table, targets=list(self.targets), freq=freq)

    def trajectory(
        self,
        freq: str = "Y",
        time_col: str = "date",
        confidence: float = 0.95,
    ) -> TemporalTrajectory:
        """Alias for :meth:`over_time`."""
        return self.over_time(freq=freq, time_col=time_col, confidence=confidence)


def track(
    corpus: Corpus | CorpusSlice, target: str | list[str]
) -> Tracker:
    """Construct a :class:`Tracker` for diachronic analysis of target term(s).

    Accepts either a :class:`Corpus` or a :class:`CorpusSlice`, so
    ``pcd.track(corpus.slice(topic="immigration"), "criminal")`` works
    out of the box.
    """
    targets = [target] if isinstance(target, str) else list(target)
    return Tracker(corpus=corpus, targets=targets)
