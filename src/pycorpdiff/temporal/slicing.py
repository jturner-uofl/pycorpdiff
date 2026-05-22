"""Temporal slicing primitives — ``TemporalCorpus``, ``track``, ``Tracker``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..corpus import Corpus
    from ..results import TemporalTrajectory


@dataclass(frozen=True)
class TemporalCorpus:
    """A corpus indexed by time period for diachronic analysis.

    Constructed via :meth:`pycorpdiff.Corpus.by_time` rather than directly.
    """

    parent: Corpus
    time_col: str
    freq: str = "Y"

    def __len__(self) -> int:
        return len(self.parent)


@dataclass(frozen=True)
class Tracker:
    """A diachronic tracker over one or more target terms."""

    corpus: Corpus
    targets: list[str]

    def over_time(self, freq: str = "Y", time_col: str = "date") -> TemporalTrajectory:
        """Return a :class:`TemporalTrajectory` of relative frequencies."""
        raise NotImplementedError("Tracker.over_time() lands in Phase 4")

    def trajectory(self, freq: str = "Y", time_col: str = "date") -> TemporalTrajectory:
        """Alias for :meth:`over_time` — kept for API symmetry with prior art."""
        return self.over_time(freq=freq, time_col=time_col)


def track(corpus: Corpus, target: str | list[str]) -> Tracker:
    """Construct a :class:`Tracker` for diachronic analysis of target term(s)."""
    targets = [target] if isinstance(target, str) else list(target)
    return Tracker(corpus=corpus, targets=targets)
