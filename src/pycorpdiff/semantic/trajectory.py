"""Multi-period semantic trajectories of target terms."""

from __future__ import annotations

import pandas as pd

from ..corpus import Corpus
from .embed import Embedder


def semantic_trajectory(
    corpus: Corpus,
    target: str | list[str],
    time_col: str = "date",
    freq: str = "Y",
    embedder: Embedder | None = None,
    align: str = "procrustes",
) -> pd.DataFrame:
    """Trajectory of target term(s) embedding across time periods."""
    raise NotImplementedError("semantic_trajectory() lands in Phase 6")
