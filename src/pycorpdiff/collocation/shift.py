"""Cross-corpus collocation shift — gained / lost collocates of a target."""

from __future__ import annotations

import pandas as pd

from ..corpus import Corpus, CorpusSlice


def collocation_shift(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str,
    window: int = 5,
    measure: str = "logDice",
    min_count: int = 5,
) -> pd.DataFrame:
    """Compute the change in target-term collocates between two corpora."""
    raise NotImplementedError("collocation_shift() lands in Phase 2")
