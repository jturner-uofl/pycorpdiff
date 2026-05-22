"""Semantic shift and neighborhood drift between corpora."""

from __future__ import annotations

import pandas as pd

from ..corpus import Corpus, CorpusSlice
from .embed import Embedder


def semantic_shift(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str | list[str],
    embedder: Embedder | None = None,
    align: str = "procrustes",
) -> pd.DataFrame:
    """Embedding-space displacement of target term(s) between corpora."""
    raise NotImplementedError("semantic_shift() lands in Phase 6")


def neighborhood_drift(
    a: Corpus | CorpusSlice,
    b: Corpus | CorpusSlice,
    target: str,
    k: int = 10,
    embedder: Embedder | None = None,
) -> pd.DataFrame:
    """Change in the k-nearest-neighbour set of ``target`` (Jaccard / set diff)."""
    raise NotImplementedError("neighborhood_drift() lands in Phase 6")
