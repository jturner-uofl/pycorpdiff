"""Explainability helpers — KWIC concordances, representative documents.

Phase 3 will populate this module. Each public analytical Result delegates
its ``.explain()`` method here so the concordance machinery lives in one
place.
"""

from __future__ import annotations

from .corpus import Corpus, CorpusSlice
from .results import ConcordanceResult


def kwic(
    corpus: Corpus | CorpusSlice,
    target: str,
    window: int = 5,
    n: int | None = None,
) -> ConcordanceResult:
    """Return KWIC concordance lines for ``target`` in ``corpus``."""
    raise NotImplementedError("kwic() lands in Phase 3")


def representative_docs(
    corpus: Corpus | CorpusSlice,
    target: str,
    n: int = 5,
) -> list[str]:
    """Return up to ``n`` documents most representative of ``target``."""
    raise NotImplementedError("representative_docs() lands in Phase 3")
