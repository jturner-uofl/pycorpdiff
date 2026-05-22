"""Embedder protocol + a lazy sentence-transformers default.

The :class:`Embedder` protocol is the package's plug point for vector
representations. Anything implementing
``encode(terms: Sequence[str]) -> np.ndarray`` of shape ``(n, d)``
satisfies it — gensim KeyedVectors with a thin adapter, HuggingFace
pipelines, etc.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps a sequence of terms to a 2-D vector array."""

    def encode(self, terms: Sequence[str]) -> np.ndarray: ...


@dataclass
class SBERTEmbedder:
    """Default :class:`Embedder` backed by sentence-transformers.

    Imported and instantiated lazily on first call to :meth:`encode` so
    the base install does not pull torch transitively.
    """

    model_name: str = "all-MiniLM-L6-v2"

    def encode(self, terms: Sequence[str]) -> np.ndarray:
        raise NotImplementedError("SBERTEmbedder.encode() lands in Phase 6")
