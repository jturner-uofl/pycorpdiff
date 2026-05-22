"""Embedding-based semantic shift and trajectory analysis."""

from __future__ import annotations

from .alignment import procrustes_align
from .embed import Embedder, SBERTEmbedder
from .shift import neighborhood_drift, semantic_shift
from .trajectory import semantic_trajectory

__all__ = [
    "Embedder",
    "SBERTEmbedder",
    "neighborhood_drift",
    "procrustes_align",
    "semantic_shift",
    "semantic_trajectory",
]
