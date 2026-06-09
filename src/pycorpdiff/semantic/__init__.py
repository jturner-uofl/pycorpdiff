"""Embedding-based semantic shift and trajectory analysis."""

from __future__ import annotations

from .alignment import procrustes_align
from .embed import Embedder, HashEmbedder, SBERTEmbedder
from .drift import SenseDriftResult, sense_drift
from .senses import SenseAgreement, SenseInductionResult, induce_senses
from .shift import neighborhood_drift, semantic_shift
from .trajectory import semantic_trajectory

__all__ = [
    "Embedder",
    "HashEmbedder",
    "SBERTEmbedder",
    "SenseAgreement",
    "SenseDriftResult",
    "SenseInductionResult",
    "induce_senses",
    "sense_drift",
    "neighborhood_drift",
    "procrustes_align",
    "semantic_shift",
    "semantic_trajectory",
]
