"""pycorpdiff — comparative corpus analysis for modern Python workflows.

The package exposes three public verbs (:func:`compare`, :func:`track`,
plus the :class:`Corpus` constructor and the I/O ``read_*`` helpers) and
four families of result objects (:class:`KeynessResult`,
:class:`CollocationShiftResult`, :class:`SemanticShiftResult`,
:class:`TemporalTrajectory`).

Layer-1 ingestion utilities are functional in this scaffolding release;
Layer-2 analytical methods raise :class:`NotImplementedError` until Phase 1
of the roadmap lands.

Example
-------

>>> import pycorpdiff as pcd
>>> pcd.__version__
'0.1.0a0'
"""

from __future__ import annotations

__version__ = "0.1.0a0"

from .compare import Comparison, compare
from .corpus import Corpus, CorpusSlice
from .datasets import load_hansard_sample
from .explain import kwic, representative_docs
from .io.duckdb import read_duckdb
from .io.readers import from_dataframe, read_csv, read_parquet, read_txt
from .results import (
    CollocationShiftResult,
    ConcordanceResult,
    KeynessResult,
    SemanticShiftResult,
    TemporalTrajectory,
)
from .semantic.embed import Embedder, HashEmbedder, SBERTEmbedder
from .semantic.shift import neighborhood_drift
from .semantic.trajectory import semantic_trajectory
from .temporal.slicing import TemporalCorpus, track
from .tokenize import RegexTokenizer, Tokenizer

__all__ = [
    "CollocationShiftResult",
    "Comparison",
    "ConcordanceResult",
    "Corpus",
    "CorpusSlice",
    "Embedder",
    "HashEmbedder",
    "KeynessResult",
    "RegexTokenizer",
    "SBERTEmbedder",
    "SemanticShiftResult",
    "TemporalCorpus",
    "TemporalTrajectory",
    "Tokenizer",
    "__version__",
    "compare",
    "from_dataframe",
    "kwic",
    "load_hansard_sample",
    "neighborhood_drift",
    "read_csv",
    "read_duckdb",
    "read_parquet",
    "read_txt",
    "representative_docs",
    "semantic_trajectory",
    "track",
]
