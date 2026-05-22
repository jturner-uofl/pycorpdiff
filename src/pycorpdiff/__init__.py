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
from .io.readers import from_dataframe, read_csv, read_parquet, read_txt
from .results import (
    CollocationShiftResult,
    ConcordanceResult,
    KeynessResult,
    SemanticShiftResult,
    TemporalTrajectory,
)
from .temporal.slicing import TemporalCorpus, track
from .tokenize import RegexTokenizer, Tokenizer

__all__ = [
    "CollocationShiftResult",
    "Comparison",
    "ConcordanceResult",
    "Corpus",
    "CorpusSlice",
    "KeynessResult",
    "RegexTokenizer",
    "SemanticShiftResult",
    "TemporalCorpus",
    "TemporalTrajectory",
    "Tokenizer",
    "__version__",
    "compare",
    "from_dataframe",
    "read_csv",
    "read_parquet",
    "read_txt",
    "track",
]
