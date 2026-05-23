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

from .collocation.network import NetworkResult, cooccurrence_network
from .compare import Comparison, compare
from .corpus import Corpus, CorpusSlice
from .datasets import (
    fetch_hansard,
    fetch_histwords_decade,
    histwords_cosine_shift,
    load_hansard_sample,
)
from .explain import kwic, representative_docs
from .io.duckdb import read_duckdb
from .io.huggingface import from_huggingface
from .io.readers import from_dataframe, read_csv, read_parquet, read_txt
from .keyness.multicorpus import keyness_multi
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
from .temporal.bocpd import BocpdResult, bocpd
from .temporal.causal_impact import CausalImpactResult, causal_impact
from .temporal.forecast import (
    ForecastResult,
    forecast_semantic_drift,
    forecast_trajectory,
)
from .temporal.slicing import TemporalCorpus, track
from .tokenize import NgramTokenizer, RegexTokenizer, Tokenizer

# Convenience aliases for the standalone (non-`.plot()`-delegated)
# visualisation functions. The full plot family also lives at
# ``pycorpdiff.viz.*`` — these are surfaced at the root so common
# patterns like ``pcd.dispersion_plot(corpus, term)`` work without
# requiring a separate import path. Plots that are only meaningful
# as ``Result.plot()`` (keyness_volcano, keyness_top_n_bar,
# collocation_diverging_bar, trajectory_with_ci) stay in ``viz`` only.
from .viz import (
    bocpd_plot,
    causal_impact_plot,
    dispersion_plot,
    forecast_plot,
    network_plot,
    scattertext_plot,
    semantic_forecast_plot,
)

__all__ = [
    "BocpdResult",
    "CausalImpactResult",
    "CollocationShiftResult",
    "Comparison",
    "ConcordanceResult",
    "Corpus",
    "CorpusSlice",
    "Embedder",
    "ForecastResult",
    "HashEmbedder",
    "KeynessResult",
    "NetworkResult",
    "NgramTokenizer",
    "RegexTokenizer",
    "SBERTEmbedder",
    "SemanticShiftResult",
    "TemporalCorpus",
    "TemporalTrajectory",
    "Tokenizer",
    "__version__",
    "bocpd",
    "bocpd_plot",
    "causal_impact",
    "causal_impact_plot",
    "compare",
    "cooccurrence_network",
    "dispersion_plot",
    "fetch_hansard",
    "fetch_histwords_decade",
    "forecast_plot",
    "forecast_semantic_drift",
    "forecast_trajectory",
    "from_dataframe",
    "from_huggingface",
    "histwords_cosine_shift",
    "keyness_multi",
    "kwic",
    "load_hansard_sample",
    "neighborhood_drift",
    "network_plot",
    "read_csv",
    "read_duckdb",
    "read_parquet",
    "read_txt",
    "representative_docs",
    "scattertext_plot",
    "semantic_forecast_plot",
    "semantic_trajectory",
    "track",
]
