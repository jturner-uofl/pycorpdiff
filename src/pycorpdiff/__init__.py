"""pycorpdiff — comparative corpus analysis for modern Python workflows.

The package exposes three public verbs (:func:`compare`, :func:`track`,
:func:`compare.before_after`) and a family of frozen-dataclass
result objects (:class:`KeynessResult`, :class:`CollocationShiftResult`,
:class:`SemanticShiftResult`, :class:`TemporalTrajectory`,
:class:`NetworkResult`, :class:`ForecastResult`,
:class:`CausalImpactResult`, :class:`BocpdResult`,
:class:`ConcordanceResult`), each implementing the relevant subset of
the ``.to_df / .plot / .explain / .summary / .to_html / .to_json``
contract. See ``docs/design.md`` for the per-Result method matrix.

Example
-------

>>> import pycorpdiff as pcd
>>> isinstance(pcd.__version__, str)
True
"""

from __future__ import annotations

__version__ = "0.1.0a32"

from .collocation.network import NetworkResult, cooccurrence_network
from .compare import Comparison, compare
from .corpus import Corpus, CorpusSlice
from .datasets import (
    fetch_hansard,
    fetch_histwords_decade,
    histwords_cosine_shift,
    load_hansard_sample,
)
from .datasets.baselines import Baseline, baseline_from_corpus, list_baselines, load_baseline
from .explain import kwic, representative_docs
from .io.duckdb import read_duckdb
from .io.huggingface import from_huggingface
from .io.readers import from_dataframe, read_csv, read_parquet, read_txt
from .keyness.baseline import against_baseline
from .keyness.multicorpus import keyness_multi
from .lexical import (
    LexicalDiversityResult,
    LexicalDiversityTrajectory,
    lexical_diversity,
)
from .matching import MatchResult, match
from .results import (
    CollocationShiftResult,
    ConcordanceResult,
    KeynessResult,
    SemanticShiftResult,
    TemporalTrajectory,
)
from .semantic.drift import SenseDriftResult, sense_drift
from .semantic.embed import Embedder, HashEmbedder, SBERTEmbedder
from .semantic.senses import SenseAgreement, SenseInductionResult, induce_senses
from .semantic.shift import neighborhood_drift
from .semantic.trajectory import semantic_trajectory
from .temporal.bocpd import BocpdResult, bocpd
from .temporal.burstiness import BurstinessResult, kleinberg_bursts
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
    burstiness_plot,
    causal_impact_plot,
    dispersion_plot,
    forecast_plot,
    network_plot,
    scattertext_plot,
    semantic_forecast_plot,
)

__all__ = [
    "Baseline",
    "BocpdResult",
    "BurstinessResult",
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
    "LexicalDiversityResult",
    "LexicalDiversityTrajectory",
    "MatchResult",
    "NetworkResult",
    "NgramTokenizer",
    "RegexTokenizer",
    "SBERTEmbedder",
    "SenseAgreement",
    "SenseDriftResult",
    "SenseInductionResult",
    "SemanticShiftResult",
    "TemporalCorpus",
    "TemporalTrajectory",
    "Tokenizer",
    "__version__",
    "against_baseline",
    "baseline_from_corpus",
    "bocpd",
    "bocpd_plot",
    "burstiness_plot",
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
    "induce_senses",
    "keyness_multi",
    "kleinberg_bursts",
    "lexical_diversity",
    "kwic",
    "list_baselines",
    "load_baseline",
    "load_hansard_sample",
    "match",
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
    "sense_drift",
    "track",
]
