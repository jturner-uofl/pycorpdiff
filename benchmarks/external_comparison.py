"""Head-to-head performance benchmarks vs NLTK on a shared real fixture.

The asv suite in :mod:`benchmarks.benchmarks` measures regressions
within pycorpdiff over time. This script measures *cross-package*
performance — pycorpdiff vs NLTK on identical inputs — on the
:file:`paper/replication/data/conventions_2012.parquet` snapshot
(189 speeches, ~135K tokens). The resulting table appears in
:numref:`tab:benchmarks` of the paper.

Run from the repository root::

    pip install -e ".[viz,temporal]" nltk
    python benchmarks/external_comparison.py

Outputs a small markdown table to stdout and writes the same data to
:file:`benchmarks/external_comparison_results.json` so the paper's
LaTeX can read it directly.

NLTK is the obvious cross-tool comparator for collocation measures
(PMI, t-score, MI³) since pycorpdiff already cross-validates against
NLTK's :class:`BigramAssocMeasures` for *correctness* in
:file:`tests/integration/test_crossval_nltk.py`. Here we measure how
long the same computation takes end-to-end.

quanteda comparisons (G² log-likelihood, dispersion) require R + the
quanteda package; those benchmarks live in
:file:`benchmarks/external_comparison_quanteda.R` and are run by
hand or via rpy2 on a CI tier with R available. Numerical agreement
is already verified by
:file:`tests/integration/test_crossval_quanteda.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd

import pycorpdiff as pcd

DATA_PATH = (
    Path(__file__).parent.parent
    / "paper"
    / "replication"
    / "data"
    / "conventions_2012.parquet"
)
RESULTS_PATH = Path(__file__).parent / "external_comparison_results.json"
N_REPEATS = 5


def _time_it(fn: Callable[[], object], repeats: int = N_REPEATS) -> tuple[float, float]:
    """Time ``fn`` ``repeats`` times. Return (mean, min) wall-time seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), min(times)


def benchmark_pmi(tokens: list[str]) -> dict[str, tuple[float, float]]:
    """Compute PMI for the bigrams in ``tokens`` two ways.

    pycorpdiff path: build a Series of bigram counts, call
    :func:`pycorpdiff.collocation.pmi` directly. NLTK path: call
    ``BigramCollocationFinder`` + ``BigramAssocMeasures.pmi``. Both
    operate on the same flat token stream so results are comparable.
    """
    results: dict[str, tuple[float, float]] = {}

    # pycorpdiff path — compute marginals + bigram counts then call pmi().
    def pcd_pmi() -> None:
        from collections import Counter

        from pycorpdiff.collocation.measures import pmi

        n = len(tokens)
        unigram = Counter(tokens)
        bigram = Counter(zip(tokens, tokens[1:], strict=False))
        bigram_counts = {b: c for b, c in bigram.items() if c >= 3}
        if not bigram_counts:
            return
        f_xy = pd.Series(
            {f"{b[0]}_{b[1]}": c for b, c in bigram_counts.items()}, dtype=float
        )
        f_y = pd.Series(
            {f"{b[0]}_{b[1]}": unigram[b[1]] for b in bigram_counts}, dtype=float
        )
        f_x_map = {b: unigram[b[0]] for b in bigram_counts}
        # pmi is vectorised on f_xy + f_y for a fixed f_x; iterate by f_x.
        seen: set[float] = set()
        for b, _ in bigram_counts.items():
            fx_val = float(f_x_map[b])
            if fx_val in seen:
                continue
            seen.add(fx_val)
            mask = [
                bb for bb in bigram_counts
                if float(unigram[bb[0]]) == fx_val
            ]
            keys = [f"{bb[0]}_{bb[1]}" for bb in mask]
            pmi(f_xy.loc[keys], fx_val, f_y.loc[keys], n)

    # NLTK path.
    def nltk_pmi() -> None:
        from nltk.collocations import (
            BigramAssocMeasures,
            BigramCollocationFinder,
        )

        finder = BigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(3)
        finder.score_ngrams(BigramAssocMeasures.pmi)

    results["pycorpdiff"] = _time_it(pcd_pmi)
    results["nltk"] = _time_it(nltk_pmi)
    return results


def benchmark_keyness(corpus: pcd.Corpus) -> dict[str, tuple[float, float]]:
    """Time the full keyness pipeline on the conventions corpus."""
    dem = corpus.slice(party="democrat")
    rep = corpus.slice(party="republican")

    def pcd_keyness() -> None:
        pcd.compare(dem, rep).keyness(min_count=10, effect_size=True)

    return {"pycorpdiff": _time_it(pcd_keyness)}


def benchmark_collocation_shift(corpus: pcd.Corpus) -> dict[str, tuple[float, float]]:
    """Time collocation_shift between two slices."""
    dem = corpus.slice(party="democrat")
    rep = corpus.slice(party="republican")

    def pcd_shift() -> None:
        pcd.compare(dem, rep).collocation_shift(
            "jobs", window=4, min_count=3, measure="logDice"
        )

    return {"pycorpdiff": _time_it(pcd_shift)}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark fixture not found at {DATA_PATH}. "
            "Run paper/replication/reproduce.py first."
        )

    df = pd.read_parquet(DATA_PATH)
    corpus = pcd.from_dataframe(
        df, text_col="text", meta_cols=("party", "speaker", "year")
    )
    flat_tokens: list[str] = [
        t for doc in corpus.tokens() for t in doc
    ]
    print(
        f"# Fixture: 2012 US Conventions — "
        f"{len(corpus):,} docs · {len(flat_tokens):,} tokens · "
        f"{len(set(flat_tokens)):,} types"
    )
    print(f"# {N_REPEATS} repeats per cell; reporting (mean, min) seconds.\n")

    benchmarks = {
        "PMI (adjacent bigrams, min_count=3)": benchmark_pmi(flat_tokens),
        "Keyness (Dem vs Rep, min_count=10, +effect_size)": benchmark_keyness(corpus),
        "Collocation shift ('jobs', window=4)": benchmark_collocation_shift(corpus),
    }

    # Markdown table.
    print("| Workload | Tool | Mean (s) | Min (s) |")
    print("|---|---|---:|---:|")
    for label, tool_results in benchmarks.items():
        for tool, (mean_s, min_s) in tool_results.items():
            print(
                f"| {label} | {tool} | {mean_s:.4f} | {min_s:.4f} |"
            )

    # JSON for paper to consume.
    json_out = {
        label: {
            tool: {"mean_s": round(mean_s, 6), "min_s": round(min_s, 6)}
            for tool, (mean_s, min_s) in tool_results.items()
        }
        for label, tool_results in benchmarks.items()
    }
    json_out["_fixture"] = {  # type: ignore[assignment]
        "n_documents": len(corpus),
        "n_tokens": len(flat_tokens),
        "n_types": len(set(flat_tokens)),
        "n_repeats": N_REPEATS,
        "source": "scattertext.SampleCorpora.ConventionData2012",
    }
    RESULTS_PATH.write_text(json.dumps(json_out, indent=2))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
