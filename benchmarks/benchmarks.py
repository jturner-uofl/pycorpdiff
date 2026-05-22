"""Airspeed-velocity benchmarks for pycorpdiff.

Run locally:

    asv dev              # quick run against current source
    asv run --quick HEAD # one iteration per benchmark, written to .asv/results
    asv publish          # render HTML report at .asv/html/

The synthetic-corpus generator below is deterministic; benchmark
timings should be reproducible across machines modulo CPU speed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pycorpdiff as pcd


def _synthetic_corpus(
    n_docs: int, avg_tokens: int = 80, vocab_size: int = 500, seed: int = 0
) -> pcd.Corpus:
    """Build a deterministic corpus of given size and vocabulary."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(vocab_size)]
    rows: list[dict[str, object]] = []
    for i in range(n_docs):
        n = max(1, int(rng.normal(avg_tokens, max(1, avg_tokens // 4))))
        tokens = rng.choice(vocab, size=n).tolist()
        rows.append(
            {
                "text": " ".join(tokens),
                "outlet": "A" if (i % 2 == 0) else "B",
                "year": 2010 + (i % 14),
            }
        )
    return pcd.from_dataframe(
        pd.DataFrame(rows), text_col="text", meta_cols=("outlet", "year")
    )


class CorpusConstruction:
    """How long does it take to derive vocab / doc-term matrices?"""

    params = [100, 1_000, 10_000]
    param_names = ["n_docs"]

    def setup(self, n_docs: int) -> None:
        self.corpus = _synthetic_corpus(n_docs)

    def time_doc_term_counts(self, n_docs: int) -> None:
        self.corpus.doc_term_counts()

    def time_vocab(self, n_docs: int) -> None:
        self.corpus.vocab()

    def time_total_tokens(self, n_docs: int) -> None:
        self.corpus.total_tokens()

    def peakmem_doc_term_counts(self, n_docs: int) -> None:
        self.corpus.doc_term_counts()


class Keyness:
    """compare(a, b).keyness() across a range of corpus sizes."""

    params = [100, 1_000, 10_000]
    param_names = ["n_docs"]

    def setup(self, n_docs: int) -> None:
        corpus = _synthetic_corpus(n_docs)
        self.a = corpus.slice(outlet="A")
        self.b = corpus.slice(outlet="B")

    def time_keyness_default(self, n_docs: int) -> None:
        pcd.compare(self.a, self.b).keyness()

    def time_keyness_with_dispersion(self, n_docs: int) -> None:
        pcd.compare(self.a, self.b).keyness(dispersion=True)


class CollocationShift:
    """compare(a, b).collocation_shift(target) — windowed co-occurrence."""

    params = [500, 5_000]
    param_names = ["n_docs"]

    def setup(self, n_docs: int) -> None:
        corpus = _synthetic_corpus(n_docs)
        self.a = corpus.slice(outlet="A")
        self.b = corpus.slice(outlet="B")
        # ``w100`` is in our 500-word vocab so will appear in most corpora.
        self.target = "w100"

    def time_collocation_shift(self, n_docs: int) -> None:
        pcd.compare(self.a, self.b).collocation_shift(
            self.target, window=5, min_count=2
        )


class TemporalTrack:
    """track(corpus, term).over_time() across a range of corpus sizes."""

    params = [1_000, 10_000]
    param_names = ["n_docs"]

    def setup(self, n_docs: int) -> None:
        self.corpus = _synthetic_corpus(n_docs)
        # Synthetic corpus has 'year' as int; track uses it as the time col.
        self.target = "w50"

    def time_track_over_time(self, n_docs: int) -> None:
        pcd.track(self.corpus, self.target).over_time(freq="Y", time_col="year")


class Tokenization:
    """The default RegexTokenizer at scale."""

    params = [1_000, 10_000]
    param_names = ["n_docs"]

    def setup(self, n_docs: int) -> None:
        self.corpus = _synthetic_corpus(n_docs)

    def time_tokens(self, n_docs: int) -> None:
        self.corpus.tokens()
