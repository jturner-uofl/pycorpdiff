"""Cross-validation against NLTK's collocation measures.

NLTK ships ``BigramCollocationFinder`` + ``BigramAssocMeasures`` as the
de facto Python reference for collocation association scores. Both
packages compute PMI and t-score from the same five-argument shape
(joint count, two marginals, total tokens), so for any fixed set of
counts we should agree *to floating-point noise*.

We assert numerical agreement on every bigram that survives a minimum
frequency filter — no ranking-overlap weasel words, the actual values
must match.

This test ships in the slow tier because NLTK is an opt-in dependency.
Skip silently if it isn't installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pycorpdiff.collocation.measures import mi_three, pmi, t_score

nltk = pytest.importorskip("nltk")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fixture_tokens() -> list[str]:
    """A repeatable token stream with several genuine collocations."""
    base = [
        "the", "cat", "sat", "on", "the", "mat",
        "the", "cat", "ran", "fast",
        "the", "dog", "sat", "by", "the", "door",
    ]
    # Repeat enough that frequency filters leave non-trivial structure.
    return base * 20


def _build_finder(tokens: list[str]) -> object:
    from nltk.collocations import BigramCollocationFinder

    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(3)
    return finder


def test_pmi_agrees_with_nltk_exactly(fixture_tokens: list[str]) -> None:
    """For every adjacent bigram surviving the freq filter, our PMI ==
    NLTK's PMI to floating-point precision (≤ 1e-12).
    """
    from nltk.collocations import BigramAssocMeasures

    finder = _build_finder(fixture_tokens)
    measures = BigramAssocMeasures()
    nltk_pmi = dict(finder.score_ngrams(measures.pmi))
    total_n = finder.N

    assert nltk_pmi, "no bigrams survived the frequency filter — fixture too small"

    max_abs_diff = 0.0
    for (w1, w2), nltk_score in nltk_pmi.items():
        f_xy = finder.ngram_fd[(w1, w2)]
        f_x = finder.word_fd[w1]
        f_y_series = pd.Series({w2: finder.word_fd[w2]}, dtype=float)
        f_xy_series = pd.Series({w2: f_xy}, dtype=float)
        our_score = pmi(f_xy_series, f_x, f_y_series, total_n)[w2]
        max_abs_diff = max(max_abs_diff, abs(our_score - nltk_score))
    assert max_abs_diff < 1e-12, f"max PMI divergence from NLTK: {max_abs_diff}"


def test_t_score_agrees_with_nltk_exactly(fixture_tokens: list[str]) -> None:
    """Same story for Student's t-score: pycorpdiff and NLTK use the
    identical ``(f_xy − E) / √f_xy`` formula, so agreement must be exact.
    """
    from nltk.collocations import BigramAssocMeasures

    finder = _build_finder(fixture_tokens)
    measures = BigramAssocMeasures()
    nltk_t = dict(finder.score_ngrams(measures.student_t))
    total_n = finder.N

    assert nltk_t, "no bigrams survived the frequency filter"

    max_abs_diff = 0.0
    for (w1, w2), nltk_score in nltk_t.items():
        f_xy = finder.ngram_fd[(w1, w2)]
        f_x = finder.word_fd[w1]
        f_y_series = pd.Series({w2: finder.word_fd[w2]}, dtype=float)
        f_xy_series = pd.Series({w2: f_xy}, dtype=float)
        our_score = t_score(f_xy_series, f_x, f_y_series, total_n)[w2]
        max_abs_diff = max(max_abs_diff, abs(our_score - nltk_score))
    assert max_abs_diff < 1e-12, f"max t-score divergence from NLTK: {max_abs_diff}"


def test_mi3_matches_nltk_mi_like_up_to_log_scaling(
    fixture_tokens: list[str],
) -> None:
    """NLTK doesn't expose Daille's MI³ directly, but its ``mi_like``
    family computes ``f_xy^power / (f_x · f_y)``. Daille's MI³ is
    ``log2( f_xy³ · N / (f_x · f_y) )``, so the two are related by
    ``mi3 == log2(N · mi_like(power=3))``. Asserting this identity
    catches accidental changes to our log-base or numerator structure.
    """
    import math

    from nltk.collocations import BigramAssocMeasures

    finder = _build_finder(fixture_tokens)
    measures = BigramAssocMeasures()
    nltk_mi_like = dict(
        finder.score_ngrams(lambda *m: measures.mi_like(*m, power=3))
    )
    total_n = finder.N

    assert nltk_mi_like, "no bigrams survived the frequency filter"

    max_abs_diff = 0.0
    for (w1, w2), nltk_score in nltk_mi_like.items():
        f_xy = finder.ngram_fd[(w1, w2)]
        f_x = finder.word_fd[w1]
        f_y_series = pd.Series({w2: finder.word_fd[w2]}, dtype=float)
        f_xy_series = pd.Series({w2: f_xy}, dtype=float)
        our_mi3 = mi_three(f_xy_series, f_x, f_y_series, total_n)[w2]
        expected_mi3 = math.log2(total_n * nltk_score)
        max_abs_diff = max(max_abs_diff, abs(our_mi3 - expected_mi3))
    assert max_abs_diff < 1e-12, f"MI³ ↔ mi_like identity divergence: {max_abs_diff}"


def test_top_pmi_rankings_align_with_nltk(fixture_tokens: list[str]) -> None:
    """Sanity ranking check: the top-5 PMI bigrams must be identical
    between the two packages (since the *scores* are identical above,
    so must be the ordering — but this assertion catches future bugs
    in our index alignment more clearly than a numerical-equality
    failure).
    """
    from nltk.collocations import BigramAssocMeasures

    finder = _build_finder(fixture_tokens)
    measures = BigramAssocMeasures()
    nltk_pmi = finder.score_ngrams(measures.pmi)
    nltk_top5 = [pair for pair, _ in nltk_pmi[:5]]

    # Recompute with pycorpdiff scores and rank.
    rows = []
    total_n = finder.N
    for (w1, w2), _ in nltk_pmi:
        f_xy = finder.ngram_fd[(w1, w2)]
        f_x = finder.word_fd[w1]
        f_y_series = pd.Series({w2: finder.word_fd[w2]}, dtype=float)
        f_xy_series = pd.Series({w2: f_xy}, dtype=float)
        rows.append(
            ((w1, w2), float(pmi(f_xy_series, f_x, f_y_series, total_n)[w2]))
        )
    our_top5 = [pair for pair, _ in sorted(rows, key=lambda r: -r[1])][:5]
    assert our_top5 == nltk_top5
