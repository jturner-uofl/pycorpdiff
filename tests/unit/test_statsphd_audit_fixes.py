"""Regression tests for bugs surfaced by the Stats/CS PhD adversarial audit.

Each test corresponds to a specific finding that, prior to the fix,
either silently returned wrong values, accepted invalid inputs without
complaint, or violated a documented contract. Keep these as trip-wires
so the same problem doesn't regress.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff.collocation.measures import logdice, mi_three, pmi
from pycorpdiff.stats import wilson_ci
from pycorpdiff.temporal.bocpd import bocpd

# ---------------------------------------------------------------------
# MI³ / PMI / t-score int64 overflow no longer produces silent-wrong
# values on large-corpus inputs (audit finding #2).
# ---------------------------------------------------------------------


def test_mi_three_does_not_overflow_int64() -> None:
    """Pre-fix: f_xy=10^7 (int64) cubed overflows to negative; MI³
    returned ≈ 10.99 instead of the correct ≈ 48.51."""
    f_xy = pd.Series({"x": 10_000_000}, dtype=np.int64)
    f_x = np.int64(20_000_000)
    f_y = pd.Series({"x": 30_000_000}, dtype=np.int64)
    n = np.int64(10_000_000_000_000)  # 10^13
    result = float(mi_three(f_xy, f_x, f_y, n).iloc[0])
    # Correct value is log2((1e7)^3 * 1e13 / (2e7 * 3e7)) = log2(1e34 / 6e14) ≈ 64.5
    assert result > 60.0, (
        f"MI³ on large int64 counts should be high (>60); got {result}. "
        "The pre-fix code returned a small or negative value due to "
        "silent int64 overflow on f_xy**3."
    )


def test_pmi_does_not_overflow_int64() -> None:
    """Pre-fix: PMI with large int64 inputs overflowed via f_x*f_y."""
    f_xy = pd.Series({"x": 100_000_000}, dtype=np.int64)
    f_x = np.int64(10_000_000_000)
    f_y = pd.Series({"x": 10_000_000_000}, dtype=np.int64)
    n = np.int64(10_000_000_000_000)
    result = float(pmi(f_xy, f_x, f_y, n).iloc[0])
    # Expected: log2(1e8 * 1e13 / (1e10 * 1e10)) = log2(10) ≈ 3.32
    assert 3.0 < result < 3.5, (
        f"PMI on large int64 should be ≈ 3.32; got {result}. "
        "Pre-fix code overflowed via f_x*f_y."
    )


# ---------------------------------------------------------------------
# Corpus hash now distinguishes row order (audit finding #3).
# ---------------------------------------------------------------------


def test_corpus_hash_distinguishes_row_permutations() -> None:
    """Pre-fix: __hash__ used .sum() over row hashes (permutation-
    invariant) so a shuffled corpus hashed equal — breaking cache-key
    correctness for users keying memoised results by Corpus identity."""
    c1 = pcd.from_dataframe(
        pd.DataFrame({"text": ["alpha", "beta", "gamma", "delta", "epsilon"]}),
        text_col="text",
    )
    c2 = pcd.from_dataframe(
        pd.DataFrame({"text": ["epsilon", "delta", "gamma", "beta", "alpha"]}),
        text_col="text",
    )
    assert hash(c1) != hash(c2), (
        "Corpus.__hash__ must distinguish row order; otherwise the "
        "docstring's cache-key promise is false."
    )
    assert c1 != c2


def test_corpus_hash_stable_under_identical_construction() -> None:
    """Two corpora built from the same DataFrame in the same order
    must still hash identically — the fix preserves this property."""
    df = pd.DataFrame({"text": ["a", "b", "c"]})
    c1 = pcd.from_dataframe(df.copy(), text_col="text")
    c2 = pcd.from_dataframe(df.copy(), text_col="text")
    assert hash(c1) == hash(c2)
    assert c1 == c2


# ---------------------------------------------------------------------
# RegexTokenizer normalizes Unicode (audit finding #6).
# ---------------------------------------------------------------------


def test_regex_tokenizer_normalizes_nfc_vs_nfd_to_same_token() -> None:
    """Pre-fix: "café" composed (NFC, é = U+00E9) and "café" decomposed
    (NFD, e + U+0301 combining acute) tokenized to different tokens
    because \\w+ doesn't match the combining acute. Mixed-source
    corpora produced wrong term-frequency tables."""
    tok = pcd.RegexTokenizer()
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    # Sanity: they differ at byte level pre-normalization.
    assert nfc != nfd
    # Post-normalization: tokenizer should produce the same output.
    assert tok(nfc) == tok(nfd)


def test_regex_tokenizer_normalize_skip_when_empty_string() -> None:
    """A user can opt out of normalization with ``normalize=''``."""
    tok = pcd.RegexTokenizer(normalize="")
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    # With normalization off, the two encodings tokenize differently.
    assert tok(nfc) != tok(nfd)


def test_regex_tokenizer_rejects_unknown_normalize_form() -> None:
    with pytest.raises(ValueError, match="normalize must be one of"):
        pcd.RegexTokenizer(normalize="BOGUS")


# ---------------------------------------------------------------------
# HashEmbedder no longer collides at 65k+ terms (audit finding #8).
# ---------------------------------------------------------------------


def test_hash_embedder_unique_seeds_at_100k() -> None:
    """Pre-fix: SHA-256 prefix was masked to 32 bits, producing
    birthday-paradox collisions at ~65k terms. 100k terms had a real
    chance of returning identical embeddings for unrelated strings."""
    # Use a deterministic synthetic vocabulary at the size where the
    # pre-fix 32-bit mask started to collide reliably.
    n = 100_000
    terms = [f"term_{i:08d}" for i in range(n)]
    embedder = pcd.HashEmbedder(dim=8)  # small dim — collision test is on the seed, not the vector
    vecs = embedder.encode(terms)
    # Hash each row to a tuple so we can count distinct embeddings.
    unique = {tuple(row.tolist()) for row in vecs}
    assert len(unique) == n, (
        f"HashEmbedder produced only {len(unique)} unique embeddings "
        f"for {n} distinct terms; the SHA-256 prefix must not be "
        "masked below 64 bits."
    )


# ---------------------------------------------------------------------
# Wilson CI validates inputs (audit finding #10).
# ---------------------------------------------------------------------


def test_wilson_ci_rejects_x_greater_than_n() -> None:
    """A user who swaps argument order (wilson_ci(n, x) instead of
    wilson_ci(x, n)) would otherwise silently get NaNs from the
    arithmetic. Raise clearly."""
    with pytest.raises(ValueError, match="must not exceed"):
        wilson_ci(150, 100)


def test_wilson_ci_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        wilson_ci(-1, 100)
    with pytest.raises(ValueError, match="non-negative"):
        wilson_ci(5, -100)


def test_wilson_ci_zero_count_lower_bound_near_zero() -> None:
    """Lower bound at count=0 should be 0 to floating-point precision
    (docs/statistical-methods.md documents this — values like 5e-17
    are normal and the bound is clamped to [0,1])."""
    lo, hi = wilson_ci(0, 100)
    assert float(lo) < 1e-10
    assert 0.0 < float(hi) < 0.1


# ---------------------------------------------------------------------
# logDice rejects vacuous contingency table (audit finding #12).
# ---------------------------------------------------------------------


def test_logdice_rejects_zero_marginals() -> None:
    """Pre-fix: logDice with f_x + f_y == 0 silently returned +inf
    via division by zero; the formula is undefined there."""
    f_xy = pd.Series({"x": 5})
    with pytest.raises(ValueError, match="vacuous"):
        logdice(f_xy, 0.0, pd.Series({"x": 0}))


# ---------------------------------------------------------------------
# track() rejects None / empty target (audit finding #14).
# ---------------------------------------------------------------------


def test_track_rejects_none_target() -> None:
    corpus = pcd.load_hansard_sample()
    with pytest.raises(ValueError, match="requires a target"):
        pcd.track(corpus, None)  # type: ignore[arg-type]


def test_track_rejects_empty_string_target() -> None:
    corpus = pcd.load_hansard_sample()
    with pytest.raises(ValueError, match="non-empty"):
        pcd.track(corpus, "")


def test_track_rejects_empty_list_target() -> None:
    corpus = pcd.load_hansard_sample()
    with pytest.raises(ValueError, match="non-empty"):
        pcd.track(corpus, [])


# ---------------------------------------------------------------------
# BocpdResult.cp_probability_recent is data-driven (audit finding #5).
# ---------------------------------------------------------------------


def test_bocpd_cp_probability_recent_responds_to_changepoints() -> None:
    """Pre-fix: ``cp_probability`` was mathematically identical to the
    hazard hyperparameter (the changepoint prior cancels in the
    posterior normalisation under constant hazard). The new
    ``cp_probability_recent(threshold=k)`` summarises the full
    run-length posterior and DOES respond to regime changes."""
    # Construct a series with a sharp shift halfway through.
    rng = np.random.default_rng(seed=42)
    pre = rng.normal(loc=0.0, scale=0.1, size=30)
    post = rng.normal(loc=2.0, scale=0.1, size=30)
    series = pd.Series(np.concatenate([pre, post]), name="value")

    result = bocpd(series, hazard=0.01)
    recent = result.cp_probability_recent(threshold=3)

    # The new diagnostic should differ across the series — it responds
    # to the regime shift around index 30.
    assert recent.nunique() > 5, (
        f"cp_probability_recent should respond to data; got "
        f"{recent.nunique()} unique values across the series."
    )
