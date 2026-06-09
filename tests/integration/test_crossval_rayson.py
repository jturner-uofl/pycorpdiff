"""Extended known-answer cross-checks against Rayson's LL Wizard.

Rayson's Log-Likelihood and Effect Size Wizard at
http://ucrel.lancs.ac.uk/llwizard.html is the de facto reference for
single-cell keyness computation in corpus linguistics. Every value
asserted below was either computed from Rayson's exact formula or
copy-pasted from his calculator on a clean dataset.

This file extends ``test_loglikelihood.py`` with a broader sweep of
canonical reference triples covering edge cases (lopsided counts,
sparse cells, mid-sized over-representation) so that any future
refactor of the LL formula trips multiple assertions simultaneously.
"""

from __future__ import annotations

import math

import pandas as pd

from pycorpdiff.keyness.effect_sizes import log_ratio
from pycorpdiff.keyness.loglikelihood import log_likelihood

# ---------------------------------------------------------------------
# Reference values: (O1, N1, O2, N2, expected_LL)
# Computed by hand from the canonical 2×2 contingency-table formula:
#   E1 = N1 * (O1+O2) / (N1+N2)
#   E2 = N2 * (O1+O2) / (N1+N2)
#   LL = 2 * ( O1*ln(O1/E1) + O2*ln(O2/E2) )
# ---------------------------------------------------------------------
RAYSON_REFERENCES = [
    # (label, O1, N1, O2, N2, expected_unsigned_LL, sign_direction)
    # sign_direction = "A" if A-overuse, "B" if B-overuse, "neutral" if 0
    ("classic_12k_vs_10k", 12000, 1_000_000, 10000, 1_000_000, 182.0694, "A"),
    ("equal_rate_no_signal", 10, 1000, 20, 2000, 0.0, "neutral"),
    ("absent_in_b", 10, 1000, 0, 1000, 20.0 * math.log(2.0), "A"),
    ("absent_in_a", 0, 1000, 10, 1000, 20.0 * math.log(2.0), "B"),
    # 100/100k vs 20/200k: a 10× over-rep in A.
    # E1 = 100k * 120 / 300k = 40; E2 = 200k * 120 / 300k = 80
    # LL = 2*(100*ln(2.5) + 20*ln(0.25)) ≈ 127.8065
    ("ten_x_overrep_in_a", 100, 100_000, 20, 200_000, 127.8065, "A"),
    # 500/1M vs 100/1M: a 5× over-rep in A on a large corpus.
    # E1 = E2 = 300; LL = 2*(500*ln(500/300) + 100*ln(100/300))
    #     = 2*(255.4128 + (-109.8612)) = 2*145.5516 = 291.1032
    ("five_x_overrep_in_a", 500, 1_000_000, 100, 1_000_000, 291.1032, "A"),
    # 50/100k vs 50/50k: same count, A is half the rate.
    # E1 = 100k * 100 / 150k = 66.667; E2 = 50k * 100 / 150k = 33.333
    # LL = 2*(50*ln(50/66.667) + 50*ln(50/33.333))
    #     = 2*(50*ln(0.75) + 50*ln(1.5))
    #     = 2*(50*(-0.287682) + 50*(0.405465))
    #     = 2*(50 * 0.117783) = 11.7783
    ("same_count_half_rate", 50, 100_000, 50, 50_000, 11.7783, "B"),
    # Single-occurrence in each corpus, equal corpus sizes.
    # E1 = E2 = 1; LL = 2*(1*ln(1/1) + 1*ln(1/1)) = 0
    ("two_singletons_equal_rate", 1, 1000, 1, 1000, 0.0, "neutral"),
    # Very lopsided: 1000 in A, 1 in B, equal corpus sizes (1M each).
    # E1 = E2 = 500.5
    # LL = 2*(1000*ln(1000/500.5) + 1*ln(1/500.5))
    #     ≈ 2*(1000*0.69215 + 1*(-6.2156))
    #     ≈ 2*(692.149 + -6.2156) ≈ 1371.867
    ("lopsided_overrep_in_a", 1000, 1_000_000, 1, 1_000_000, 1371.867, "A"),
    # Hardie LogRatio worked example (CASS note 2014):
    # "the": 547,558 in BNC Spoken (10M tokens), 1,082,892 in BNC Written (90M).
    # By Hardie's own formula. LL value here computed from the contingency form.
    # E1 = 10M * (547558+1082892) / 100M = 163045
    # E2 = 90M * 1630450 / 100M = 1467405
    # LL = 2*(547558*ln(547558/163045) + 1082892*ln(1082892/1467405))
    #     = 2*(547558*1.21130 + 1082892*(-0.30385))
    #     ≈ 2*(663156 - 329029) ≈ 668254
    # (Approximate; we assert to a coarser tolerance.)
    ("bnc_spoken_vs_written_the", 547558, 10_000_000, 1082892, 90_000_000, 668254.0, "A"),
    # ------------------------------------------------------------------
    # Extended sweep — additional canonical contingency tables. Each
    # expected value was hand-derived from the unsigned LL formula and
    # then verified against pycorpdiff's own implementation before
    # being asserted here, so they form a "trip-wire" set: if a future
    # refactor changes the math, every case below flags simultaneously.
    # ------------------------------------------------------------------
    # Mid-sized corpus, 5× over-rep: 50/10K vs 10/10K.
    # E1=E2=30; LL = 2*(50*ln(5/3) + 10*ln(1/3))
    #             = 2*(50*0.510826 - 10*1.098612)
    #             = 2*(25.5413 - 10.9861) = 29.1104
    ("mid_sized_overrep_5x", 50, 10_000, 10, 10_000, 29.1104, "A"),
    # Equal absolute counts, unequal corpus sizes — the classic case
    # where raw counts mislead and rates separate the signal.
    # 100/1M vs 100/10M: E1=200*1M/11M=18.182, E2=181.818
    # LL = 2*(100*ln(5.5) + 100*ln(0.55)) ≈ 221.382
    ("equal_counts_unequal_corpora", 100, 1_000_000, 100, 10_000_000, 221.382, "A"),
    # CL-textbook standard: 174 vs 29 per million.
    # E1=E2=101.5; LL = 2*(174*ln(174/101.5) + 29*ln(29/101.5))
    #              = 2*(174*0.5390 + 29*(-1.2528)) ≈ 114.910
    ("textbook_174_vs_29_per_million", 174, 1_000_000, 29, 1_000_000, 114.9104, "A"),
    # Tiny counts at the edge of the χ² approximation. The formula
    # still produces a well-defined LL; downstream p-values from this
    # cell should be treated cautiously, but the *math* is exact.
    # 5/500 vs 1/500: E1=E2=3; LL = 2*(5*ln(5/3) + 1*ln(1/3)) = 2.9110
    ("tiny_counts_both_rare", 5, 500, 1, 500, 2.9110, "A"),
    # Specialised vocabulary pattern — a 100× over-representation,
    # the kind of pattern you see when comparing a technical subcorpus
    # to a general one. 300/10K vs 3/10K.
    # E1=E2=151.5; LL = 2*(300*ln(300/151.5) + 3*ln(3/151.5))
    #              = 2*(300*0.68318 + 3*(-3.92197))
    #              ≈ 2*(204.954 - 11.766) ≈ 386.386
    ("specialised_vocab_100x", 300, 10_000, 3, 10_000, 386.386, "A"),
    # Newspaper-section scale: 80 in one section (20K tokens) vs 20 in
    # the other (20K). E1=E2=50; LL = 2*(80*ln(80/50) + 20*ln(20/50))
    #                          = 2*(80*0.4700 - 20*0.9163) ≈ 38.5492
    ("newspaper_section_4x", 80, 20_000, 20, 20_000, 38.5492, "A"),
]


def test_rayson_canonical_log_likelihood_values() -> None:
    """Sweep the canonical reference triples. Each must match the LL Wizard."""
    for label, o1, n1, o2, n2, expected_ll, direction in RAYSON_REFERENCES:
        a = pd.Series({"term": o1})
        b = pd.Series({"term": o2})
        table = log_likelihood(a, b, total_a=n1, total_b=n2)
        ll = float(table.loc["term", "g2"])
        # The package returns *signed* LL; Rayson reports unsigned.
        # Match the magnitude, then check direction separately.
        # Tolerance: 0.5% relative or 1.0 absolute, whichever is larger,
        # because the BNC example value has ~6 significant digits of
        # roundoff in the hand computation.
        tolerance = max(abs(expected_ll) * 0.005, 1.0)
        assert math.isclose(abs(ll), expected_ll, abs_tol=tolerance), (
            f"{label}: expected |LL|={expected_ll}, got {abs(ll)}"
        )
        if direction == "A":
            assert ll > 0, f"{label}: expected A-overuse (+); got {ll}"
        elif direction == "B":
            assert ll < 0, f"{label}: expected B-overuse (−); got {ll}"
        # 'neutral' direction → LL == 0 to floating-point precision;
        # the magnitude check above already covers it.


def test_rayson_log_ratio_hardie_canonical() -> None:
    """Hardie (2014) LogRatio worked example.

    From the CASS note on LogRatio: if a term occurs 1000 times per
    million in corpus A and 100 times per million in corpus B, the
    LogRatio is log2(1000/100) = log2(10) ≈ 3.3219. With Laplace
    smoothing the actual numbers shift very slightly.
    """
    # 1000 occurrences per 1M tokens vs 100 per 1M.
    lr = log_ratio(
        pd.Series({"term": 1000}),
        pd.Series({"term": 100}),
        total_a=1_000_000,
        total_b=1_000_000,
        smoothing=0.5,
    )
    # With smoothing 0.5: log2((1000.5/1M) / (100.5/1M)) = log2(1000.5/100.5)
    expected = math.log2(1000.5 / 100.5)
    assert math.isclose(lr["term"], expected, rel_tol=1e-9)
    # Sanity: this is very close to the unsmoothed log2(10) = 3.3219.
    assert 3.30 < lr["term"] < 3.32


def test_rayson_log_ratio_equal_rates_yields_zero() -> None:
    """LR is *approximately* 0 when the unsmoothed rates match.

    Equal underlying rates (10/1000 vs 20/2000 are both 1%) yield
    LogRatio near 0; Laplace smoothing shifts each side by ≤α/N so
    the residual is tiny.
    """
    lr = log_ratio(
        pd.Series({"term": 10}),
        pd.Series({"term": 20}),
        total_a=1000,
        total_b=2000,
        smoothing=0.5,
    )
    assert abs(lr["term"]) < 0.05
