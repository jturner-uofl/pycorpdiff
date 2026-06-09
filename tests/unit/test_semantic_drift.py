"""Tests for :func:`pycorpdiff.sense_drift`.

Synthetic corpora with *known* temporal structure exercise each change
type the explanation layer must discriminate: a new coherent sense
appearing (emergence), diffuse novelty (broadening), a re-weighting of
known senses with no new material (frequency shift), and no change at
all (stable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd

pytest.importorskip("sklearn")

D = 20
SCALE = 5.0
NOISE = 0.35


def _centroids(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, D)) * SCALE


def _emergence(seed=1):
    """3 stable senses every year; a 4th coherent sense appears in 2012."""
    rng = np.random.default_rng(seed)
    C = _centroids(4)
    rows, emb = [], []
    for y in range(2000, 2021):
        for _ in range(60):
            s = rng.integers(0, 3)
            emb.append(C[s] + rng.standard_normal(D) * NOISE)
            rows.append({"year": y, "text": f"known sense {s} alpha"})
        if y >= 2012:
            for _ in range((y - 2011) * 5):
                emb.append(C[3] + rng.standard_normal(D) * NOISE)
                rows.append({"year": y, "text": "novel coherent emergent epilepsy seizure"})
    return pd.DataFrame(rows), np.vstack(emb)


def _broadening(seed=2):
    """3 stable senses; from 2012, diffuse novel points (each its own
    random far-off direction -> not a coherent cluster)."""
    rng = np.random.default_rng(seed)
    C = _centroids(3)
    rows, emb = [], []
    for y in range(2000, 2021):
        for _ in range(60):
            s = rng.integers(0, 3)
            emb.append(C[s] + rng.standard_normal(D) * NOISE)
            rows.append({"year": y, "text": f"known sense {s}"})
        if y >= 2012:
            for _ in range((y - 2011) * 5):
                v = rng.standard_normal(D)
                emb.append(v / np.linalg.norm(v) * SCALE * 2.5)  # scattered, far
                rows.append({"year": y, "text": f"diffuse novelty topic{rng.integers(0,99)}"})
    return pd.DataFrame(rows), np.vstack(emb)


def _frequency_shift(seed=3):
    """3 senses present throughout, all near known centroids (no novelty),
    but the mixing proportions shift from uniform (early) to mostly
    sense 2 (late)."""
    rng = np.random.default_rng(seed)
    C = _centroids(3)
    rows, emb = [], []
    for y in range(2000, 2021):
        frac_c = 0.33 if y < 2012 else 0.85  # sense 2 takes over late
        for _ in range(120):
            r = rng.random()
            s = 2 if r < frac_c else rng.integers(0, 2)
            emb.append(C[s] + rng.standard_normal(D) * NOISE)
            rows.append({"year": y, "text": f"known sense {s}"})
    return pd.DataFrame(rows), np.vstack(emb)


def _stable(seed=4):
    rng = np.random.default_rng(seed)
    C = _centroids(3)
    rows, emb = [], []
    for y in range(2000, 2021):
        for _ in range(80):
            s = rng.integers(0, 3)
            emb.append(C[s] + rng.standard_normal(D) * NOISE)
            rows.append({"year": y, "text": f"stable sense {s}"})
    return pd.DataFrame(rows), np.vstack(emb)


REF = list(range(2000, 2010))


def test_emergence_detected_and_classified():
    df, X = _emergence()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    assert res.table["drift"].any()
    assert res.change_type == "emergence"
    onset = res.table.loc[res.table["drift"], "period"].min()
    assert 2012 <= onset <= 2014  # planted at 2012, detected as it grows
    assert any(t in res.drift_terms for t in ("emergent", "epilepsy", "seizure", "coherent"))


def test_broadening_detected_and_classified():
    df, X = _broadening()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    assert res.table["drift"].any()
    assert res.change_type == "broadening"


def test_frequency_shift_detected_and_classified():
    df, X = _frequency_shift()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    # re-weighting raises JSD even though margin density stays low
    assert res.table["drift"].any()
    assert res.change_type == "frequency_shift"


def test_stable_corpus_no_drift():
    df, X = _stable()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    assert not res.table["drift"].any()
    assert res.change_type is None
    assert "No sense drift" in res.summary()


def test_sustained_run_filters_isolated_spike():
    # stable corpus + a single anomalous year of novelty -> not confirmed
    df, X = _stable()
    rng = np.random.default_rng(9)
    spike_rows = pd.DataFrame({"year": [2015] * 40, "text": ["blip"] * 40})
    spike_emb = rng.standard_normal((40, D)) * SCALE * 2.5
    df2 = pd.concat([df, spike_rows], ignore_index=True)
    X2 = np.vstack([X, spike_emb])
    res = pcd.sense_drift(df2, X2, time_col="year", reference=REF, k=3, min_run=2)
    # the single 2015 spike must not be confirmed as drift
    assert not res.table.loc[res.table["period"] == 2015, "drift"].any()


def test_deterministic():
    df, X = _emergence()
    a = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    b = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    pd.testing.assert_frame_equal(a.table, b.table)


def test_cosine_novelty_runs():
    df, X = _emergence()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3, novelty="cosine")
    assert res.table["drift"].any()


def test_result_contract():
    df, X = _emergence()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    assert set(res.table.columns) == {"period", "n", "margin_density", "jsd", "drift"}
    assert "<table" in res.to_html().lower()
    assert res.to_json().startswith("[")
    assert "drift" in res.summary().lower()
    assert len(res.flagged_records()) > 0


def test_plot_returns_chart():
    pytest.importorskip("altair")
    df, X = _emergence()
    res = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    assert res.plot(width=300, height=150).__class__.__name__ in {"LayerChart", "Chart"}


# ---- validation / edge cases ------------------------------------------------
def test_reference_too_small_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="reference period has only"):
        pcd.sense_drift(df, X, time_col="year", reference=[1999], k=3)  # absent year


def test_mismatched_lengths_raise():
    df, X = _stable()
    with pytest.raises(ValueError, match="rows but items"):
        pcd.sense_drift(df, X[:-1], time_col="year", reference=REF, k=3)


def test_nan_embeddings_raise():
    df, X = _stable()
    X2 = X.copy()
    X2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        pcd.sense_drift(df, X2, time_col="year", reference=REF, k=3)


def test_bad_time_col_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="time_col"):
        pcd.sense_drift(df, X, time_col="nope", reference=REF, k=3)


def test_bad_novelty_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="mahalanobis.*cosine"):
        pcd.sense_drift(df, X, time_col="year", reference=REF, k=3, novelty="bogus")
