"""Tests for :func:`pycorpdiff.knn_density_drift` --- the sense-free,
streaming-capable sibling of :func:`sense_drift`.

Reuses the same synthetic temporal structures (emergence, stable) so the two
detectors can be compared head to head, including a **cross-detector agreement**
check: independent formulations (k-means margin density vs k-NN density) should
flag the same emergence --- the robustness claim, as a test.
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
REF = list(range(2000, 2010))


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


def test_emergence_detected():
    df, X = _emergence()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    assert res.table["drift"].any()
    onset = res.table.loc[res.table["drift"], "period"].min()
    assert 2012 <= onset <= 2015


def test_stable_no_drift():
    df, X = _stable()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    assert not res.table["drift"].any()
    assert "No k-NN-density drift" in res.summary()


def test_result_contract():
    df, X = _emergence()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    assert set(res.table.columns) == {"period", "n", "novelty_density", "drift"}
    assert "<table" in res.to_html().lower()
    assert res.to_json().startswith("[")
    assert "drift" in res.summary().lower()
    assert len(res.flagged_records()) > 0


def test_deterministic():
    df, X = _emergence()
    a = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    b = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    pd.testing.assert_frame_equal(a.table, b.table)


def test_cumulative_mode_responds_to_emergence_onset():
    # Streaming novelty is TRANSIENT: a sense spikes when it first appears with
    # no precedent, then becomes 'known' as it accrues its own history. So the
    # cumulative signal peaks at onset and decays -- the opposite of the
    # persistent reference-mode signal. (This is why min_run>=2 won't flag a
    # one-year emergence in cumulative mode; the spike IS the signal.)
    df, X = _emergence()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10, mode="cumulative")
    assert res.mode == "cumulative"
    nd = res.table.set_index("period")["novelty_density"]
    base = float(nd.loc[REF].mean())
    onset = float(nd.loc[[2012, 2013]].max())
    late = float(nd.loc[[2018, 2019, 2020]].mean())
    assert onset > base   # responds at first-appearance
    assert onset > late   # ...and decays as the sense becomes known


def test_cumulative_permutation_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="permutation null is only defined"):
        pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10,
                              mode="cumulative", n_permutations=10)


def test_exemplars_sorted_by_novelty():
    df, X = _emergence()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    ex = res.exemplars(top=5)
    assert 0 < len(ex) <= 5
    assert (ex["novelty"].diff().dropna() <= 1e-9).all()  # non-increasing
    assert ex["text"].str.contains("emergent|epilepsy|seizure").any()


def test_permutation_pvalue_significant_for_emergence():
    df, X = _emergence()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10, n_permutations=20)
    assert res.p_value is not None
    assert 0.0 < res.p_value <= 1.0
    assert res.p_value < 0.2
    assert "Permutation p=" in res.summary()


def test_permutation_stable_high_pvalue():
    df, X = _stable()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10, n_permutations=20)
    assert res.p_value is not None
    assert res.p_value > 0.1


def test_sustained_run_filters_isolated_spike():
    df, X = _stable()
    rng = np.random.default_rng(9)
    spike_rows = pd.DataFrame({"year": [2015] * 40, "text": ["blip"] * 40})
    spike_emb = rng.standard_normal((40, D)) * SCALE * 2.5
    df2 = pd.concat([df, spike_rows], ignore_index=True)
    X2 = np.vstack([X, spike_emb])
    res = pcd.knn_density_drift(df2, X2, time_col="year", reference=REF, k=10, min_run=2)
    assert not res.table.loc[res.table["period"] == 2015, "drift"].any()


def test_plot_returns_chart():
    pytest.importorskip("altair")
    df, X = _emergence()
    res = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    assert res.plot(width=300, height=150).__class__.__name__ in {"LayerChart", "Chart"}


def test_agrees_with_sense_drift_on_emergence():
    """Two independent formulations should flag the same emergence; their
    per-period novelty densities correlate (the robustness result)."""
    df, X = _emergence()
    knn = pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10)
    sd = pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)
    assert knn.table["drift"].any() and sd.table["drift"].any()
    knn_onset = int(knn.table.loc[knn.table["drift"], "period"].min())
    sd_onset = int(sd.table.loc[sd.table["drift"], "period"].min())
    assert abs(knn_onset - sd_onset) <= 2
    mon = ~knn.table["period"].isin(REF)
    r = np.corrcoef(knn.table.loc[mon, "novelty_density"],
                    sd.table.loc[mon, "margin_density"])[0, 1]
    assert r > 0.5


# ---- validation / edge cases -------------------------------------------------
def test_reference_too_small_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="reference period has only"):
        pcd.knn_density_drift(df, X, time_col="year", reference=[1999], k=10)


def test_mismatched_lengths_raise():
    df, X = _stable()
    with pytest.raises(ValueError, match="rows but items"):
        pcd.knn_density_drift(df, X[:-1], time_col="year", reference=REF, k=10)


def test_nan_embeddings_raise():
    df, X = _stable()
    X2 = X.copy(); X2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        pcd.knn_density_drift(df, X2, time_col="year", reference=REF, k=10)


def test_bad_time_col_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="time_col"):
        pcd.knn_density_drift(df, X, time_col="nope", reference=REF, k=10)


def test_bad_mode_raises():
    df, X = _stable()
    with pytest.raises(ValueError, match="mode must be"):
        pcd.knn_density_drift(df, X, time_col="year", reference=REF, k=10, mode="bogus")


def test_range_reference_equivalent_to_list():
    """Regression: a bare ``range`` reference must work (the README uses it) and
    match the equivalent list --- not collapse to a one-element ``[range(...)]``
    that matches zero rows and raises ``reference period has only 0 records``.
    Both detectors shared the normalization, so cover sense_drift too."""
    df, X = _emergence()
    list_ref = list(range(2000, 2010))

    k_rng = pcd.knn_density_drift(df, X, time_col="year", reference=range(2000, 2010), k=5)
    k_lst = pcd.knn_density_drift(df, X, time_col="year", reference=list_ref, k=5)
    assert k_rng.reference == list_ref
    pd.testing.assert_frame_equal(k_rng.table, k_lst.table)

    s_rng = pcd.sense_drift(df, X, time_col="year", reference=range(2000, 2010), k=3)
    s_lst = pcd.sense_drift(df, X, time_col="year", reference=list_ref, k=3)
    pd.testing.assert_frame_equal(s_rng.table, s_lst.table)
