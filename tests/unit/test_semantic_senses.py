"""Tests for :func:`pycorpdiff.induce_senses` and the audit operations.

Uses :class:`HashEmbedder` plus a hand-built three-sense fixture so the
embeddings carry a known, separable structure: within-sense texts share
a sense-specific keyword, so their hashed vectors are identical and the
clusters are recoverable. This verifies the orchestration (clustering,
agreement, leakage, share-over-time) without paying for a real model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd

pytest.importorskip("sklearn")


def _three_sense_fixture(n_per: int = 20):
    """Three well-separated senses via shared per-sense tokens.

    HashEmbedder maps identical strings to identical vectors, so we make
    each sense's texts *almost* identical (a shared stem) and embed a
    distinctive token per item to keep vectors stable but separable.
    """
    senses = {
        "cannabidiol": "cannabidiol cannabinoid receptor epilepsy seizure",
        "biliary": "bile duct biliary cholangiography common obstruction",
        "neurology": "corticobasal degeneration parkinsonism cortical atrophy",
    }
    rows = []
    for ref, stem in senses.items():
        for i in range(n_per):
            rows.append({"text": stem, "ref": ref, "date": f"20{10 + i % 10}-01-01"})
    frame = pd.DataFrame(rows)
    emb = pcd.HashEmbedder(dim=24)
    X = emb.encode(frame["text"].tolist())
    return frame, X


def test_induce_senses_recovers_known_k() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, text_col="text", random_state=42)
    assert res.k == 3
    # Perfect structure: each induced cluster maps 1:1 to a ref sense.
    agree = res.agreement_with(frame["ref"])
    assert agree.ari == pytest.approx(1.0)
    assert agree.v_measure == pytest.approx(1.0)


def test_silhouette_auto_selects_k() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=None, k_range=(2, 6), random_state=42)
    assert res.k == 3
    assert res.silhouette is not None
    assert res.k_scores is not None
    assert set(res.k_scores["k"]) == {2, 3, 4, 5, 6}


def test_deterministic_labels_across_runs() -> None:
    frame, X = _three_sense_fixture()
    a = pcd.induce_senses(frame, X, k=3, random_state=42)
    b = pcd.induce_senses(frame, X, k=3, random_state=42)
    assert np.array_equal(a.labels, b.labels)


def test_agreement_identical_is_one_shuffled_is_low() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, random_state=42)
    identical = res.agreement_with(res.labels)
    assert identical.ari == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(frame["ref"].to_numpy())
    low = res.agreement_with(shuffled)
    assert low.ari < 0.5


def test_agreement_contingency_shape() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, random_state=42)
    agree = res.agreement_with(frame["ref"])
    assert agree.contingency.shape == (3, 3)
    assert "ARI=" in agree.summary()


def test_leakage_audit_flags_planted_record() -> None:
    frame, X = _three_sense_fixture()
    # Plant a leak: relabel one clearly-biliary row as cannabidiol.
    ref = frame["ref"].to_numpy().copy()
    biliary_idx = np.flatnonzero(ref == "biliary")[0]
    ref[biliary_idx] = "cannabidiol"
    res = pcd.induce_senses(frame, X, k=3, random_state=42)
    suspects = res.leakage_audit(ref, k=10)
    assert len(suspects) >= 1
    # The planted record should be the/a top suspect: ref says cannabidiol
    # but geometry pulls it to biliary.
    top = suspects.iloc[0]
    assert top["reference_sense"] == "cannabidiol"
    assert top["nearest_other_sense"] == "biliary"
    assert top["margin"] > 0


def test_leakage_audit_clean_labels_no_suspects() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, random_state=42)
    suspects = res.leakage_audit(frame["ref"], k=10)
    assert len(suspects) == 0


def test_share_over_time_sums_to_one_per_period() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, text_col="text", time_col="date")
    sot = res.share_over_time(freq="Y")
    period_sums = sot.groupby("period")["share"].sum()
    assert np.allclose(period_sums.to_numpy(), 1.0)


def test_token_mode_aggregates_to_docs() -> None:
    frame, X = _three_sense_fixture(n_per=10)
    # Treat each row as a token-occurrence; assign two occurrences per doc.
    doc_ids = np.repeat(np.arange(len(frame) // 2), 2)[: len(frame)]
    res = pcd.induce_senses(
        frame, X, k=3, unit="token", item_to_doc=doc_ids, random_state=42
    )
    assert res.unit == "token"
    assert set(res.assignments["doc_id"]) == set(doc_ids)


def test_result_contract_methods() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, text_col="text")
    assert isinstance(res.to_df(), pd.DataFrame)
    assert "<table" in res.to_html().lower()
    assert res.to_json().startswith("[") or res.to_json().startswith("{")
    assert "senses induced" in res.summary()
    assert set(res.clusters.columns) == {"sense", "size", "share", "top_terms"}


def test_embedding_meta_echoed() -> None:
    frame, X = _three_sense_fixture()
    meta = {"model": "all-MiniLM-L6-v2", "revision": "abc123", "hash": "deadbeef"}
    res = pcd.induce_senses(frame, X, k=3, embedding_meta=meta)
    assert res.embedding_meta == meta


# ---- validation / edge cases ------------------------------------------------
def test_mismatched_lengths_raise() -> None:
    frame, X = _three_sense_fixture()
    with pytest.raises(ValueError, match="align by position"):
        pcd.induce_senses(frame, X[:-1], k=3)


def test_k_too_large_raises() -> None:
    frame, X = _three_sense_fixture(n_per=2)
    with pytest.raises(ValueError, match="must be < n_items"):
        pcd.induce_senses(frame, X, k=999)


def test_non_finite_embeddings_raise() -> None:
    frame, X = _three_sense_fixture()
    X2 = X.copy()
    X2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        pcd.induce_senses(frame, X2, k=3)


def test_token_mode_requires_item_to_doc() -> None:
    frame, X = _three_sense_fixture()
    with pytest.raises(ValueError, match="requires item_to_doc"):
        pcd.induce_senses(frame, X, k=3, unit="token")


def test_bad_method_raises() -> None:
    frame, X = _three_sense_fixture()
    with pytest.raises(ValueError, match="kmeans.*agglomerative"):
        pcd.induce_senses(frame, X, k=3, method="dbscan")


def test_agglomerative_method_runs() -> None:
    frame, X = _three_sense_fixture()
    res = pcd.induce_senses(frame, X, k=3, method="agglomerative")
    assert res.k == 3
    agree = res.agreement_with(frame["ref"])
    assert agree.ari == pytest.approx(1.0)


def test_plain_sequence_items() -> None:
    _, X = _three_sense_fixture()
    texts = ["cannabidiol"] * 20 + ["bile"] * 20 + ["cortical"] * 20
    res = pcd.induce_senses(texts, X, k=3)
    assert len(res.assignments) == 60
