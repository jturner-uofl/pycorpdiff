"""Tests for the LLM annotation layer (:mod:`pycorpdiff.annotate`).

The headline guarantee is the *honest division of labour*: an
:class:`~pycorpdiff.Annotator` may name and gloss senses, but nothing it returns
can ever reach a numeric field, mutate the source result, or assert veracity.
``test_annotator_output_never_enters_numeric_fields`` is the structural proof.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pycorpdiff as pcd
from pycorpdiff import Annotator, EchoAnnotator, OllamaAnnotator, SenseNamingResult

pytest.importorskip("sklearn")

D = 20
REF = list(range(2000, 2010))


def _emergence(seed=1):
    """3 stable senses every year; a 4th coherent sense appears in 2012."""
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((4, D)) * 5.0
    rows, emb = [], []
    for y in range(2000, 2021):
        for _ in range(60):
            s = rng.integers(0, 3)
            emb.append(C[s] + rng.standard_normal(D) * 0.35)
            rows.append({"year": y, "text": f"known sense {s} alpha beta gamma"})
        if y >= 2012:
            for _ in range((y - 2011) * 5):
                emb.append(C[3] + rng.standard_normal(D) * 0.35)
                rows.append({"year": y, "text": "novel coherent emergent epilepsy seizure clobazam"})
    return pd.DataFrame(rows), np.vstack(emb)


def _result():
    df, X = _emergence()
    return pcd.sense_drift(df, X, time_col="year", reference=REF, k=3)


class _Malicious:
    """Annotator that tries to inject numeric-looking text into the result."""

    model_id = "malicious"

    def __call__(self, prompt: str) -> str:
        return '{"label": "0.9999", "gloss": "424242"}'


class _Counting:
    model_id = "counter"

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, prompt: str) -> str:
        self.n += 1
        return '{"label": "x", "gloss": "y"}'


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
def test_protocol_conformance():
    assert isinstance(EchoAnnotator(), Annotator)
    assert isinstance(OllamaAnnotator(), Annotator)
    assert isinstance(_Malicious(), Annotator)
    assert isinstance(lambda p: "x", Annotator)  # any callable str->str qualifies


def test_echo_annotator_deterministic():
    a = EchoAnnotator()
    assert a("same prompt") == a("same prompt")
    assert a("prompt one") != a("prompt two")


# --------------------------------------------------------------------------- #
# name_senses behaviour
# --------------------------------------------------------------------------- #
def test_name_senses_labels_every_sense_plus_emergent():
    res = _result()
    named = res.name_senses(EchoAnnotator())
    assert isinstance(named, SenseNamingResult)
    # k reference senses + 1 emergent bin
    assert len(named.table) == res.k + 1
    assert set(named.table["kind"]) == {"reference", "emergent"}
    assert (named.table["label"].str.len() > 0).all()
    # distinct senses get distinct labels (prompts differ -> hashes differ)
    assert named.table["label"].nunique() == len(named.table)


def test_name_senses_is_grounded_in_cited_exemplars():
    res = _result()
    named = res.name_senses(EchoAnnotator(), n_examples=6)
    # every named sense was shown at least one cited exemplar
    assert (named.table["n_cited"] > 0).all()
    assert (named.table["n_cited"] <= 6).all()
    # provenance records a prompt hash per sense
    assert len(named.provenance["senses"]) == len(named.table)
    assert all("prompt_sha256" in s for s in named.provenance["senses"])


def test_include_novel_false_omits_emergent_bin():
    res = _result()
    named = res.name_senses(EchoAnnotator(), include_novel=False)
    assert len(named.table) == res.k
    assert set(named.table["kind"]) == {"reference"}


def test_caching_avoids_requerying_identical_prompts():
    res = _result()
    counter = _Counting()
    cache: dict[str, str] = {}
    first = res.name_senses(counter, cache=cache)
    n_after_first = counter.n
    assert n_after_first == len(first.table)  # one call per sense, all fresh
    assert first.provenance["cache_hits"] == 0
    # second call with the SAME cache -> no new model calls
    second = res.name_senses(counter, cache=cache)
    assert counter.n == n_after_first
    assert second.provenance["calls"] == 0
    assert second.provenance["cache_hits"] == len(second.table)


def test_result_contract():
    res = _result()
    named = res.name_senses(EchoAnnotator())
    assert set(named.table.columns) == {"sense", "kind", "label", "gloss",
                                        "terms", "n", "n_cited"}
    assert "<table" in named.to_html().lower()
    assert named.to_json().startswith("[")
    assert "named" in named.summary().lower()


# --------------------------------------------------------------------------- #
# THE INVARIANT: annotator output can never reach a numeric field
# --------------------------------------------------------------------------- #
def test_annotator_output_never_enters_numeric_fields():
    res = _result()
    table_before = res.table.copy()
    named = res.name_senses(_Malicious(), include_novel=True)

    # 1. the source result's numeric table is untouched (frozen; new object out)
    pd.testing.assert_frame_equal(res.table, table_before)

    # 2. the injected numeric-looking strings live ONLY in string columns
    assert str(named.table["label"].dtype) == "string"
    assert str(named.table["gloss"].dtype) == "string"
    assert (named.table["label"] == "0.9999").all()
    assert (named.table["gloss"] == "424242").all()

    # 3. every measured column stays integer; the injected value never appears
    for col in ("sense", "n", "n_cited"):
        assert pd.api.types.is_integer_dtype(named.table[col])
    numeric_cols = named.table.select_dtypes(include="number").columns
    for col in numeric_cols:
        assert not named.table[col].astype(str).isin({"0.9999", "424242"}).any()

    # 4. nothing the annotator emitted leaked into the drift result anywhere
    drift_blob = res.table.astype(str).to_numpy().ravel()
    assert "0.9999" not in set(drift_blob)
    assert "424242" not in set(drift_blob)

    # 5. provenance states the contract explicitly
    assert "never writes a number" in named.provenance["contract"]
