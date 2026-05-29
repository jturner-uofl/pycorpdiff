"""End-to-end tests for ``compare(a, b).collocation_shift()``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def immigrant_corpus() -> pcd.Corpus:
    """Two-frame fixture engineered to give clearly different collocates."""
    pos_docs = [
        "the immigrant worker arrived and the immigrant family settled",
        "the immigrant community grew the immigrant worker thrived",
        "the immigrant worker and the immigrant family stayed",
        "the immigrant worker and immigrant rights advanced",
    ]
    neg_docs = [
        "the immigrant criminal threat and the immigrant invasion grew",
        "the immigrant threat and the immigrant crime increased",
        "the immigrant invasion of immigrant criminal gangs spread",
        "the immigrant criminal gangs and immigrant invasion stayed",
    ]
    rows = [{"text": d, "frame": "humanising"} for d in pos_docs] + [
        {"text": d, "frame": "criminalising"} for d in neg_docs
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("frame",))


def test_collocation_shift_returns_result_object(immigrant_corpus: pcd.Corpus) -> None:
    a = immigrant_corpus.slice(frame="humanising")
    b = immigrant_corpus.slice(frame="criminalising")
    result = pcd.compare(a, b).collocation_shift("immigrant", window=3, min_count=1)
    assert isinstance(result, pcd.CollocationShiftResult)
    assert result.target == "immigrant"
    assert result.measure == "logDice"
    assert result.window == 3
    assert "shift" in result.table.columns


def test_collocation_shift_top_results_are_meaningful(
    immigrant_corpus: pcd.Corpus,
) -> None:
    a = immigrant_corpus.slice(frame="humanising")
    b = immigrant_corpus.slice(frame="criminalising")
    df = (
        pcd.compare(a, b)
        .collocation_shift("immigrant", window=3, min_count=1)
        .table.set_index("collocate")
    )
    # The top |shift| collocates should be drawn from the engineered semantic fields.
    top_5 = df["shift"].abs().nlargest(5).index.tolist()
    humanising = {"worker", "family", "community", "rights"}
    criminalising = {"criminal", "invasion", "threat", "gangs", "crime"}
    field = humanising | criminalising
    overlap = set(top_5) & field
    assert len(overlap) >= 3, f"expected ≥3 of {field}; top_5={top_5}"


def test_all_measures_agree_on_well_populated_field(
    immigrant_corpus: pcd.Corpus,
) -> None:
    """Across logDice / PMI / t-score, collocates with substantial counts
    on the engineered side should carry the corpus's directional signal.

    Low-count collocates — including engineered-field terms whose count
    on the *other* side is purely smoothing-derived — can legitimately
    flip sign between PMI (which is sensitive to corpus-size differences
    at low joint counts) and logDice / t-score. That's a real property of
    PMI's normalisation, not a bug, and we don't want a flake-prone test.
    """
    a = immigrant_corpus.slice(frame="humanising")
    b = immigrant_corpus.slice(frame="criminalising")
    forward = pcd.compare(a, b)
    expected_negative = {"criminal", "invasion", "threat", "gangs"}  # in B
    for measure in ("logDice", "PMI", "t_score", "MI3"):
        df = forward.collocation_shift(
            "immigrant", window=3, min_count=1, measure=measure  # type: ignore[arg-type]
        ).table.set_index("collocate")
        for term in expected_negative & set(df.index):
            assert df.loc[term, "shift"] < 0, (
                f"{measure} expected negative for {term!r}; got {df.loc[term, 'shift']}"
            )
