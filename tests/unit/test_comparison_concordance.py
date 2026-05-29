"""Tests for ``Comparison.concordance(target)``."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def two_outlet_corpus() -> pcd.Corpus:
    rows = [
        {"text": "the migrant worker arrived and the family settled", "outlet": "A"},
        {"text": "the migrant family stayed", "outlet": "A"},
        {"text": "the migrant criminal threat grew", "outlet": "B"},
        {"text": "the migrant invasion grew worse", "outlet": "B"},
    ]
    return pcd.from_dataframe(pd.DataFrame(rows), text_col="text", meta_cols=("outlet",))


def test_comparison_concordance_returns_concordance_result(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=2, window=2)
    assert isinstance(result, pcd.ConcordanceResult)
    assert (result.table["keyword"] == "migrant").all()


def test_comparison_concordance_labels_each_side(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=5)
    assert set(result.table["corpus"]) == {"outlet='A'", "outlet='B'"}


def test_comparison_concordance_respects_n_per_side(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=1)
    by_corpus = result.table.groupby("corpus").size()
    assert by_corpus.loc["outlet='A'"] == 1
    assert by_corpus.loc["outlet='B'"] == 1


def test_comparison_concordance_missing_target_returns_empty(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("unicorn")
    assert len(result.table) == 0


def test_comparison_concordance_table_has_left_keyword_right(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """Every KWIC row carries the surrounding context plus the keyword."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=5, window=3)
    for col in ("left", "keyword", "right", "corpus"):
        assert col in result.table.columns


def test_comparison_concordance_window_controls_context_size(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """A wider window must produce at least as much left+right context
    per row as a narrower one — counting tokens, not characters."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    narrow = pcd.compare(a, b).concordance("migrant", n=5, window=1)
    wide = pcd.compare(a, b).concordance("migrant", n=5, window=4)

    def avg_context_tokens(tbl: pd.DataFrame) -> float:
        if tbl.empty:
            return 0.0
        token_counts = (
            tbl["left"].str.split().str.len().fillna(0)
            + tbl["right"].str.split().str.len().fillna(0)
        )
        return float(token_counts.mean())

    assert avg_context_tokens(wide.table) >= avg_context_tokens(narrow.table)


def test_comparison_concordance_records_window(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """ConcordanceResult.window faithfully echoes the window= argument."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", window=7)
    assert result.window == 7


def test_comparison_concordance_target_is_recorded(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """ConcordanceResult.target carries the searched term."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant")
    assert result.target == "migrant"


def test_comparison_concordance_to_df_returns_copy(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """to_df() should be an independent copy — mutating it must not
    affect the underlying Result."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=2)
    df = result.to_df()
    df.loc[df.index[0], "keyword"] = "MUTATED"
    assert (result.table["keyword"] == "migrant").all()


def test_comparison_concordance_html_and_json_exports(
    two_outlet_corpus: pcd.Corpus,
    tmp_path,
) -> None:
    """to_html() / to_json() round-trip both inline and to disk."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=2)

    html = result.to_html()
    assert "<table" in html
    assert "migrant" in html

    json_str = result.to_json()
    assert "migrant" in json_str

    # Disk paths
    html_path = tmp_path / "kwic.html"
    json_path = tmp_path / "kwic.json"
    result.to_html(html_path)
    result.to_json(json_path)
    assert html_path.exists() and html_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0


def test_comparison_concordance_summary_string(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """summary() reports the target and the line count."""
    a = two_outlet_corpus.slice(outlet="A")
    b = two_outlet_corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=3)
    s = result.summary()
    assert "ConcordanceResult" in s
    assert "migrant" in s
    assert f"lines={len(result.table):,}" in s


def test_comparison_concordance_target_absent_on_one_side() -> None:
    """A target appearing only on one side returns lines for that side
    plus an empty contribution from the other (zero-row, not error)."""
    df = pd.DataFrame(
        [
            {"text": "the migrant arrived peacefully", "outlet": "A"},
            {"text": "the migrant settled happily", "outlet": "A"},
            {"text": "unrelated content here", "outlet": "B"},
        ]
    )
    corpus = pcd.from_dataframe(df, text_col="text", meta_cols=("outlet",))
    a = corpus.slice(outlet="A")
    b = corpus.slice(outlet="B")
    result = pcd.compare(a, b).concordance("migrant", n=5)
    by_corpus = result.table.groupby("corpus").size()
    assert by_corpus.get("outlet='A'", 0) == 2
    assert by_corpus.get("outlet='B'", 0) == 0


def test_comparison_concordance_uses_corpus_tokenizer(
    two_outlet_corpus: pcd.Corpus,
) -> None:
    """KWIC respects the configured tokenizer — case-insensitive matches
    via the default RegexTokenizer's lowercase=True default."""
    df = pd.DataFrame(
        [
            {"text": "The Migrant arrived", "outlet": "A"},
            {"text": "MIGRANT crisis grows", "outlet": "B"},
        ]
    )
    corpus = pcd.from_dataframe(df, text_col="text", meta_cols=("outlet",))
    result = pcd.compare(
        corpus.slice(outlet="A"), corpus.slice(outlet="B")
    ).concordance("migrant", n=5)
    # Both cased forms ('Migrant', 'MIGRANT') are matched via default tokenizer.
    assert len(result.table) >= 2
