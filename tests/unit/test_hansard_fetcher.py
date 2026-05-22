"""Tests for ``pycorpdiff.fetch_hansard``.

All HTTP calls are mocked via the ``_fetch`` hook on
:func:`fetch_hansard`; no network access is required to run these
tests. A real-API smoke test could be added later as a slow-tier job
once we're comfortable hitting parliament.uk in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pycorpdiff as pcd
from pycorpdiff.datasets.hansard import _default_parse_search_response


def _sample_payload() -> dict[str, Any]:
    """A realistic-looking search response payload."""
    return {
        "Results": [
            {
                "ContributionText": (
                    "I rise to address the House on the matter of immigration "
                    "and the migrant worker who arrived seeking refuge."
                ),
                "AttributedTo": "Yvette Cooper",
                "MemberParty": "Labour",
                "SittingDate": "2020-03-15T00:00:00",
                "DebateSection": "Immigration Policy",
                "ContributionExtId": "abc-001",
            },
            {
                "ContributionText": (
                    "The migrant criminal threat must be confronted with "
                    "stronger borders and tougher enforcement measures."
                ),
                "AttributedTo": "Priti Patel",
                "MemberParty": "Conservative",
                "SittingDate": "2020-06-20T00:00:00",
                "DebateSection": "Border Security",
                "ContributionExtId": "abc-002",
            },
        ]
    }


def test_fetch_hansard_builds_correct_search_url() -> None:
    captured_urls: list[str] = []

    def mock_fetch(url: str) -> dict[str, Any]:
        captured_urls.append(url)
        return _sample_payload()

    pcd.fetch_hansard(
        "immigration",
        start_date="2020-01-01",
        end_date="2020-12-31",
        max_results=50,
        _fetch=mock_fetch,
    )
    assert len(captured_urls) == 1
    url = captured_urls[0]
    assert "hansard-api.parliament.uk" in url
    assert "queryParameters.searchTerm=immigration" in url
    assert "queryParameters.startDate=2020-01-01" in url
    assert "queryParameters.endDate=2020-12-31" in url
    assert "queryParameters.take=50" in url


def test_fetch_hansard_returns_corpus_with_expected_columns() -> None:
    corpus = pcd.fetch_hansard(
        "immigration",
        start_date="2020-01-01",
        end_date="2020-12-31",
        _fetch=lambda url: _sample_payload(),
    )
    assert isinstance(corpus, pcd.Corpus)
    assert len(corpus) == 2
    expected_cols = {"text", "speaker", "party", "date", "debate_title", "hansard_id"}
    assert expected_cols <= set(corpus.docs.columns)


def test_fetch_hansard_normalises_dates_to_iso() -> None:
    corpus = pcd.fetch_hansard(
        "immigration", "2020-01-01", "2020-12-31",
        _fetch=lambda url: _sample_payload(),
    )
    # SittingDate "2020-03-15T00:00:00" → "2020-03-15"
    assert corpus.docs["date"].tolist() == ["2020-03-15", "2020-06-20"]


def test_fetch_hansard_drives_analytical_pipeline() -> None:
    """The fetched corpus should work end-to-end through compare()."""
    corpus = pcd.fetch_hansard(
        "immigration", "2020-01-01", "2020-12-31",
        _fetch=lambda url: _sample_payload(),
    )
    lab = corpus.slice(party="Labour")
    con = corpus.slice(party="Conservative")
    result = pcd.compare(lab, con).keyness(min_count=1)
    assert isinstance(result, pcd.KeynessResult)


def test_fetch_hansard_empty_response_yields_empty_corpus() -> None:
    corpus = pcd.fetch_hansard(
        "unicorn", "2020-01-01", "2020-12-31",
        _fetch=lambda url: {"Results": []},
    )
    assert len(corpus) == 0
    # Schema should still be present so downstream code doesn't break.
    assert "text" in corpus.docs.columns


def test_fetch_hansard_writes_and_reads_cache(tmp_path: Path) -> None:
    payload = _sample_payload()
    call_count = {"n": 0}

    def counting_fetch(url: str) -> dict[str, Any]:
        call_count["n"] += 1
        return payload

    # First call: cache miss → fetches.
    corpus1 = pcd.fetch_hansard(
        "immigration", "2020-01-01", "2020-12-31",
        cache_dir=tmp_path, _fetch=counting_fetch,
    )
    assert call_count["n"] == 1
    assert len(corpus1) == 2

    # Second call with identical args: cache hit → no fetch.
    corpus2 = pcd.fetch_hansard(
        "immigration", "2020-01-01", "2020-12-31",
        cache_dir=tmp_path, _fetch=counting_fetch,
    )
    assert call_count["n"] == 1  # still 1
    assert len(corpus2) == 2

    # Different search term → cache miss again.
    pcd.fetch_hansard(
        "brexit", "2020-01-01", "2020-12-31",
        cache_dir=tmp_path, _fetch=counting_fetch,
    )
    assert call_count["n"] == 2


def test_fetch_hansard_alternate_response_shapes() -> None:
    """The default parser tolerates the common JSON-schema variations."""
    # List at root.
    corpus = pcd.fetch_hansard(
        "x", "2020-01-01", "2020-01-31",
        _fetch=lambda url: [
            {
                "Text": "the migrant worker",
                "AttributedTo": "MP A",
                "Party": "Labour",
                "DebateDate": "2020-01-15",
                "Title": "Debate",
                "Id": "001",
            }
        ],
    )
    assert len(corpus) == 1
    # "SearchResults" wrapping
    corpus2 = pcd.fetch_hansard(
        "y", "2020-01-01", "2020-01-31",
        _fetch=lambda url: {
            "SearchResults": [
                {"Snippet": "the foo bar", "MemberName": "MP B"}
            ]
        },
    )
    assert len(corpus2) == 1


def test_fetch_hansard_custom_response_parser() -> None:
    """Users can supply their own parser for a non-standard schema."""

    def my_parser(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "text": item["my_text"],
                "speaker": "anon",
                "party": "",
                "date": "",
                "debate_title": "",
                "hansard_id": "",
            }
            for item in payload.get("custom_field", [])
        ]

    corpus = pcd.fetch_hansard(
        "x", "2020-01-01", "2020-01-31",
        _fetch=lambda url: {"custom_field": [{"my_text": "alpha beta"}]},
        response_parser=my_parser,
    )
    assert len(corpus) == 1
    assert corpus.docs["text"].iloc[0] == "alpha beta"


def test_default_parser_skips_rows_with_no_text() -> None:
    """Rows lacking any text variant should drop silently."""
    rows = _default_parse_search_response(
        {
            "Results": [
                {"AttributedTo": "MP X"},  # no text fields at all
                {"ContributionText": "the actual speech", "AttributedTo": "MP Y"},
            ]
        }
    )
    assert len(rows) == 1
    assert rows[0]["speaker"] == "MP Y"


def test_fetch_hansard_uses_custom_base_url() -> None:
    captured: list[str] = []
    pcd.fetch_hansard(
        "x", "2020-01-01", "2020-01-31",
        base_url="https://example.test/mirror",
        _fetch=lambda url: (captured.append(url), {"Results": []})[1],
    )
    assert captured[0].startswith("https://example.test/mirror")
