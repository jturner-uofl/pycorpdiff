"""UK Hansard loader: bundled synthetic sample + live fetcher.

Two functions live here:

- :func:`load_hansard_sample` — return the bundled 193-speech synthetic
  sample. Deterministic; ships with the package; no network. Use this
  for tutorials, tests, and offline demos.
- :func:`fetch_hansard` — query the live UK Parliament Hansard search
  API, optionally caching to a local parquet, and return the matched
  speeches as a :class:`Corpus`. Use this for actual research.

The live API
------------

``fetch_hansard`` hits the public Hansard search endpoint at
``https://hansard-api.parliament.uk/``. The endpoint requires no auth
and serves UK parliamentary speeches under the Open Government
Licence (essentially public domain, attribution requested).

The API surface changes occasionally; if a field name changes upstream
the function exposes a ``response_parser`` hook so users can adapt
without monkey-patching. The defaults match the schema as of
early 2026.

Alternative sources documented for completeness:

- **TheyWorkForYou** — https://www.theyworkforyou.com/api/ (free, free
  registration for API key). Different schema; would need a separate
  adapter.
- **HuggingFace datasets** — search for ``hansard``. Pre-cleaned
  variants with permissive licences. Just :func:`pycorpdiff.from_dataframe`
  the result.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ..corpus import Corpus
from ..io.readers import from_dataframe, read_parquet

DEFAULT_HANSARD_BASE_URL = "https://hansard-api.parliament.uk"
SEARCH_DEBATES_PATH = "/search/debates.json"


def load_hansard_sample() -> Corpus:
    """Return the bundled 193-speech synthetic Hansard sample as a :class:`Corpus`.

    The corpus has columns ``speech_id``, ``text``, ``topic``,
    ``frame``, ``party``, ``date``, ``year``. Frames shift over time
    to mimic real discourse: immigration goes humanising → criminalising
    around 2016 (Brexit referendum), Brexit moves emerging → peak →
    aftermath, NHS has austerity (2010-14) and COVID (2020-22)
    pressure points, climate sharpens scientific → policy → crisis.

    Use this for tutorials, demos, and reproducible package tests. For
    actual research, fetch real Hansard via :func:`fetch_hansard`.
    """
    data_path = Path(__file__).parent / "_data" / "hansard_sample.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Hansard sample not found at {data_path}. The package may have "
            "been installed without its bundled data; re-run "
            "`python -m pycorpdiff.datasets._generate_hansard` to regenerate."
        )
    return read_parquet(
        data_path,
        text_col="text",
        id_col="speech_id",
        meta_cols=("topic", "frame", "party", "date", "year"),
    )


def _http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Plain GET → JSON. Isolated so tests can monkey-patch it cleanly."""
    req = urllib.request.Request(url, headers={"User-Agent": "pycorpdiff/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    result: dict[str, Any] = json.loads(payload)
    return result


def _default_parse_search_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract rows from a Hansard search-results payload.

    The Hansard search endpoint returns a JSON object with a top-level
    list of search hits. Field names vary slightly across endpoints and
    over time; this parser tolerates the common variations
    (``Results`` / ``SearchResults`` / list-at-root) and surfaces a
    canonical set of fields.
    """
    # The response can be {"Results": [...]} or just [...] depending on endpoint.
    if isinstance(payload, list):
        hits = payload
    else:
        hits = (
            payload.get("Results")
            or payload.get("SearchResults")
            or payload.get("Contributions")
            or []
        )
    rows: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        text = (
            hit.get("ContributionText")
            or hit.get("ContentText")
            or hit.get("Snippet")
            or hit.get("Text")
            or ""
        )
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "speaker": hit.get("AttributedTo") or hit.get("MemberName") or "",
                "party": hit.get("MemberParty") or hit.get("Party") or "",
                "date": (hit.get("SittingDate") or hit.get("DebateDate") or "")[:10],
                "debate_title": hit.get("DebateSection") or hit.get("Title") or "",
                "hansard_id": str(
                    hit.get("ContributionExtId")
                    or hit.get("DebateSectionExtId")
                    or hit.get("Id")
                    or ""
                ),
            }
        )
    return rows


def fetch_hansard(
    search_term: str,
    start_date: str,
    end_date: str,
    *,
    max_results: int = 100,
    cache_dir: str | Path | None = None,
    base_url: str = DEFAULT_HANSARD_BASE_URL,
    response_parser: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    _fetch: Callable[[str], dict[str, Any]] | None = None,
) -> Corpus:
    """Fetch UK Hansard speeches matching ``search_term`` and return a :class:`Corpus`.

    Parameters
    ----------
    search_term
        Free-text query passed to the Hansard search API.
    start_date, end_date
        ISO date strings (``"YYYY-MM-DD"``) bounding the search range.
    max_results
        Cap on the number of speeches to retrieve. The API paginates;
        we just take the first page sized at ``max_results``.
    cache_dir
        If given, results are cached as parquet keyed on the URL.
        Subsequent calls with the same arguments read from disk —
        useful for reproducibility and rate-limit etiquette.
    base_url
        Override the default ``https://hansard-api.parliament.uk`` if
        you're hitting a mirror or a staging endpoint.
    response_parser
        Override the default JSON-to-rows parser if the upstream schema
        has changed since this code was written. Receives the decoded
        JSON, returns a list of dicts with at least a ``text`` key.
    _fetch
        Internal hook so tests can substitute the HTTP layer.

    Returns
    -------
    Corpus
        With columns ``text``, ``speaker``, ``party``, ``date``,
        ``debate_title``, ``hansard_id``. Empty if the query returns no
        hits.

    Examples
    --------
    >>> import pycorpdiff as pcd
    >>> corpus = pcd.datasets.hansard.fetch_hansard(  # doctest: +SKIP
    ...     "immigration",
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     max_results=200,
    ...     cache_dir="~/.cache/pycorpdiff/hansard",
    ... )
    """
    fetch = _fetch or _http_get_json
    parse = response_parser or _default_parse_search_response

    params = {
        "queryParameters.searchTerm": search_term,
        "queryParameters.startDate": start_date,
        "queryParameters.endDate": end_date,
        "queryParameters.take": str(max_results),
        "queryParameters.skip": "0",
    }
    url = f"{base_url}{SEARCH_DEBATES_PATH}?{urllib.parse.urlencode(params)}"

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir_p = Path(cache_dir).expanduser()
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir_p / f"hansard_{key}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return from_dataframe(
                df,
                text_col="text",
                meta_cols=("speaker", "party", "date", "debate_title", "hansard_id"),
            )

    payload = fetch(url)
    rows = parse(payload)
    df = pd.DataFrame(
        rows,
        columns=["text", "speaker", "party", "date", "debate_title", "hansard_id"],
    )

    if cache_path is not None and len(df) > 0:
        df.to_parquet(cache_path, index=False)

    return from_dataframe(
        df,
        text_col="text",
        meta_cols=("speaker", "party", "date", "debate_title", "hansard_id"),
    )
