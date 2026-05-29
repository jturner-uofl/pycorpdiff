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
import html
import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from ..corpus import Corpus
from ..io.readers import from_dataframe, read_parquet

# Hansard API responses embed HTML markup in the speech text
# (``<em>``, ``<span>``, ``<strong>``, ``<p>``, plus the proprietary
# ``<TableWrapper ...>`` for parliamentary tables). Naïve tokenisation
# turns the tag *names* into apparent words (``em`` shows up ~10 000
# times in a moderate corpus, polluting every keyness / collocation
# / network analysis). The strip is applied as text comes off the wire.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_hansard_text(raw: str) -> str:
    """Strip HTML markup + decode entities + collapse whitespace.

    Applied to ``ContributionTextFull`` / ``ContributionText`` before the
    text is stored as the canonical document body.
    """
    if not raw:
        return ""
    no_tags = _HTML_TAG_RE.sub(" ", raw)
    decoded = html.unescape(no_tags)
    return _WHITESPACE_RE.sub(" ", decoded).strip()

DEFAULT_HANSARD_BASE_URL = "https://hansard-api.parliament.uk"
# Spoken contributions endpoint. The earlier ``/search/debates.json``
# returned only debate-level metadata (titles, sitting dates) without
# the speech text itself — fetch_hansard always came back empty even
# when the API was reachable. The contributions endpoint returns the
# actual ``ContributionText`` along with ``MemberId``, ``MemberName``,
# ``DebateSection``, and ``SittingDate``.
SEARCH_CONTRIBUTIONS_PATH = "/search/contributions/Spoken.json"
DEFAULT_MEMBERS_BASE_URL = "https://members-api.parliament.uk"
MEMBERS_PATH = "/api/Members/{member_id}"


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
        # Prefer ``ContributionTextFull`` (untruncated) when present;
        # fall back to the highlighted ``ContributionText`` and then to
        # the legacy field names from older endpoint schemas.
        raw_text = (
            hit.get("ContributionTextFull")
            or hit.get("ContributionText")
            or hit.get("ContentText")
            or hit.get("Snippet")
            or hit.get("Text")
            or ""
        )
        # Strip HTML markup the API embeds in contribution text
        # (``<em>``, ``<span>``, ``<TableWrapper>`` etc.). Without this
        # step, tag names tokenise as apparent English words and
        # systematically inflate the keyness signal.
        text = _clean_hansard_text(raw_text)
        if not text:
            continue
        member_id_raw = hit.get("MemberId")
        try:
            member_id: int | None = int(member_id_raw) if member_id_raw is not None else None
        except (TypeError, ValueError):
            member_id = None
        rows.append(
            {
                "text": text,
                "speaker": hit.get("AttributedTo") or hit.get("MemberName") or "",
                "member_id": member_id,
                "party": hit.get("MemberParty") or hit.get("Party") or "",
                "date": (hit.get("SittingDate") or hit.get("DebateDate") or "")[:10],
                "debate_title": hit.get("DebateSection") or hit.get("Title") or "",
                "house": hit.get("House") or "",
                "hansard_id": str(
                    hit.get("ContributionExtId")
                    or hit.get("DebateSectionExtId")
                    or hit.get("Id")
                    or ""
                ),
            }
        )
    return rows


def _enrich_with_party(
    rows: list[dict[str, Any]],
    *,
    members_base_url: str,
    _fetch: Callable[[str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Look up party affiliation per unique MemberId via the Members API.

    The Hansard contributions endpoint carries ``MemberId`` but not the
    member's party. For cross-party comparative work (the standard
    pycorpdiff use case), party is essential — so we offer a one-call-per-
    unique-member enrichment behind the ``enrich_party=True`` flag.
    The Members API is unauthenticated; ~0.3s per call.
    """
    unique_ids = {r["member_id"] for r in rows if r.get("member_id") is not None}
    party_map: dict[int, dict[str, str]] = {}
    for mid in unique_ids:
        url = f"{members_base_url}{MEMBERS_PATH.format(member_id=mid)}"
        try:
            data = _fetch(url)
        except Exception:
            party_map[mid] = {"party": "", "party_abbrev": ""}
            continue
        value = data.get("value", {}) if isinstance(data, dict) else {}
        party = value.get("latestParty") or {}
        party_map[mid] = {
            "party": party.get("name", "") or "",
            "party_abbrev": party.get("abbreviation", "") or "",
        }
    for row in rows:
        mid = row.get("member_id")
        info = party_map.get(mid) if mid is not None else None
        if info is not None:
            row["party"] = info["party"] or row.get("party", "")
            row["party_abbrev"] = info["party_abbrev"]
        else:
            row.setdefault("party_abbrev", "")
    return rows


def fetch_hansard(
    search_term: str,
    start_date: str,
    end_date: str,
    *,
    max_results: int = 100,
    page_size: int = 50,
    enrich_party: bool = False,
    cache_dir: str | Path | None = None,
    base_url: str = DEFAULT_HANSARD_BASE_URL,
    members_base_url: str = DEFAULT_MEMBERS_BASE_URL,
    response_parser: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
    _fetch: Callable[[str], dict[str, Any]] | None = None,
) -> Corpus:
    """Fetch UK Hansard spoken contributions and return a :class:`Corpus`.

    Hits the ``/search/contributions/Spoken.json`` endpoint of the
    Hansard search API (Commons + Lords spoken contributions) and
    paginates automatically until ``max_results`` is reached or the
    upstream `TotalResultCount` is exhausted, whichever comes first.

    Parameters
    ----------
    search_term
        Free-text query passed to the Hansard search API. Phrase
        searches; the API does not natively expand OR / boolean
        operators the way some other archives do.
    start_date, end_date
        ISO date strings (``"YYYY-MM-DD"``) bounding the search range.
    max_results
        Cap on the number of contributions returned. The function
        paginates internally; values above the API's per-request
        ``page_size`` are handled transparently.
    page_size
        Per-request batch size. Default 50 matches the API's stable
        cap; raising above ~100 has been observed to return 500 errors.
    enrich_party
        When ``True``, calls the Members API once per unique
        ``MemberId`` to populate ``party`` and ``party_abbrev`` columns.
        Adds ~0.3 s per unique member; on a 10k-contribution fetch
        with ~1k unique speakers this is ~5 minutes. Required for any
        downstream cross-party comparative analysis.
    cache_dir
        If given, results are cached as parquet keyed on the URL +
        parameters. Subsequent calls with the same arguments read from
        disk — useful for reproducibility and rate-limit etiquette.
    base_url
        Override the default ``https://hansard-api.parliament.uk`` if
        you're hitting a mirror or a staging endpoint.
    members_base_url
        Override the default ``https://members-api.parliament.uk`` for
        the party-enrichment step. Ignored when ``enrich_party=False``.
    response_parser
        Override the default JSON-to-rows parser if the upstream schema
        has changed since this code was written. Receives the decoded
        JSON, returns a list of dicts with at least a ``text`` key.
    _fetch
        Internal hook so tests can substitute the HTTP layer.

    Returns
    -------
    Corpus
        With columns ``text``, ``speaker``, ``member_id``, ``party``,
        ``date``, ``debate_title``, ``house``, ``hansard_id`` —
        plus ``party_abbrev`` when ``enrich_party=True``. Empty if the
        query returns no hits.

    Examples
    --------
    >>> import pycorpdiff as pcd
    >>> corpus = pcd.fetch_hansard(  # doctest: +SKIP
    ...     "asylum",
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     max_results=500,
    ...     enrich_party=True,
    ...     cache_dir="~/.cache/pycorpdiff/hansard",
    ... )
    """
    fetch = _fetch or _http_get_json
    parse = response_parser or _default_parse_search_response

    # Cache lookup is keyed on (query, dates, max_results, enrich_party).
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir_p = Path(cache_dir).expanduser()
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        key_src = (
            f"{search_term}|{start_date}|{end_date}|{max_results}|{enrich_party}"
        )
        key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
        cache_path = cache_dir_p / f"hansard_{key}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            meta_cols = tuple(c for c in df.columns if c != "text")
            return from_dataframe(df, text_col="text", meta_cols=meta_cols)

    # Paginate. The API caps a single request at ~50-100; this loop
    # walks pages until we hit `max_results` or run out of upstream rows.
    rows: list[dict[str, Any]] = []
    skip = 0
    while len(rows) < max_results:
        take = min(page_size, max_results - len(rows))
        params = {
            "queryParameters.searchTerm": search_term,
            "queryParameters.startDate": start_date,
            "queryParameters.endDate": end_date,
            "queryParameters.take": str(take),
            "queryParameters.skip": str(skip),
        }
        url = f"{base_url}{SEARCH_CONTRIBUTIONS_PATH}?{urllib.parse.urlencode(params)}"
        payload = fetch(url)
        page_rows = parse(payload)
        if not page_rows:
            break
        rows.extend(page_rows)
        skip += len(page_rows)
        # Stop when the API returns fewer rows than asked for (we've
        # drained it) or when ``TotalResultCount`` says we're done.
        total = int(payload.get("TotalResultCount", 0)) if isinstance(payload, dict) else 0
        if total and skip >= total:
            break
        if len(page_rows) < take:
            break

    if enrich_party and rows:
        rows = _enrich_with_party(
            rows, members_base_url=members_base_url, _fetch=fetch
        )

    base_meta = ["speaker", "member_id", "party", "date", "debate_title", "house", "hansard_id"]
    if enrich_party:
        base_meta.append("party_abbrev")
    df = pd.DataFrame(rows, columns=["text", *base_meta])

    if cache_path is not None and len(df) > 0:
        df.to_parquet(cache_path, index=False)

    return from_dataframe(df, text_col="text", meta_cols=tuple(base_meta))
