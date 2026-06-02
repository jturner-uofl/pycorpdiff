"""Full title+abstract harvest for the PubMed terminology case study.

Step B of the pilot → study pipeline: now that Step A (per-year counts
across the 57-pair inventory) has identified the 5 headline shifts, we
fetch the actual title+abstract text for each side of each shift.
That gives pycorpdiff something to chew on — semantic trajectory,
keyness pre/post anchor, neighbourhood drift, causal_impact at the
anchor date — same pipeline as the CBD case study.

Headline shifts (5 chosen for anchor-decade diversity + one striking
negative finding):

  1960s  mongolism             -> Down syndrome / trisomy 21
  1980s  shell shock + family  -> PTSD
  1990s  multiple personality  -> dissociative identity
  2010s  mental retardation    -> intellectual disability
  ----   "committed suicide"   -> "died by suicide"     (NEGATIVE FINDING:
                                                         new phrase has
                                                         ~zero penetration
                                                         in PubMed despite
                                                         AAS recommendations)

Per-term [Title/Abstract] qualification — see fetch_pubmed.py
docstring on why this matters; auto-mapping would otherwise drag in
modernised MeSH synonyms.

Output: one parquet per (pair, side) under data/pubmed_abstracts/.
Each row: pmid, title, abstract, journal, year, mesh_terms (list[str]),
pair (str), side ('old' or 'new'), term_label (str).

Public domain (US government data); free to redistribute.
"""

from __future__ import annotations

import argparse
import http.client
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = (
    "pycorpdiff-pubmed-terminology/0.1 "
    "(https://github.com/jturner-uofl/pycorpdiff)"
)
RETMAX_PER_PAGE = 9999  # esearch returns IDs in pages up to 10000
EFETCH_BATCH = 200      # NLM recommends <= 500/batch; 200 is conservative


_TRANSIENT_EXCS = (
    HTTPError,
    URLError,
    TimeoutError,
    http.client.HTTPException,   # covers IncompleteRead, RemoteDisconnected, etc.
    ConnectionError,             # OSError subclass; defensive
)


def _http_get(url: str, *, max_retries: int = 6) -> bytes:
    """GET a URL with exponential backoff over the wider set of transient errors.

    NCBI's E-utilities returns chunked-encoded responses; a stream that
    closes mid-read raises ``http.client.IncompleteRead`` (which is an
    ``HTTPException`` subclass, NOT an ``HTTPError`` from ``urllib.error``).
    We caught only ``HTTPError`` previously and the process died on a
    mid-batch efetch. Broadened to ``HTTPException`` here.
    """
    backoff = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=120) as r:
                return r.read()
        except _TRANSIENT_EXCS as e:
            last_err = e
            # 4xx (except 429) is non-transient — bail immediately.
            if isinstance(e, HTTPError) and 400 <= e.code < 500 and e.code != 429:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(
        f"http_get failed after {max_retries} retries: "
        f"{type(last_err).__name__}: {last_err}"
    )


def _http_get_json_loose(url: str, *, max_retries: int = 5) -> dict:
    """GET a JSON endpoint, tolerating embedded control characters.

    NCBI's paginated esearch JSON responses sometimes contain stray
    control bytes that the strict JSON decoder rejects. We retry the
    request; if all retries still produce un-strict-parseable JSON,
    fall back to non-strict mode (allows raw control chars in strings).
    """
    import json
    last_err: Exception | None = None
    backoff = 1.0
    for _ in range(max_retries):
        body = _http_get(url)
        text = body.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            try:
                return json.loads(text, strict=False)
            except json.JSONDecodeError as e2:
                last_err = e2
                time.sleep(backoff)
                backoff *= 2
                continue
    raise RuntimeError(f"esearch JSON parse failed after {max_retries} retries: {last_err}")


def _build_query(terms: Iterable[str]) -> str:
    """OR-join with per-term [Title/Abstract] qualifier."""
    return " OR ".join(f"{t}[Title/Abstract]" for t in terms)


def esearch_pmids_one_year(
    terms: list[str],
    year: int,
    *,
    api_key: str | None = None,
    sleep: float = 0.4,
) -> list[str]:
    """Return all PMIDs for `terms` published in a single year.

    Per-year esearch keeps each call's result set well under the
    ~10,000-PMID effective ceiling on NCBI's esearch endpoint (the
    history-server pagination silently truncates beyond that cap on
    some queries — a verified June-2026 failure mode where
    1980s_ptsd_new came back with only the most recent 9,943 of
    ~59,000 actual records). Calling per-year is robust and the
    per-year volume for our headline terms peaks ~6,000/year — well
    inside the limit.
    """
    if api_key:
        sleep = min(sleep, 0.12)
    params = {
        "db": "pubmed",
        "term": _build_query(terms),
        "datetype": "pdat",
        "mindate": str(year),
        "maxdate": str(year),
        "retmax": RETMAX_PER_PAGE,
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{EUTILS}/esearch.fcgi?{urlencode(params)}"
    data = _http_get_json_loose(url)
    time.sleep(sleep)
    result = data.get("esearchresult", {})
    return list(result.get("idlist", []))


def esearch_pmids(
    terms: list[str],
    *,
    start_year: int,
    end_year: int,
    api_key: str | None = None,
    sleep: float = 0.4,
    progress: bool = True,
) -> list[str]:
    """Return all PMIDs matching `terms` in title/abstract, given date range.

    Year-by-year iteration: one esearch call per year. See
    ``esearch_pmids_one_year`` for the rationale (avoids the
    ~10K-PMID ceiling on history-server-paginated queries). Total
    cost: ``(end_year - start_year + 1)`` esearch calls — fine at
    NCBI's 3 req/s rate limit for our 75-year windows.
    """
    all_pmids: list[str] = []
    seen: set[str] = set()
    n_years = end_year - start_year + 1
    for i, year in enumerate(range(start_year, end_year + 1)):
        year_pmids = esearch_pmids_one_year(
            terms, year, api_key=api_key, sleep=sleep
        )
        # Deduplicate (overlapping years shouldn't occur, but defensive)
        new_pmids = [p for p in year_pmids if p not in seen]
        seen.update(new_pmids)
        all_pmids.extend(new_pmids)
        if progress:
            print(
                f"  esearch year {year} ({i + 1}/{n_years}): "
                f"{len(year_pmids):>5,} PMIDs ({len(all_pmids):>6,} total)",
                file=sys.stderr,
            )
    return all_pmids


def _parse_pubmed_article(article: ET.Element) -> dict | None:
    """Parse a single <PubmedArticle> element into a flat dict."""
    pmid_el = article.find(".//PMID")
    if pmid_el is None or pmid_el.text is None:
        return None
    pmid = pmid_el.text

    # Title
    title_el = article.find(".//Article/ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""

    # Abstract — may have multiple AbstractText elements (structured abstracts)
    abstract_parts: list[str] = []
    for abs_el in article.findall(".//Article/Abstract/AbstractText"):
        label = abs_el.get("Label", "")
        text = "".join(abs_el.itertext()).strip()
        if not text:
            continue
        abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = " ".join(abstract_parts)

    # Journal title (full or abbrev)
    journal_el = article.find(".//Journal/Title")
    journal = journal_el.text.strip() if (journal_el is not None and journal_el.text) else ""

    # Publication year: prefer ArticleDate, fall back to PubDate
    year_el = article.find(".//Article/ArticleDate/Year")
    if year_el is None or year_el.text is None:
        year_el = article.find(".//Article/Journal/JournalIssue/PubDate/Year")
    if year_el is None or year_el.text is None:
        # Some old records use MedlineDate ("1965-1966")
        md = article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
        if md is not None and md.text:
            year = md.text.strip()[:4]
        else:
            year = ""
    else:
        year = year_el.text.strip()
    try:
        year_int = int(year)
    except ValueError:
        year_int = None  # type: ignore

    # MeSH descriptors
    mesh_terms = []
    for mh in article.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
        if mh.text:
            mesh_terms.append(mh.text.strip())

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year_int,
        "mesh_terms": mesh_terms,
    }


def efetch_records(
    pmids: list[str],
    *,
    api_key: str | None = None,
    sleep: float = 0.4,
    batch_size: int = EFETCH_BATCH,
    progress: bool = True,
) -> list[dict]:
    """Bulk-fetch PubMed records for a list of PMIDs. Returns flat dicts."""
    if api_key:
        sleep = min(sleep, 0.12)
    if not pmids:
        return []
    out: list[dict] = []
    n_batches = (len(pmids) + batch_size - 1) // batch_size
    t0 = time.time()
    for i in range(0, len(pmids), batch_size):
        chunk = pmids[i : i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "rettype": "xml",
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key
        url = f"{EUTILS}/efetch.fcgi?{urlencode(params)}"
        body = _http_get(url)
        root = ET.fromstring(body)
        for art in root.findall(".//PubmedArticle"):
            parsed = _parse_pubmed_article(art)
            if parsed is not None:
                out.append(parsed)
        if progress:
            done = (i // batch_size) + 1
            elapsed = time.time() - t0
            print(
                f"  efetch batch {done:>3}/{n_batches:>3}: "
                f"{len(out):>6,} records so far ({elapsed:.0f}s)",
                file=sys.stderr,
            )
        time.sleep(sleep)
    return out


# Headline shifts — see module docstring.
#
# Each value is (old_terms, new_terms, start_year, end_year). The
# pair is queried twice (once per side); the resulting two parquets
# stay separate so downstream can do pycorpdiff `compare(old, new)`.

HEADLINE_PAIRS: dict[str, tuple[list[str], list[str], int, int]] = {
    # 1960s anchor: WHO ICD-8 rename ~1965 / Lancet 1961
    "1960s_down": (
        ["mongolism", '"Mongolian idiocy"'],
        ['"Down syndrome"', '"Down\'s syndrome"', '"trisomy 21"'],
        1950, 2024,
    ),
    # 1980s anchor: DSM-III introduces PTSD
    "1980s_ptsd": (
        ['"shell shock"', '"war neurosis"', '"combat fatigue"'],
        ['"post-traumatic stress disorder"',
         '"posttraumatic stress disorder"', "PTSD"],
        1940, 2024,
    ),
    # 1990s anchor: DSM-IV renames MPD -> DID
    "1990s_did": (
        ['"multiple personality disorder"', '"multiple personality"'],
        ['"dissociative identity disorder"'],
        1950, 2024,
    ),
    # 2010s anchor: Rosa's Law 2010 + DSM-5 2013
    "2010s_id": (
        ['"mental retardation"', '"mentally retarded"', '"mental retardate"'],
        ['"intellectual disability"', '"intellectually disabled"',
         '"intellectual disabilities"'],
        1950, 2024,
    ),
    # Negative finding: AAS-recommended phrasing has ~zero PubMed penetration
    "neg_suicide_phrasing": (
        ['"committed suicide"', '"commits suicide"'],
        ['"died by suicide"'],
        1970, 2024,
    ),
    # iter-5c: Sepsis-3 (2016) operational-definition revision archetype.
    # Old: SIRS criteria (1991) + Sepsis-2 framing (2001) — systemic
    # inflammatory response syndrome + sepsis-related organ failure.
    # New: Sepsis-3 definition (Singer et al., JAMA 2016) — replaced SIRS
    # with SOFA score + introduced qSOFA bedside screen + redefined
    # septic shock by lactate + vasopressor requirement.
    "2016_sepsis3": (
        ['"systemic inflammatory response syndrome"', "SIRS",
         '"severe sepsis"',
         '"sepsis syndrome"'],
        ['"Sepsis-3"', '"Sepsis 3"', "qSOFA", '"quick SOFA"',
         '"Third International Consensus Definitions for Sepsis"'],
        1990, 2024,
    ),
    # iter-5d: Asperger's -> ASD (DSM-5 2013) dual-rationale retirement
    # archetype. Terminology rationale: DSM-5 folded Asperger's syndrome
    # + PDD-NOS + childhood disintegrative disorder into Autism Spectrum
    # Disorder. Ethical rationale: Czech (2018) and Sheffer (2018)
    # documented Hans Asperger's wartime collaboration with the Vienna
    # Spiegelgrund child-euthanasia program, accelerating
    # community/clinical retirement of the eponym.
    "2013_asperger": (
        ['"Asperger syndrome"', '"Asperger\'s syndrome"',
         '"Asperger disorder"', '"Asperger\'s disorder"'],
        ['"autism spectrum disorder"', '"autism spectrum disorders"', "ASD"],
        1980, 2024,
    ),
}


def fetch_pair(
    label: str,
    old_terms: list[str],
    new_terms: list[str],
    start_year: int,
    end_year: int,
    out_dir: Path,
    *,
    api_key: str | None = None,
) -> dict:
    """Fetch both sides of a pair; write two parquets. Cached."""
    out_dir.mkdir(exist_ok=True, parents=True)
    stats: dict[str, int] = {}
    for side, terms in (("old", old_terms), ("new", new_terms)):
        path = out_dir / f"{label}_{side}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  [cache] {label}/{side}: {len(df):,} records at {path.name}",
                  file=sys.stderr)
            stats[f"{side}_records"] = len(df)
            continue
        print(f"[fetch] {label}/{side}: searching {' OR '.join(terms)[:60]}... "
              f"[{start_year}-{end_year}]", file=sys.stderr)
        pmids = esearch_pmids(terms, start_year=start_year, end_year=end_year,
                              api_key=api_key)
        print(f"  esearch found {len(pmids):,} PMIDs", file=sys.stderr)
        records = efetch_records(pmids, api_key=api_key)
        df = pd.DataFrame(records)
        if len(df):
            df["pair"] = label
            df["side"] = side
            df["term_label"] = " OR ".join(terms)
            df.to_parquet(path, index=False)
            print(f"  wrote {len(df):,} records to {path}", file=sys.stderr)
        else:
            print(f"  (no records — likely a true zero, "
                  f"e.g. the negative-finding pair)", file=sys.stderr)
            # Write an empty marker so we don't re-fetch
            pd.DataFrame(columns=["pmid", "title", "abstract", "journal", "year",
                                  "mesh_terms", "pair", "side", "term_label"]).to_parquet(path, index=False)
        stats[f"{side}_records"] = len(df)
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-key", default=None)
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to pair labels (e.g. 1960s_down 1980s_ptsd).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pubmed_abstracts",
    )
    args = p.parse_args(argv)

    summary = {}
    for label, (old_terms, new_terms, s, e) in HEADLINE_PAIRS.items():
        if args.only and label not in args.only:
            continue
        stats = fetch_pair(label, old_terms, new_terms, s, e,
                           args.out_dir, api_key=args.api_key)
        summary[label] = stats

    print("\n=== Fetch summary ===", file=sys.stderr)
    for label, st in summary.items():
        print(f"  {label}: old={st.get('old_records', 0):,}  "
              f"new={st.get('new_records', 0):,}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
