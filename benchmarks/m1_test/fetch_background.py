"""Fetch a topic-neutral PubMed *background* corpus (~150 abstracts/year,
2000-2024) and embed it, for the M1 'is background correction useful on the
register-stable core?' test. Reuses the paper's proven E-utilities helpers."""
import os, sys, time, urllib.request, urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, "/Users/jasonturner/Projects/cbd-discourse/build")
from fetch_pubmed_abstracts import esearch_pmids_one_year, _parse_pubmed_article  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PER_YEAR = 150
OUT = Path("/Users/jasonturner/Projects/cbd-discourse/data")

def efetch(pmids):
    url = f"{EUTILS}/efetch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
    xml = urllib.request.urlopen(url, timeout=120).read()
    root = ET.fromstring(xml)
    return [a for a in (_parse_pubmed_article(art) for art in root.findall(".//PubmedArticle")) if a]

rows, periods = [], []
for y in range(2000, 2025):
    try:
        pmids = esearch_pmids_one_year(["disease"], y)[:PER_YEAR]   # broad, topic-neutral
        if not pmids:
            print(f"year {y}: 0 pmids", file=sys.stderr); continue
        for a in efetch(pmids):
            txt = (a.get("title", "") + " " + a.get("abstract", "")).strip()
            if len(txt) > 30:
                rows.append(txt); periods.append(y)
        time.sleep(0.4)
        print(f"year {y}: {sum(p == y for p in periods)} recs (total {len(rows)})", file=sys.stderr)
    except Exception as e:
        print(f"year {y}: ERROR {e}", file=sys.stderr)

print(f"embedding {len(rows)} background records ...", file=sys.stderr)
m = SentenceTransformer("all-MiniLM-L6-v2")
E = m.encode(rows, batch_size=128, show_progress_bar=False).astype("float32")
np.save(OUT / "pubmed_background_embeddings.npy", E)
np.save(OUT / "pubmed_background_periods.npy", np.asarray(periods))
print(f"SAVED {E.shape} background embeddings + {len(periods)} periods", file=sys.stderr)
