"""Home-run step 1: build + embed a weekly CBD-tweet corpus, 2019 reference +
weekly 2020 (Jan-Jun), for unsupervised emergence detection of the CBD-COVID
sub-discourse. Light polysemy filter (drop business-district)."""
import os, re; os.environ.update(HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA = Path("/Users/jasonturner/Projects/cbd-discourse/data")
df = pd.read_parquet(DATA / "cbd_tweets_2011_2021.parquet", columns=["date", "text"])
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "text"])
df["text"] = df["text"].astype(str)
DISTRICT = re.compile(r"business district|central business|\bcbd location|real estate|"
                      r"for lease|office space|apartment", re.I)
df = df[~df["text"].str.contains(DISTRICT)]
df["wk"] = df["date"].dt.strftime("%G-W%V")   # ISO year-week, sortable

ref = df[df.date.dt.year == 2019]
ref = ref.sample(min(6000, len(ref)), random_state=42).assign(period="2019")
mon = df[(df.date >= "2020-01-01") & (df.date < "2020-07-01")].copy()
mon["period"] = mon["wk"]
mon = mon.groupby("period", group_keys=False).apply(
    lambda g: g.sample(min(1000, len(g)), random_state=42))
samp = pd.concat([ref, mon[ref.columns]]).reset_index(drop=True)
print(f"reference(2019)={len(ref)} | monitor(2020 wk)={len(mon)} "
      f"weeks={mon.period.nunique()} | total={len(samp)}")

m = SentenceTransformer("all-MiniLM-L6-v2")
E = m.encode(samp["text"].tolist(), batch_size=256, show_progress_bar=False).astype("float32")
np.save(DATA / "covid_corpus_emb.npy", E)
samp[["text", "date", "period", "wk"]].to_parquet(DATA / "covid_corpus_meta.parquet")
print(f"SAVED {E.shape} embeddings + meta ({samp.period.nunique()} periods)")
