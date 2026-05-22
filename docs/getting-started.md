# Getting started

## Install (from source, until PyPI release)

```bash
git clone https://github.com/jasonsturner/pycorpdiff
cd pycorpdiff
pip install -e ".[dev]"
pytest
```

## Construct a Corpus

```python
import pandas as pd
import pycorpdiff as pcd

df = pd.DataFrame({
    "text": ["the cat sat on the mat", "the dog sat on the log"],
    "outlet": ["A", "B"],
    "year": [2020, 2020],
})
corpus = pcd.from_dataframe(df, text_col="text", meta_cols=("outlet", "year"))
```

## Slice it

```python
a = corpus.slice(outlet="A")
b = corpus.slice(outlet="B")
```

## (Coming in Phase 1) Compare

```python
result = pcd.compare(a, b).keyness()
result.to_df()
result.plot()
```
