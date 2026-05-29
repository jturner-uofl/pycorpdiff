# Multilingual support

`pycorpdiff` is Unicode-native for alphabetic scripts. The default
`RegexTokenizer` uses Python's `\w+` under `re.UNICODE`, which treats
letters from any script as word characters:

```python
import pycorpdiff as pcd

tok = pcd.RegexTokenizer()
tok("hello мир γειά τέλος")
# → ['hello', 'мир', 'γειά', 'τέλος']
```

Emoji, punctuation-only sequences, and CJK-without-spaces are
**not** captured by `\w+`; they require a script-aware tokeniser (see
the adapters below). The default is fine for Latin / Cyrillic /
Greek / accented-Latin starting points — but real multilingual work
usually wants language-specific behaviour (lemmatisation, CJK
segmentation, agglutinative-language handling, MWE rules). The
`Tokenizer` Protocol is the plug point.

## The Tokenizer protocol

Anything callable that maps a string to a list of token strings
satisfies it:

```python
from typing import Protocol

class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[str]: ...
```

One-line adapters cover most of the popular NLP libraries.

## spaCy adapter

```python
import spacy
import pycorpdiff as pcd

nlp = spacy.load("en_core_web_sm")  # or "de_core_news_sm", "es_core_news_sm", ...

class SpacyTokenizer:
    def __init__(self, nlp, *, lemmatize: bool = False):
        self.nlp = nlp
        self.lemmatize = lemmatize

    def __call__(self, text: str) -> list[str]:
        doc = self.nlp(text, disable=["parser", "ner"])
        if self.lemmatize:
            return [tok.lemma_.lower() for tok in doc if not tok.is_punct]
        return [tok.text.lower() for tok in doc if not tok.is_punct]

corpus = pcd.from_dataframe(
    df, text_col="text", tokenizer=SpacyTokenizer(nlp, lemmatize=True),
)
```

## Stanza adapter (for less-resourced languages)

```python
import stanza
import pycorpdiff as pcd

stanza.download("fi")  # Finnish, for example
nlp = stanza.Pipeline("fi", processors="tokenize,lemma")

class StanzaTokenizer:
    def __init__(self, nlp): self.nlp = nlp
    def __call__(self, text: str) -> list[str]:
        doc = self.nlp(text)
        return [word.lemma.lower() for sent in doc.sentences for word in sent.words]

corpus = pcd.from_dataframe(df, text_col="text", tokenizer=StanzaTokenizer(nlp))
```

## jieba adapter (for Chinese)

```python
import jieba
import pycorpdiff as pcd

class JiebaTokenizer:
    def __call__(self, text: str) -> list[str]:
        return [tok for tok in jieba.cut(text) if tok.strip()]

corpus = pcd.from_dataframe(df, text_col="text", tokenizer=JiebaTokenizer())
```

## fugashi adapter (for Japanese)

```python
import fugashi
import pycorpdiff as pcd

class FugashiTokenizer:
    def __init__(self):
        self.tagger = fugashi.Tagger()
    def __call__(self, text: str) -> list[str]:
        return [word.surface for word in self.tagger(text)]

corpus = pcd.from_dataframe(df, text_col="text", tokenizer=FugashiTokenizer())
```

## Swapping tokenizers on an existing corpus

```python
corpus = pcd.read_parquet("news.parquet", text_col="body")
lemmatized = corpus.with_tokenizer(SpacyTokenizer(nlp, lemmatize=True))
```

Tokenizers are stored on the `Corpus` (and shared with all of its
slices); swapping them is an immutable transformation that returns a
new `Corpus`.

## Multilingual embeddings

If you use the semantic-shift API, pair your multilingual tokenizer
with a multilingual embedder:

```python
embedder = pcd.SBERTEmbedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")
result = pcd.compare(a, b).semantic_shift("Migrant", embedder=embedder)
```

The default `all-MiniLM-L6-v2` is English-only. The sentence-transformers
hub has multilingual options spanning 50+ languages.

## What pycorpdiff deliberately does *not* do

- Train tokenisers or language models.
- Ship language-specific preprocessing rules.
- Detect document language automatically.

These are well-served by the libraries above. The comparative-analysis
layer is what `pycorpdiff` provides; the linguistic preprocessing is
borrowed in from the ecosystem.
