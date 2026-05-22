"""Core ``Corpus`` and ``CorpusSlice`` data structures.

A :class:`Corpus` wraps a :class:`pandas.DataFrame` of documents plus
metadata. Slicing returns a :class:`CorpusSlice` that shares the
parent's configuration (text column, tokenizer) but presents a
boolean-masked view. Both objects are immutable frozen dataclasses;
mutations produce new objects.

**Polars interop.** A Corpus stores its documents internally as a
pandas DataFrame because that's what the analytical layer is built on,
but the constructors and round-trip helpers accept and produce polars
DataFrames so they slot into polars-native pipelines:

>>> import polars as pl
>>> df = pl.DataFrame({"text": ["the cat sat"], "outlet": ["A"]})
>>> corpus = pcd.from_dataframe(df, text_col="text")  # polars → pandas internally
>>> corpus.to_polars().shape   # round-trip back
(1, 2)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .tokenize import RegexTokenizer, Tokenizer

if TYPE_CHECKING:
    import polars as pl

    from .temporal.slicing import TemporalCorpus


def _doc_term_counts(
    docs: pd.DataFrame,
    text_col: str,
    tokenizer: Tokenizer,
    min_count: int = 1,
) -> pd.DataFrame:
    """Build a docs × term integer count matrix.

    The result is dense (``int64``) and indexed by the parent frame's index.
    Sparse representations are deferred to a later phase — for the corpus
    sizes pycorpdiff targets in MVP scope (medium millions of tokens) dense
    is fast enough and downstream operations stay vectorisable.
    """
    counts_per_doc: list[Counter[str]] = [Counter(tokenizer(t)) for t in docs[text_col]]
    all_terms: list[str] = sorted({term for c in counts_per_doc for term in c})
    term_to_idx: dict[str, int] = {t: i for i, t in enumerate(all_terms)}

    data = np.zeros((len(counts_per_doc), len(all_terms)), dtype=np.int64)
    for i, doc_counts in enumerate(counts_per_doc):
        for term, n in doc_counts.items():
            data[i, term_to_idx[term]] = n

    df = pd.DataFrame(data, columns=all_terms, index=docs.index)
    if min_count > 1:
        df = df.loc[:, df.sum(axis=0) >= min_count]
    return df


def _coerce_to_pandas(docs: Any) -> pd.DataFrame:
    """Accept a pandas or polars DataFrame; return a pandas one.

    The analytical layer is pandas-based; this function is the single
    boundary where polars input gets converted. ``polars.DataFrame``
    has ``.to_pandas()`` so the conversion is one method call.
    """
    if isinstance(docs, pd.DataFrame):
        return docs
    # Defer the polars import — it's an optional dep.
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        pl = None  # type: ignore[assignment]
    if pl is not None and isinstance(docs, pl.DataFrame):
        return docs.to_pandas()
    raise TypeError(
        f"docs must be a pandas or polars DataFrame; got {type(docs).__name__}"
    )


@dataclass(frozen=True)
class Corpus:
    """A corpus of documents with optional metadata columns.

    Parameters
    ----------
    docs
        A DataFrame whose rows are documents. Must contain at least the
        text column named by ``text_col``. Accepts either
        :class:`pandas.DataFrame` or :class:`polars.DataFrame`; polars
        input is converted to pandas internally.
    text_col
        Name of the column containing document text.
    id_col
        Name of an optional unique-document-id column.
    meta_cols
        Tuple of column names treated as metadata available for slicing.
        If empty (the default), every non-text column is considered
        metadata.
    tokenizer
        A callable conforming to :class:`pycorpdiff.tokenize.Tokenizer`.
        Defaults to the package's :class:`RegexTokenizer`.
    """

    docs: pd.DataFrame
    text_col: str = "text"
    id_col: str | None = None
    meta_cols: tuple[str, ...] = ()
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)

    def __post_init__(self) -> None:
        # Coerce polars to pandas if needed; otherwise leave alone.
        if not isinstance(self.docs, pd.DataFrame):
            object.__setattr__(self, "docs", _coerce_to_pandas(self.docs))
        if self.text_col not in self.docs.columns:
            raise ValueError(
                f"text_col={self.text_col!r} not found in DataFrame columns "
                f"{list(self.docs.columns)!r}"
            )
        if self.id_col is not None and self.id_col not in self.docs.columns:
            raise ValueError(
                f"id_col={self.id_col!r} not found in DataFrame columns "
                f"{list(self.docs.columns)!r}"
            )

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def metadata_columns(self) -> tuple[str, ...]:
        """Effective metadata columns — explicit if given, else inferred."""
        if self.meta_cols:
            return self.meta_cols
        return tuple(c for c in self.docs.columns if c != self.text_col)

    def slice(self, **filters: Any) -> CorpusSlice:
        """Return a :class:`CorpusSlice` filtered on metadata columns.

        Each keyword argument is a column name; the value may be a scalar
        (exact match) or an iterable (membership). All conditions are
        combined with logical AND.
        """
        mask = pd.Series(True, index=self.docs.index)
        for col, value in filters.items():
            if col not in self.docs.columns:
                raise KeyError(f"slice() got unknown column {col!r}")
            if isinstance(value, (list, tuple, set, pd.Series)):
                mask &= self.docs[col].isin(list(value))
            else:
                mask &= self.docs[col] == value
        return CorpusSlice(parent=self, mask=mask, filters=dict(filters))

    def by_time(self, col: str, freq: str = "Y") -> TemporalCorpus:
        """Return a :class:`TemporalCorpus` indexed by time-period.

        ``col`` must be parseable as datetime; ``freq`` is any pandas
        offset alias (``"Y"``, ``"Q"``, ``"M"``, ``"W"``, ``"D"``).
        """
        from .temporal.slicing import TemporalCorpus  # local import to break cycle

        return TemporalCorpus(parent=self, time_col=col, freq=freq)

    def with_tokenizer(self, tokenizer: Tokenizer) -> Corpus:
        """Return a copy of the corpus with a different tokenizer."""
        return replace(self, tokenizer=tokenizer)

    def tokens(self) -> list[list[str]]:
        """Tokenize every document; return one list of tokens per doc."""
        return [self.tokenizer(t) for t in self.docs[self.text_col]]

    def doc_term_counts(self, min_count: int = 1) -> pd.DataFrame:
        """Return a docs × term integer count DataFrame."""
        return _doc_term_counts(self.docs, self.text_col, self.tokenizer, min_count)

    def vocab(self, min_count: int = 1) -> pd.Series:
        """Return a term → total-count Series sorted descending."""
        counts = self.doc_term_counts(min_count=min_count).sum(axis=0)
        return counts.rename("count").sort_values(ascending=False)

    def total_tokens(self) -> int:
        """Total tokens across all documents (before any min_count filter)."""
        return int(self.doc_term_counts(min_count=1).values.sum())

    def to_polars(self) -> pl.DataFrame:
        """Return the corpus's documents as a polars DataFrame.

        Requires the ``polars`` extra. The original pandas index is
        dropped (polars has no concept of a row index); the document
        ordering is preserved.
        """
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_polars() requires polars. Install with: pip install 'pycorpdiff[polars]'"
            ) from exc
        return pl.from_pandas(self.docs.reset_index(drop=True))


@dataclass(frozen=True)
class CorpusSlice:
    """A boolean-masked view of a :class:`Corpus`.

    Slices behave like corpora for downstream analytical purposes —
    they expose the same ``docs``, ``text_col``, ``tokenizer`` surface —
    but also remember the ``filters`` that produced them, which the
    :class:`pycorpdiff.compare.Comparison` machinery uses to label
    plots and result tables.
    """

    parent: Corpus
    mask: pd.Series
    filters: dict[str, Any]

    @property
    def docs(self) -> pd.DataFrame:
        return self.parent.docs.loc[self.mask]

    @property
    def text_col(self) -> str:
        return self.parent.text_col

    @property
    def id_col(self) -> str | None:
        return self.parent.id_col

    @property
    def tokenizer(self) -> Tokenizer:
        return self.parent.tokenizer

    def __len__(self) -> int:
        return int(self.mask.sum())

    @property
    def label(self) -> str:
        """A short human-readable label derived from the slice's filters."""
        if not self.filters:
            return "slice"
        return ", ".join(f"{k}={v!r}" for k, v in self.filters.items())

    def tokens(self) -> list[list[str]]:
        """Tokenize every document in the slice."""
        return [self.tokenizer(t) for t in self.docs[self.text_col]]

    def doc_term_counts(self, min_count: int = 1) -> pd.DataFrame:
        return _doc_term_counts(self.docs, self.text_col, self.tokenizer, min_count)

    def vocab(self, min_count: int = 1) -> pd.Series:
        counts = self.doc_term_counts(min_count=min_count).sum(axis=0)
        return counts.rename("count").sort_values(ascending=False)

    def total_tokens(self) -> int:
        return int(self.doc_term_counts(min_count=1).values.sum())

    def slice(self, **filters: Any) -> CorpusSlice:
        """Further filter this slice — produces a new CorpusSlice on the
        same parent with the masks AND-ed and the filters merged.

        Lets you chain: ``corpus.slice(topic="x").slice(party="y")``.
        """
        new_mask = self.mask.copy()
        merged_filters = dict(self.filters)
        for col, value in filters.items():
            if col not in self.parent.docs.columns:
                raise KeyError(f"slice() got unknown column {col!r}")
            if isinstance(value, (list, tuple, set, pd.Series)):
                new_mask &= self.parent.docs[col].isin(list(value))
            else:
                new_mask &= self.parent.docs[col] == value
            merged_filters[col] = value
        return CorpusSlice(parent=self.parent, mask=new_mask, filters=merged_filters)

    def by_time(self, col: str, freq: str = "Y") -> TemporalCorpus:
        """Return a TemporalCorpus over the slice's documents only.

        Materialises the slice's masked rows into a fresh Corpus, then
        delegates to :meth:`Corpus.by_time`. Lets you chain
        ``corpus.slice(topic="x").by_time("date")``.
        """
        from .temporal.slicing import TemporalCorpus  # local import to break cycle

        fresh = Corpus(
            docs=self.docs.reset_index(drop=True),
            text_col=self.text_col,
            id_col=self.id_col,
            meta_cols=self.parent.meta_cols,
            tokenizer=self.tokenizer,
        )
        return TemporalCorpus(parent=fresh, time_col=col, freq=freq)

    def to_polars(self) -> pl.DataFrame:
        """Return the slice's documents as a polars DataFrame."""
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_polars() requires polars. Install with: pip install 'pycorpdiff[polars]'"
            ) from exc
        return pl.from_pandas(self.docs.reset_index(drop=True))
