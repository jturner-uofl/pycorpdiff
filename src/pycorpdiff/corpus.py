"""Core ``Corpus`` and ``CorpusSlice`` data structures.

A :class:`Corpus` wraps a :class:`pandas.DataFrame` of documents plus
metadata. Slicing returns a :class:`CorpusSlice` that shares the parent's
configuration (text column, tokenizer, backend) but presents a
boolean-masked or filtered view. Both objects are immutable frozen
dataclasses; mutations produce new objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from .tokenize import RegexTokenizer, Tokenizer

if TYPE_CHECKING:
    from .temporal.slicing import TemporalCorpus


Backend = Literal["pandas", "polars"]


@dataclass(frozen=True)
class Corpus:
    """A corpus of documents with optional metadata columns.

    Parameters
    ----------
    docs
        A DataFrame whose rows are documents. Must contain at least the
        text column named by ``text_col``.
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
    backend
        ``"pandas"`` (default) or ``"polars"``. Only pandas is wired up
        in the scaffolding release.
    """

    docs: pd.DataFrame
    text_col: str = "text"
    id_col: str | None = None
    meta_cols: tuple[str, ...] = ()
    tokenizer: Tokenizer = field(default_factory=RegexTokenizer)
    backend: Backend = "pandas"

    def __post_init__(self) -> None:
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
        if self.backend not in ("pandas", "polars"):
            raise ValueError(f"backend must be 'pandas' or 'polars', got {self.backend!r}")
        if self.backend == "polars":
            # Polars backend is reserved for a later phase; surfacing it as
            # NotImplementedError now (rather than silently coercing) keeps
            # the contract honest.
            raise NotImplementedError("polars backend is not yet wired up")

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
