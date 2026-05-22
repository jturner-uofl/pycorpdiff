"""HuggingFace Datasets loader.

The dominant modern source of public text corpora — Hansard mirrors,
news datasets, social-media archives, academic corpora — is the
HuggingFace `datasets` hub. This module wraps `datasets.load_dataset`
in a thin shim that converts the result to a :class:`pycorpdiff.Corpus`.

``datasets`` is heavy (pulls ``pyarrow``, ``fsspec``, ``requests``,
``aiohttp``), so it lives in the optional ``huggingface`` extra:

    pip install 'pycorpdiff[huggingface]'

Then::

    corpus = pcd.from_huggingface(
        "stanfordnlp/imdb", split="train",
        text_col="text", meta_cols=("label",),
    )
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from ..corpus import Corpus
from ..io.readers import from_dataframe
from ..tokenize import Tokenizer


def from_huggingface(
    dataset_id: str,
    *,
    split: str = "train",
    text_col: str = "text",
    id_col: str | None = None,
    meta_cols: tuple[str, ...] = (),
    tokenizer: Tokenizer | None = None,
    config_name: str | None = None,
    columns: list[str] | None = None,
    n_rows: int | None = None,
    _loader: Callable[..., Any] | None = None,
    **load_dataset_kwargs: Any,
) -> Corpus:
    """Load a HuggingFace dataset and wrap it as a :class:`Corpus`.

    Parameters
    ----------
    dataset_id
        The hub identifier — e.g. ``"stanfordnlp/imdb"``,
        ``"openwebtext"``, ``"wikitext"``, or any private repo path
        you have access to.
    split
        Which split to materialise — ``"train"`` (default),
        ``"test"``, ``"validation"``, or a slice expression like
        ``"train[:1000]"``.
    text_col
        Name of the column carrying document text in the dataset.
    id_col
        Optional unique-document-id column.
    meta_cols
        Tuple of metadata column names to surface for slicing. If
        empty, every non-text column becomes metadata (matching the
        :func:`from_dataframe` default).
    tokenizer
        Optional :class:`Tokenizer`.
    config_name
        HuggingFace's "name" parameter for multi-config datasets
        (e.g. ``"wikitext-103-v1"`` for the ``wikitext`` dataset).
    columns
        Restrict materialisation to a subset of columns — useful when
        the dataset has many fields you don't need.
    n_rows
        Materialise only the first ``n_rows`` documents. Equivalent
        to passing ``split=f"{split}[:{n_rows}]"``; the explicit
        parameter is just more discoverable.
    _loader
        Internal hook for unit tests; substitutes
        :func:`datasets.load_dataset`.
    **load_dataset_kwargs
        Anything else gets forwarded to ``datasets.load_dataset``.

    Examples
    --------
    >>> import pycorpdiff as pcd
    >>> corpus = pcd.from_huggingface(                                  # doctest: +SKIP
    ...     "stanfordnlp/imdb", split="train[:1000]",
    ...     text_col="text", meta_cols=("label",),
    ... )
    >>> pos = corpus.slice(label=1); neg = corpus.slice(label=0)        # doctest: +SKIP
    >>> pcd.compare(pos, neg).keyness().plot()                          # doctest: +SKIP
    """
    loader = _loader
    if loader is None:
        try:
            from datasets import load_dataset as _hf_load  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "from_huggingface requires the `datasets` library. "
                "Install with: pip install 'pycorpdiff[huggingface]'"
            ) from exc
        loader = _hf_load

    effective_split = split if n_rows is None else f"{split}[:{int(n_rows)}]"

    ds = loader(
        dataset_id,
        name=config_name,
        split=effective_split,
        **load_dataset_kwargs,
    )

    # The datasets library exposes Arrow Tables; the canonical conversion
    # to pandas is .to_pandas(), which most dataset objects implement.
    if hasattr(ds, "to_pandas"):
        df = ds.to_pandas()
    elif isinstance(ds, pd.DataFrame):
        df = ds
    else:
        # Last-resort: iterate as dicts.
        df = pd.DataFrame(list(ds))

    if columns is not None:
        # Keep just the requested columns (plus text_col if not listed).
        keep = list(dict.fromkeys([text_col, *columns]))
        df = df[[c for c in keep if c in df.columns]]

    if text_col not in df.columns:
        raise ValueError(
            f"text_col={text_col!r} not found in dataset columns "
            f"{list(df.columns)!r}"
        )

    return from_dataframe(
        df,
        text_col=text_col,
        id_col=id_col,
        meta_cols=meta_cols,
        tokenizer=tokenizer,
    )
