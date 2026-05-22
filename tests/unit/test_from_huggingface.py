"""Tests for ``pycorpdiff.from_huggingface``.

All `datasets.load_dataset` calls are mocked via the `_loader` hook on
:func:`from_huggingface`; no network access or `datasets` installation
is required to run these tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

import pycorpdiff as pcd


def _make_mock_dataset(df: pd.DataFrame) -> Any:
    """Mock a HuggingFace `Dataset` object with .to_pandas()."""
    ds = MagicMock()
    ds.to_pandas.return_value = df
    return ds


def _mock_loader(df: pd.DataFrame):
    """Returns a callable matching the load_dataset signature, returning df."""
    captured: dict[str, Any] = {}

    def loader(dataset_id: str, *, name: str | None = None, split: str = "train", **kw: Any) -> Any:
        captured["dataset_id"] = dataset_id
        captured["name"] = name
        captured["split"] = split
        captured["kw"] = kw
        return _make_mock_dataset(df)

    loader.captured = captured  # type: ignore[attr-defined]
    return loader


def test_from_huggingface_basic_load() -> None:
    df = pd.DataFrame(
        {
            "text": ["one document", "two document", "three document"],
            "label": [1, 0, 1],
        }
    )
    corpus = pcd.from_huggingface(
        "stanfordnlp/imdb",
        text_col="text",
        meta_cols=("label",),
        _loader=_mock_loader(df),
    )
    assert isinstance(corpus, pcd.Corpus)
    assert len(corpus) == 3
    assert corpus.docs["label"].tolist() == [1, 0, 1]


def test_from_huggingface_passes_arguments_to_loader() -> None:
    df = pd.DataFrame({"text": ["hello"]})
    loader = _mock_loader(df)
    pcd.from_huggingface(
        "wikitext",
        config_name="wikitext-103-v1",
        split="validation",
        text_col="text",
        _loader=loader,
    )
    assert loader.captured["dataset_id"] == "wikitext"  # type: ignore[attr-defined]
    assert loader.captured["name"] == "wikitext-103-v1"  # type: ignore[attr-defined]
    assert loader.captured["split"] == "validation"  # type: ignore[attr-defined]


def test_from_huggingface_n_rows_appends_slice_to_split() -> None:
    df = pd.DataFrame({"text": ["a"] * 5})
    loader = _mock_loader(df)
    pcd.from_huggingface(
        "test/data", split="train", n_rows=100, _loader=loader,
    )
    assert loader.captured["split"] == "train[:100]"  # type: ignore[attr-defined]


def test_from_huggingface_columns_subsets() -> None:
    df = pd.DataFrame(
        {
            "text": ["one", "two"],
            "label": [1, 0],
            "language": ["en", "en"],
            "irrelevant": ["x", "y"],
        }
    )
    corpus = pcd.from_huggingface(
        "test/data", text_col="text", columns=["label"],
        _loader=_mock_loader(df),
    )
    # Should contain text + label, not language / irrelevant.
    assert "label" in corpus.docs.columns
    assert "irrelevant" not in corpus.docs.columns


def test_from_huggingface_missing_text_col_raises() -> None:
    df = pd.DataFrame({"body": ["hello"]})  # text col is missing
    with pytest.raises(ValueError, match="text_col"):
        pcd.from_huggingface(
            "test/data", text_col="text", _loader=_mock_loader(df),
        )


def test_from_huggingface_accepts_arrow_table_without_to_pandas() -> None:
    """Some `datasets` objects might be returned as plain DataFrames."""
    df = pd.DataFrame({"text": ["one", "two"]})
    corpus = pcd.from_huggingface(
        "test/data", _loader=lambda *a, **k: df,
    )
    assert len(corpus) == 2


def test_from_huggingface_drives_analytical_pipeline() -> None:
    """End-to-end: load → slice → keyness."""
    df = pd.DataFrame(
        {
            "text": [
                "the migrant worker arrived",
                "the migrant family settled",
                "the migrant criminal threat",
                "the migrant invasion grew",
            ],
            "frame": ["h", "h", "c", "c"],
        }
    )
    corpus = pcd.from_huggingface(
        "test/data",
        meta_cols=("frame",),
        _loader=_mock_loader(df),
    )
    result = pcd.compare(
        corpus.slice(frame="h"), corpus.slice(frame="c")
    ).keyness(min_count=1)
    assert isinstance(result, pcd.KeynessResult)


def test_from_huggingface_forwards_extra_kwargs() -> None:
    """Any kwargs not named in the signature pass through to load_dataset."""
    df = pd.DataFrame({"text": ["x"]})
    loader = _mock_loader(df)
    pcd.from_huggingface(
        "test/data",
        _loader=loader,
        revision="abc123",
        trust_remote_code=False,
    )
    assert loader.captured["kw"].get("revision") == "abc123"  # type: ignore[attr-defined]
    assert loader.captured["kw"].get("trust_remote_code") is False  # type: ignore[attr-defined]
