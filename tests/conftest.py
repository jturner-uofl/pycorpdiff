"""Shared pytest fixtures for pycorpdiff tests."""

from __future__ import annotations

import pandas as pd
import pytest

import pycorpdiff as pcd


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """A tiny three-row, two-outlet, two-year fixture for plumbing tests."""
    return pd.DataFrame(
        {
            "text": [
                "the cat sat on the mat",
                "the dog sat on the log",
                "the cat chased the dog around the yard",
            ],
            "outlet": ["A", "B", "A"],
            "year": [2020, 2020, 2021],
        }
    )


@pytest.fixture
def toy_corpus(toy_df: pd.DataFrame) -> pcd.Corpus:
    return pcd.from_dataframe(toy_df, text_col="text", meta_cols=("outlet", "year"))
