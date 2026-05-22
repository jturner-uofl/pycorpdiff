"""Tests for the HistWords loader infrastructure.

The HTTP layer is mocked via the ``_fetch`` hook on
:func:`fetch_histwords_decade`, so these tests run fully offline. A
real-fetch slow-tier test against Stanford's snap.stanford.edu lives in
``tests/integration/test_crossval_histwords.py``.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

import pycorpdiff as pcd


def _make_fake_histwords_zip(
    zip_path: Path, decades: dict[int, dict[str, np.ndarray]]
) -> None:
    """Write a HistWords-style zip with one .pkl + .npy per decade.

    Each ``decades[YYYY]`` is a {word: vector} mapping; we serialise
    its keys as a pickle list and its values as a stacked ndarray.
    """
    import io

    with zipfile.ZipFile(zip_path, "w") as zf:
        for decade, vecs in decades.items():
            words = list(vecs.keys())
            matrix = np.stack([vecs[w] for w in words])
            # Pickle the vocab list.
            zf.writestr(f"fake/{decade}.pkl", pickle.dumps(words))
            # Write the npy buffer.
            buf = io.BytesIO()
            np.save(buf, matrix)
            zf.writestr(f"fake/{decade}.npy", buf.getvalue())


def _mock_fetch(decades: dict[int, dict[str, np.ndarray]]):
    """Build an `_fetch` callable that writes a fake HistWords zip."""

    def fetch(url: str, dest: Path) -> None:
        _make_fake_histwords_zip(dest, decades)

    return fetch


def test_fetch_histwords_decade_round_trips_vectors(tmp_path: Path) -> None:
    """Synthetic data fed through the zip layer round-trips intact."""
    vecs_1900 = {
        "gay": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "the": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    out = pcd.fetch_histwords_decade(
        1900,
        source="eng-all",
        cache_dir=tmp_path,
        _fetch=_mock_fetch({1900: vecs_1900}),
    )
    assert set(out.keys()) == {"gay", "the"}
    np.testing.assert_array_equal(out["gay"], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(out["the"], [0.0, 1.0, 0.0])


def test_fetch_histwords_decade_uses_cache(tmp_path: Path) -> None:
    """Subsequent calls with the same args don't re-fetch."""
    call_count = {"n": 0}
    vecs_1900 = {"gay": np.array([1.0, 0.0], dtype=np.float32)}

    def counting_fetch(url: str, dest: Path) -> None:
        call_count["n"] += 1
        _make_fake_histwords_zip(dest, {1900: vecs_1900})

    pcd.fetch_histwords_decade(
        1900, source="eng-all", cache_dir=tmp_path, _fetch=counting_fetch
    )
    pcd.fetch_histwords_decade(
        1900, source="eng-all", cache_dir=tmp_path, _fetch=counting_fetch
    )
    assert call_count["n"] == 1


def test_fetch_histwords_decade_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        pcd.fetch_histwords_decade(1900, source="bogus")


def test_fetch_histwords_decade_missing_decade_raises(tmp_path: Path) -> None:
    """If the archive doesn't contain the requested decade, raise."""
    only_1990 = {1990: {"gay": np.array([1.0, 0.0], dtype=np.float32)}}
    with pytest.raises(FileNotFoundError, match="decade 1900"):
        pcd.fetch_histwords_decade(
            1900, source="eng-all", cache_dir=tmp_path,
            _fetch=_mock_fetch(only_1990),
        )


def test_histwords_cosine_shift_orthogonal_vectors(tmp_path: Path) -> None:
    """Orthogonal target vectors → cosine distance = 1.0."""
    vecs = {
        1900: {"target": np.array([1.0, 0.0, 0.0], dtype=np.float32)},
        1990: {"target": np.array([0.0, 1.0, 0.0], dtype=np.float32)},
    }
    distance = pcd.histwords_cosine_shift(
        1900, 1990, "target", source="eng-all",
        cache_dir=tmp_path, _fetch=_mock_fetch(vecs),
    )
    assert abs(distance - 1.0) < 1e-6


def test_histwords_cosine_shift_identical_vectors(tmp_path: Path) -> None:
    """Identical target vectors → cosine distance = 0.0."""
    v = np.array([0.6, 0.8, 0.0], dtype=np.float32)
    vecs = {1900: {"target": v}, 1990: {"target": v}}
    distance = pcd.histwords_cosine_shift(
        1900, 1990, "target", source="eng-all",
        cache_dir=tmp_path, _fetch=_mock_fetch(vecs),
    )
    assert abs(distance) < 1e-6


def test_histwords_cosine_shift_target_missing_in_a_raises(tmp_path: Path) -> None:
    vecs = {
        1900: {"other": np.array([1.0, 0.0], dtype=np.float32)},
        1990: {"target": np.array([1.0, 0.0], dtype=np.float32)},
    }
    with pytest.raises(KeyError, match="not in eng-all 1900"):
        pcd.histwords_cosine_shift(
            1900, 1990, "target", source="eng-all",
            cache_dir=tmp_path, _fetch=_mock_fetch(vecs),
        )


def test_histwords_cosine_shift_target_missing_in_b_raises(tmp_path: Path) -> None:
    vecs = {
        1900: {"target": np.array([1.0, 0.0], dtype=np.float32)},
        1990: {"other": np.array([1.0, 0.0], dtype=np.float32)},
    }
    with pytest.raises(KeyError, match="not in eng-all 1990"):
        pcd.histwords_cosine_shift(
            1900, 1990, "target", source="eng-all",
            cache_dir=tmp_path, _fetch=_mock_fetch(vecs),
        )


def test_histwords_cosine_shift_recovers_known_value(tmp_path: Path) -> None:
    """Vectors at 60° apart → cosine similarity 0.5 → distance 0.5."""
    vecs = {
        1900: {"x": np.array([1.0, 0.0], dtype=np.float32)},
        1990: {
            "x": np.array(
                [np.cos(np.pi / 3), np.sin(np.pi / 3)], dtype=np.float32
            )
        },
    }
    distance = pcd.histwords_cosine_shift(
        1900, 1990, "x", source="eng-all",
        cache_dir=tmp_path, _fetch=_mock_fetch(vecs),
    )
    assert abs(distance - 0.5) < 1e-5


def test_fetch_histwords_supports_documented_sources() -> None:
    """The HISTWORDS_DOWNLOAD_URLS map covers Stanford's main subsets."""
    from pycorpdiff.datasets.histwords import HISTWORDS_DOWNLOAD_URLS

    expected = {"eng-all", "eng-fiction", "coha", "coha-lemma", "chi-sim"}
    assert expected <= set(HISTWORDS_DOWNLOAD_URLS)
    for url in HISTWORDS_DOWNLOAD_URLS.values():
        assert url.startswith("http")
        assert url.endswith(".zip")


def test_hamilton_reference_shifts_partition_into_shifters_and_controls() -> None:
    """Sanity-check the bundled reference values: known shifters > 0.3,
    stable controls < 0.2."""
    from pycorpdiff.datasets.histwords import HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990

    shifters = {"gay", "broadcast", "awful", "terrific", "guy"}
    controls = {"the", "and", "of", "is"}
    for term in shifters:
        assert HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990[term] > 0.3
    for term in controls:
        assert HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990[term] < 0.2
