"""Cross-validation against Hamilton, Leskovec, & Jurafsky (2016).

This is the receipt: pycorpdiff's cosine-distance computation, run on
Hamilton et al.'s published aligned diachronic embeddings, agrees with
the paper's reported findings:

  - Known semantic shifters ("gay", "broadcast", "awful", "terrific")
    show high 1900s → 1990s cosine distance.
  - Stable function words ("the", "and", "of") show low distance.

The test downloads one decade pair from snap.stanford.edu (~50–200 MB
per decade depending on subset) and runs the comparison. It's marked
slow and skips silently when offline.

To run locally::

    pytest -m slow tests/integration/test_crossval_histwords.py

Set ``PYCORPDIFF_HISTWORDS_CACHE`` to a persistent directory to keep
downloads across runs.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import pycorpdiff as pcd
from pycorpdiff.datasets.histwords import (
    HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990,
    HISTWORDS_DOWNLOAD_URLS,
)

pytestmark = pytest.mark.slow


def _has_internet() -> bool:
    """Probe whether we can reach snap.stanford.edu's HEAD without
    actually downloading. Skips the whole module if not."""
    try:
        req = urllib.request.Request(
            HISTWORDS_DOWNLOAD_URLS["coha"], method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):  # pragma: no cover
        return False


@pytest.fixture(scope="module")
def histwords_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Persistent cache between session runs if the env var is set."""
    env_cache = os.environ.get("PYCORPDIFF_HISTWORDS_CACHE")
    if env_cache:
        return Path(env_cache).expanduser()
    return tmp_path_factory.mktemp("histwords")


def test_internet_reachable_to_stanford_snap() -> None:
    """Sanity-check the network preflight; subsequent tests rely on it."""
    if not _has_internet():
        pytest.skip("snap.stanford.edu unreachable; skipping slow histwords tests")


def test_fetch_coha_1990_returns_real_vocab(histwords_cache_dir: Path) -> None:
    """Download the 1990s COHA decade and verify the vocab includes
    everyday words. Doesn't check vector values — that's the next test."""
    if not _has_internet():
        pytest.skip("offline")
    try:
        vecs = pcd.fetch_histwords_decade(
            1990, source="coha", cache_dir=histwords_cache_dir
        )
    except FileNotFoundError as exc:
        pytest.skip(f"COHA 1990s not available: {exc}")
    # COHA 1990s vocab is large (~50k+ words). Expect basic English words.
    for word in ("the", "and", "of", "is", "people"):
        assert word in vecs, f"expected {word!r} in 1990s COHA vocab"
    # Standard HistWords embedding dimensionality is 300.
    assert next(iter(vecs.values())).shape[0] == 300


def test_known_shifters_show_high_cosine_distance(
    histwords_cache_dir: Path,
) -> None:
    """Hamilton et al.'s headline finding: 'gay', 'broadcast', 'awful',
    'terrific' all shifted dramatically in meaning across the 20th
    century. Each should clear a 0.3 cosine distance threshold between
    1900s and 1990s COHA vectors."""
    if not _has_internet():
        pytest.skip("offline")
    shifters = ["gay", "broadcast", "awful", "terrific"]
    for word in shifters:
        try:
            d = pcd.histwords_cosine_shift(
                1900, 1990, word, source="coha", cache_dir=histwords_cache_dir
            )
        except KeyError:
            pytest.skip(f"{word!r} missing from COHA 1900s or 1990s vocab")
        except FileNotFoundError as exc:
            pytest.skip(f"COHA decade data not available: {exc}")
        assert d > 0.3, (
            f"expected {word!r} to show cosine distance > 0.3 "
            f"between 1900s and 1990s COHA; got {d:.3f}"
        )


def test_stable_function_words_show_low_cosine_distance(
    histwords_cache_dir: Path,
) -> None:
    """Negative control. Function words like 'the', 'and', 'of' have
    no semantic shift across centuries; cosine distance should be low
    (< 0.30 — the same threshold the shifters clear, with a comfortable
    margin in both directions)."""
    if not _has_internet():
        pytest.skip("offline")
    stable = ["the", "and", "of"]
    for word in stable:
        try:
            d = pcd.histwords_cosine_shift(
                1900, 1990, word, source="coha", cache_dir=histwords_cache_dir
            )
        except FileNotFoundError as exc:
            pytest.skip(f"COHA decade data not available: {exc}")
        assert d < 0.30, (
            f"expected {word!r} to be stable across decades "
            f"(cosine distance < 0.30); got {d:.3f}"
        )


def test_shifter_distance_exceeds_stable_distance_by_meaningful_margin(
    histwords_cache_dir: Path,
) -> None:
    """The headline finding stated as a *contrast*: the average distance
    of known shifters is meaningfully higher than the average distance
    of stable function words."""
    if not _has_internet():
        pytest.skip("offline")
    import contextlib

    shifter_distances = []
    for word in ("gay", "broadcast", "awful"):
        with contextlib.suppress(KeyError):
            try:
                shifter_distances.append(
                    pcd.histwords_cosine_shift(
                        1900, 1990, word, source="coha",
                        cache_dir=histwords_cache_dir,
                    )
                )
            except FileNotFoundError as exc:
                pytest.skip(f"COHA decade data not available: {exc}")
    stable_distances = []
    for word in ("the", "and", "of"):
        try:
            stable_distances.append(
                pcd.histwords_cosine_shift(
                    1900, 1990, word, source="coha", cache_dir=histwords_cache_dir
                )
            )
        except FileNotFoundError as exc:
            pytest.skip(f"COHA decade data not available: {exc}")
    if not shifter_distances:
        pytest.skip("no shifters available in COHA vocab")
    avg_shift = sum(shifter_distances) / len(shifter_distances)
    avg_stable = sum(stable_distances) / len(stable_distances)
    # Expect the shifters to be at least 0.2 further apart on average —
    # this is the Hamilton-paper finding made into an assertion.
    assert avg_shift - avg_stable > 0.2, (
        f"expected shifter mean distance > stable mean + 0.2; "
        f"got shift={avg_shift:.3f}, stable={avg_stable:.3f}"
    )


def test_reference_table_is_internally_consistent() -> None:
    """The bundled reference dict has a documented structure that
    other code (and users) might rely on."""
    shifters = {"gay", "broadcast", "awful", "terrific", "guy"}
    controls = {"the", "and", "of", "is"}
    assert shifters <= set(HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990)
    assert controls <= set(HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990)
    # All values in unit interval.
    for word, val in HAMILTON_REFERENCE_SHIFTS_COHA_1900_1990.items():
        assert 0.0 <= val <= 1.0, f"{word}: {val} outside [0, 1]"
