"""Embedder protocol, a lazy SBERT default, and a deterministic test helper.

The :class:`Embedder` protocol is the package's plug point for vector
representations. Anything implementing
``encode(terms: Sequence[str]) -> np.ndarray`` of shape ``(n, d)``
satisfies it — a thin wrapper around gensim KeyedVectors, a HuggingFace
pipeline, your own trained vectors, etc.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Embedder(Protocol):
    """Anything callable that maps a sequence of strings to a 2-D vector array."""

    def encode(self, terms: Sequence[str]) -> npt.NDArray[np.float64]: ...


@dataclass
class SBERTEmbedder:
    """Default :class:`Embedder` backed by sentence-transformers.

    sentence-transformers is in the optional ``semantic`` extra and is
    imported lazily on first call to :meth:`encode` so the base
    install does not pull torch transitively. The default model
    ``all-MiniLM-L6-v2`` is ~22 MB; for non-English corpora prefer one
    of the multilingual options (e.g. ``paraphrase-multilingual-MiniLM-L12-v2``).
    """

    model_name: str = "all-MiniLM-L6-v2"
    # ``Any`` because sentence_transformers isn't a base dependency — the
    # real type is :class:`sentence_transformers.SentenceTransformer` but
    # we can't import it at module load time without breaking the
    # "base install stays light" contract.
    _model: Any = field(default=None, init=False, repr=False, compare=False)

    def encode(self, terms: Sequence[str]) -> npt.NDArray[np.float64]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SBERTEmbedder requires sentence-transformers. "
                    "Install with: pip install 'pycorpdiff[semantic]'"
                ) from exc
            object.__setattr__(self, "_model", SentenceTransformer(self.model_name))
        vectors = self._model.encode(list(terms), convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float64)


@dataclass(frozen=True)
class HashEmbedder:
    """Deterministic seed-derived embedder for testing and offline demos.

    Maps each input string to a vector by seeding a per-string RNG with
    a SHA-256 digest of the string. Same input always yields the same
    vector; different inputs yield uncorrelated unit vectors. This
    isn't useful for *semantic* analysis — there's no signal beyond the
    string equality — but it's perfect for verifying that the orchestrators
    (averaging, alignment, neighborhood drift) wire up correctly without
    paying the cost of a real model.
    """

    dim: int = 32

    def encode(self, terms: Sequence[str]) -> npt.NDArray[np.float64]:
        out = np.zeros((len(terms), self.dim), dtype=np.float64)
        for i, term in enumerate(terms):
            digest = hashlib.sha256(term.encode("utf-8")).digest()
            # Use the full 8-byte (64-bit) prefix as the seed. The
            # earlier 32-bit mask (``& 0xFFFFFFFF``) produced birthday
            # collisions at ~65k distinct terms — well within real
            # corpus vocabularies — and broke the "different inputs
            # yield uncorrelated vectors" promise. numpy's
            # ``default_rng`` accepts arbitrary-size integer seeds.
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed=seed)
            v = rng.standard_normal(self.dim)
            n = np.linalg.norm(v)
            out[i] = v / n if n > 0 else v
        return out
