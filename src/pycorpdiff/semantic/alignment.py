"""Vector-space alignment for diachronic embeddings.

Reference
---------
Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word
embeddings reveal statistical laws of semantic change. In *Proceedings
of ACL 2016*.
"""

from __future__ import annotations

import numpy as np


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return ``source`` rotated to best align with ``target``.

    Implements the orthogonal-Procrustes solution: ``R = U V^T`` where
    ``U Σ V^T = SVD(target.T @ source)``. The returned matrix is the
    rotated source, not the rotation operator itself.
    """
    raise NotImplementedError("procrustes_align() lands in Phase 6")
