"""Vector-space alignment for diachronic embeddings.

Reference
---------
Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word
embeddings reveal statistical laws of semantic change. In *Proceedings
of ACL 2016*.

Schönemann, P. H. (1966). A generalized solution of the orthogonal
Procrustes problem. *Psychometrika*, 31(1), 1-10.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def procrustes_align(
    source: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return ``source`` rotated to best match ``target`` via orthogonal Procrustes.

    Implements Schönemann's closed-form solution: ``R = U V^T`` where
    ``U Σ V^T = SVD(source^T target)``. The returned matrix is the
    rotated source, not the rotation operator itself.

    Both matrices must have the same shape ``(n, d)``. Rotation
    preserves vector norms — if ``source`` is L2-normalised the
    output rows are still on the unit sphere.

    Use this when the two embedding spaces were trained independently
    (e.g. Hamilton-style diachronic word2vec). Modern shared-model
    encoders like SBERT already live in a common space, so alignment
    is unnecessary.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have the same shape; got {source.shape} vs {target.shape}"
        )
    m = source.T @ target
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    rotation = u @ vt
    rotated: npt.NDArray[np.float64] = source @ rotation
    return rotated
