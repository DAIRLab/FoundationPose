"""Discrete rotational-symmetry transforms for FoundationPose.

The cone mesh (``demo_data/cone.obj``) is a regular hexagonal pyramid, so it has
N-fold discrete symmetry (N = 6) about its body axis. Telling FoundationPose
about this via ``symmetry_tfs`` collapses the equivalent orientation hypotheses
that otherwise cause the large single-frame orientation flips and the
about-axis jitter seen in the logs.
"""

import numpy as np


def make_symmetry_tfs(symmetry_count: int, axis=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Return an ``(N, 4, 4)`` stack of rotations by ``2*pi*k/N`` about ``axis``.

    ``k = 0`` gives identity. The mesh FoundationPose uses is centered on its
    own centroid; the cone's symmetry axis passes through that centroid, so a
    pure rotation (zero translation) is the correct symmetry transform.
    """
    if symmetry_count < 1:
        raise ValueError(f'symmetry_count must be >= 1, got {symmetry_count}')
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    tfs = np.zeros((symmetry_count, 4, 4), dtype=np.float32)
    for k in range(symmetry_count):
        angle = 2.0 * np.pi * k / symmetry_count
        tfs[k] = np.eye(4, dtype=np.float32)
        tfs[k][:3, :3] = _axis_angle(axis, angle)
    return tfs


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float32)
