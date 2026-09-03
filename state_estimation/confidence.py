"""Per-frame estimate-quality signal.

RealSense gives no covariance, so we use FoundationPose's own fit as a proxy:
how well the object rendered at the estimated pose agrees with the observed
depth inside the object silhouette, and how much of that silhouette actually has
valid observed depth (low coverage => heavy occlusion).

The scalar in [0, 1] feeds the ``low_confidence`` re-registration health check
and is published on ``OBJECT_STATE_CONFIDENCE`` for logging.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceResult:
    confidence: float
    mask_coverage: float
    depth_residual_mm: float

    def as_signal(self):
        """(names, values) for a drake ``lcmt_drake_signal``."""
        return (
            ["confidence", "mask_coverage", "depth_residual_mm"],
            [float(self.confidence), float(self.mask_coverage),
             float(self.depth_residual_mm)],
        )


def _lerp_clamped(x, x_good, x_bad):
    """1 at ``x_good``, 0 at ``x_bad``, linear between, clamped outside."""
    if x_good == x_bad:
        return 1.0 if x <= x_good else 0.0
    t = (x - x_good) / (x_bad - x_good)
    return float(np.clip(1.0 - t, 0.0, 1.0))


class ConfidenceEstimator:
    def __init__(self, params):
        c = params.confidence
        self.good_residual_mm = float(c['good_residual_mm'])
        self.bad_residual_mm = float(c['bad_residual_mm'])
        self.good_coverage = float(c['good_coverage'])
        self.bad_coverage = float(c['bad_coverage'])

    def evaluate(self, rendered_depth: np.ndarray,
                 observed_depth: np.ndarray) -> ConfidenceResult:
        """Both arrays are (H, W) in metres; 0 = no data."""
        silhouette = rendered_depth > 1e-4
        n_sil = int(silhouette.sum())
        if n_sil == 0:
            return ConfidenceResult(0.0, 0.0, float('nan'))

        obs_valid = silhouette & (observed_depth > 1e-4)
        coverage = obs_valid.sum() / n_sil

        if obs_valid.any():
            residual_mm = float(
                np.abs(rendered_depth[obs_valid] - observed_depth[obs_valid]).mean()
                * 1000.0
            )
        else:
            residual_mm = float('nan')

        residual_term = (
            0.0 if not np.isfinite(residual_mm)
            else _lerp_clamped(residual_mm, self.good_residual_mm, self.bad_residual_mm)
        )
        coverage_term = _lerp_clamped(coverage, self.good_coverage, self.bad_coverage)
        confidence = min(residual_term, coverage_term)
        return ConfidenceResult(confidence, float(coverage), residual_mm)
