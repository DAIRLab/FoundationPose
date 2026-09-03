"""Helpers for running FoundationPose as the SC3 cone-demo state estimator.

Everything here is driven by ``config/state_estimation_params.yaml`` and is kept
separate from ``run_live_demo.py`` so the individual pieces (intrinsics
validation, depth masking, symmetry, re-registration logic, confidence) can be
unit-tested and tuned without the camera in the loop.
"""

from .config import load_params, Params
from .intrinsics import (
    CameraIntrinsics,
    read_reference_intrinsics,
    write_reference_intrinsics,
    intrinsics_from_rs_profile,
    validate_against_reference,
    scale_K,
)
from .symmetry import make_symmetry_tfs
from .masking import WorkspaceMasker
from .confidence import ConfidenceEstimator
from .reregistration import ReregistrationController
from .lcm_inputs import ControllerModeListener
from . import realsense_setup

__all__ = [
    "load_params",
    "Params",
    "CameraIntrinsics",
    "read_reference_intrinsics",
    "write_reference_intrinsics",
    "intrinsics_from_rs_profile",
    "validate_against_reference",
    "scale_K",
    "make_symmetry_tfs",
    "WorkspaceMasker",
    "ConfidenceEstimator",
    "ReregistrationController",
    "ControllerModeListener",
    "realsense_setup",
]
