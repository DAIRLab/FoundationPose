"""Read / write / validate the camera intrinsics reference file.

``run_live_demo.py`` no longer hard-codes ``cam_K``. It reads the color-stream
intrinsics live from the RealSense pipeline and checks them against
``intrinsics/cam_K.txt`` (written by ``camera_calibration.py``) before it starts
estimating. Because depth is aligned into the color frame, the color intrinsics
are the single correct ``K`` for both the RGB and the aligned-depth inputs.
"""

import os.path as op
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int
    serial: str = ""
    product_line: str = ""
    distortion_model: str = ""
    distortion_coeffs: List[float] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.ppx],
                         [0.0, self.fy, self.ppy],
                         [0.0, 0.0, 1.0]])

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics for the same view resampled to a new resolution."""
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx, fy=self.fy * sy,
            ppx=self.ppx * sx, ppy=self.ppy * sy,
            width=width, height=height,
            serial=self.serial, product_line=self.product_line,
            distortion_model=self.distortion_model,
            distortion_coeffs=list(self.distortion_coeffs),
        )


def scale_K(K: np.ndarray, src_wh, dst_wh) -> np.ndarray:
    """Scale a 3x3 pinhole matrix from ``src_wh`` to ``dst_wh`` resolution."""
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    out = K.astype(float).copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def intrinsics_from_rs_profile(color_profile, device=None) -> CameraIntrinsics:
    """Build ``CameraIntrinsics`` from a pyrealsense2 color video-stream profile."""
    import pyrealsense2 as rs  # local import: only available in the camera env

    intr = color_profile.as_video_stream_profile().get_intrinsics()
    serial = ""
    product_line = ""
    if device is not None:
        try:
            serial = str(device.get_info(rs.camera_info.serial_number))
        except Exception:
            pass
        try:
            product_line = str(device.get_info(rs.camera_info.product_line))
        except Exception:
            pass
    return CameraIntrinsics(
        fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
        width=intr.width, height=intr.height,
        serial=serial, product_line=product_line,
        distortion_model=str(intr.model),
        distortion_coeffs=list(intr.coeffs),
    )


def write_reference_intrinsics(path: str, intr: CameraIntrinsics) -> None:
    K = intr.K
    lines = [
        f'serial: {intr.serial}',
        f'product_line: {intr.product_line}',
        f'width: {intr.width}',
        f'height: {intr.height}',
        f'distortion_model: {intr.distortion_model}',
        'distortion_coeffs: ' + ' '.join(f'{c:.10g}' for c in intr.distortion_coeffs),
        'K:',
    ]
    for row in K:
        lines.append(' '.join(f'{v:.10g}' for v in row))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def read_reference_intrinsics(path: str) -> CameraIntrinsics:
    if not op.isfile(path):
        raise FileNotFoundError(
            f'Intrinsics reference not found: {path}. '
            f'Run camera_calibration.py to create it.'
        )
    meta = {}
    K_rows = []
    reading_K = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == 'K:':
                reading_K = True
                continue
            if reading_K:
                K_rows.append([float(x) for x in line.split()])
            elif ':' in line:
                key, _, val = line.partition(':')
                meta[key.strip()] = val.strip()
    if len(K_rows) != 3:
        raise ValueError(f'Expected a 3x3 K in {path}, got {len(K_rows)} rows')
    K = np.array(K_rows)
    coeffs = []
    if meta.get('distortion_coeffs'):
        coeffs = [float(x) for x in meta['distortion_coeffs'].split()]
    return CameraIntrinsics(
        fx=K[0, 0], fy=K[1, 1], ppx=K[0, 2], ppy=K[1, 2],
        width=int(meta.get('width', 0)), height=int(meta.get('height', 0)),
        serial=meta.get('serial', ''),
        product_line=meta.get('product_line', ''),
        distortion_model=meta.get('distortion_model', ''),
        distortion_coeffs=coeffs,
    )


class IntrinsicsMismatch(RuntimeError):
    pass


def validate_against_reference(live: CameraIntrinsics,
                               reference: CameraIntrinsics,
                               focal_tol_px: float,
                               principal_tol_px: float,
                               require_serial_match: bool) -> None:
    """Raise ``IntrinsicsMismatch`` if the live camera disagrees with the file."""
    problems = []
    if (live.width, live.height) != (reference.width, reference.height):
        problems.append(
            f'resolution {live.width}x{live.height} != reference '
            f'{reference.width}x{reference.height}'
        )
    if require_serial_match and reference.serial and live.serial \
            and live.serial != reference.serial:
        problems.append(
            f'camera serial {live.serial!r} != reference {reference.serial!r} '
            f'(wrong camera connected?)'
        )
    if abs(live.fx - reference.fx) > focal_tol_px:
        problems.append(f'fx {live.fx:.2f} vs reference {reference.fx:.2f}')
    if abs(live.fy - reference.fy) > focal_tol_px:
        problems.append(f'fy {live.fy:.2f} vs reference {reference.fy:.2f}')
    if abs(live.ppx - reference.ppx) > principal_tol_px:
        problems.append(f'ppx {live.ppx:.2f} vs reference {reference.ppx:.2f}')
    if abs(live.ppy - reference.ppy) > principal_tol_px:
        problems.append(f'ppy {live.ppy:.2f} vs reference {reference.ppy:.2f}')
    if problems:
        raise IntrinsicsMismatch(
            'Live camera intrinsics differ from the calibration reference:\n  - '
            + '\n  - '.join(problems)
            + '\nRe-run camera_calibration.py, or check which camera is connected.'
        )
