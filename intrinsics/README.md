# Camera intrinsics

`run_live_demo.py` no longer hard-codes the camera matrix. It reads the color
intrinsics live from the RealSense pipeline and validates them against
`cam_K.txt` before it starts estimating.

## Files

- **`cam_K.txt`** — written by `camera_calibration.py`. The active reference:
  color-stream `K` plus the resolution, distortion coefficients, camera serial
  and product line it was captured at. Regenerate it whenever the camera or the
  capture resolution changes (`python camera_calibration.py`).
  Each calibration run also archives this file and `color_tf_world.npy` under
  `logs/calibration_<date>_<time>/`.
- **`cam_K_legacy.txt`**, **`old_cam_legacy.txt`** — the two matrices that used
  to be hard-coded in `run_live_demo.py` (`cam_K` at 640x480, and the older
  `old_cam`). Kept for reference only; not read by any code.

## Format of `cam_K.txt`

Plain text, `key: value` lines followed by the 3x3 matrix:

```
serial: 943222072000
product_line: D435I
width: 640
height: 480
distortion_model: inverse_brown_conrady
distortion_coeffs: 0.0 0.0 0.0 0.0 0.0
K:
604.0 0.0 326.8
0.0 603.4 253.5
0.0 0.0 1.0
```

Because depth is aligned into the color frame (`rs.align(rs.stream.color)`), the
aligned depth frame shares the color intrinsics, so this single `K` is correct
for both the RGB and depth inputs FoundationPose consumes.
