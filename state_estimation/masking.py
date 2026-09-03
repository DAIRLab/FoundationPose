"""Per-frame workspace + self-occlusion depth mask.

``track_one`` consumes the whole depth frame, so depth samples on the end
effector, the ramp walls and the build plate get fed straight into the
point-to-plane pose refinement and pull the estimate off. This builds a keep
mask each frame from:

  * a world-frame axis-aligned workspace box (drops the plate, far background,
    most of the ramp),
  * an optional local XY disc around the last object estimate,
  * the object silhouette rendered from the last pose, dilated (keeps only depth
    plausibly on the object; rejects an end effector that crosses in front).

Everything outside the union is zeroed before the frame reaches FoundationPose.
"""

import cv2
import numpy as np


def _depth_to_cam_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = depth
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).astype(np.float32)  # (H, W, 3)


class WorkspaceMasker:
    def __init__(self, params, mesh_diameter: float):
        m = params.mask
        self.enabled = bool(m['enabled'])
        self.dilation_px = int(m['silhouette_dilation_px'])
        box = m['workspace_box_world']
        self.box_lo = np.array([box['x'][0], box['y'][0], box['z'][0]], dtype=np.float32)
        self.box_hi = np.array([box['x'][1], box['y'][1], box['z'][1]], dtype=np.float32)
        radius_diam = m.get('local_xy_radius_diameters', None)
        self.local_xy_radius = (
            None if radius_diam in (None, 0)
            else float(radius_diam) * float(mesh_diameter)
        )
        self.mask_rgb = bool(m['mask_rgb'])
        self._dilate_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.dilation_px + 1, 2 * self.dilation_px + 1))
            if self.dilation_px > 0 else None
        )

    def keep_mask(self, depth: np.ndarray, K: np.ndarray, X_wc: np.ndarray,
                  silhouette: np.ndarray = None,
                  last_obj_xyz_world: np.ndarray = None) -> np.ndarray:
        """Boolean (H, W) mask of depth pixels to keep."""
        valid = depth > 1e-4
        cam_pts = _depth_to_cam_points(depth, K).reshape(-1, 3)
        world_pts = (X_wc[:3, :3] @ cam_pts.T).T + X_wc[:3, 3]
        world_pts = world_pts.reshape(depth.shape[0], depth.shape[1], 3)

        inside_box = np.all((world_pts >= self.box_lo) & (world_pts <= self.box_hi),
                            axis=-1)
        keep = valid & inside_box

        if self.local_xy_radius is not None and last_obj_xyz_world is not None:
            dxy = world_pts[..., :2] - np.asarray(last_obj_xyz_world)[:2]
            keep &= (np.linalg.norm(dxy, axis=-1) <= self.local_xy_radius)

        if silhouette is not None:
            sil = silhouette.astype(np.uint8)
            if self._dilate_kernel is not None:
                sil = cv2.dilate(sil, self._dilate_kernel)
            keep &= sil.astype(bool)

        return keep

    def apply(self, depth: np.ndarray, color: np.ndarray, K: np.ndarray,
              X_wc: np.ndarray, silhouette: np.ndarray = None,
              last_obj_xyz_world: np.ndarray = None):
        """Return ``(masked_depth, masked_color, keep_mask)``.

        If masking is disabled, returns the inputs untouched with an all-True
        mask so callers do not need to special-case it.
        """
        if not self.enabled:
            return depth, color, np.ones(depth.shape[:2], dtype=bool)

        keep = self.keep_mask(depth, K, X_wc, silhouette, last_obj_xyz_world)
        masked_depth = np.where(keep, depth, 0.0).astype(depth.dtype)
        masked_color = color
        if self.mask_rgb and color is not None:
            masked_color = color.copy()
            masked_color[~keep] = 0
        return masked_depth, masked_color, keep
