"""Silhouette / depth rendering helpers used by the live demo and the offline
replay. Kept out of ``state_estimation/__init__`` so importing the package does
not pull in torch / nvdiffrast (``camera_calibration.py`` only wants the
intrinsics helpers).
"""

import numpy as np
import torch

from Utils import nvdiffrast_render


def render_silhouette(estimator, camera_T_object, K, height, width):
    """Render the object at ``camera_T_object`` (original CAD frame).

    Returns ``(mask_uint8, rendered_depth_m)``. FoundationPose centers the mesh
    internally by subtracting ``model_center``; compensate so ``camera_T_object``
    keeps referring to the original CAD frame.
    """
    original_T_centered = np.eye(4, dtype=np.float32)
    original_T_centered[:3, 3] = estimator.model_center
    camera_T_centered = camera_T_object @ original_T_centered

    pose_batch = torch.as_tensor(
        camera_T_centered[None], dtype=torch.float32, device='cuda')
    _, rendered_depth, _ = nvdiffrast_render(
        K=K, H=height, W=width, ob_in_cams=pose_batch,
        glctx=estimator.glctx, mesh_tensors=estimator.mesh_tensors,
        output_size=(height, width), get_normal=False, use_light=False,
    )
    rendered_depth = rendered_depth[0].detach().cpu().numpy()
    mask = (rendered_depth > 0).astype(np.uint8)
    return mask, rendered_depth


def mesh_min_z_in_world(mesh, obj_pose_in_world):
    """Lowest vertex of the posed mesh in the world frame (metres)."""
    v = np.asarray(mesh.vertices)
    world = (obj_pose_in_world[:3, :3] @ v.T).T + obj_pose_in_world[:3, 3]
    return float(world[:, 2].min())
