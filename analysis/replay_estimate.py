"""Offline replay of the FoundationPose cone estimator on a dumped RGB-D
sequence (``run_live_demo.py --dump_dir``).

Lets you A/B the three tracking-side changes deterministically, on the same
frames, without the camera:

    python analysis/replay_estimate.py DUMP_DIR --tag baseline \
        --no-mask --no-symmetry --no-reregistration
    python analysis/replay_estimate.py DUMP_DIR --tag all-on

If ``extrinsics/color_tf_world.npy`` was not freshly calibrated for this dump
(no charuco board in view), pass ``--manual_mask`` on the first invocation for a
dump: instead of the WORLD_T_CONE-rendered frame-0 silhouette (which depends on
that extrinsics file and may not line up with where the object actually is),
you hand-click the outline once; it's cached to ``DUMP_DIR/manual_mask.png`` and
reused by every later ``--tag`` run on the same dump. Note this only fixes
*registration* -- the world-frame metrics below (tilt from vertical, z vs the
build plate) still read the pose through that same (possibly stale) extrinsics
file, so treat them cautiously; per-step rotation and jitter are computed in
the camera frame and are unaffected either way.

Each run writes ``DUMP_DIR/replay_<tag>/ob_in_cam/*.txt`` + ``metrics.json`` and
prints the same summary as ``analyze_object_state_log.py``.
"""

import argparse
import glob
import json
import os
import os.path as op
import sys

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

CODE_DIR = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from estimater import (FoundationPose, ScorePredictor, PoseRefinePredictor,
                       set_logging_format, set_seed)
import nvdiffrast.torch as dr

from state_estimation import (load_params, WorkspaceMasker, ConfidenceEstimator,
                              ReregistrationController, make_symmetry_tfs)
from state_estimation.render import render_silhouette, mesh_min_z_in_world

CONE_BODY_AXIS = np.array([1.0, 0.0, 0.0])


def pick_manual_mask(color_bgr, cache_path, redo=False):
    """Interactive polygon mask picker for frame 0 (same interaction as
    ``mask.py``): left-click to add points, any key to finish and fill.

    Bypasses the automatic WORLD_T_CONE-rendered mask, which depends on
    ``extrinsics/color_tf_world.npy`` -- useful when that calibration is stale
    (e.g. no charuco board was in view today) and the auto mask may not line up
    with where the object actually is. Caches to ``cache_path`` so repeated
    ``--tag`` runs on the same dump reuse one picked mask instead of re-asking.
    """
    if op.isfile(cache_path) and not redo:
        print(f'(reusing cached manual mask: {cache_path}; pass --redo_mask to redraw)')
        return cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)

    points = []
    display = color_bgr.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], (0, 255, 0), 1)
            cv2.imshow('pick object outline (click points, any key when done)', display)

    win = 'pick object outline (click points, any key when done)'
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    cv2.imshow(win, display)
    print('Click points around the object outline in the window, then press any key.')
    cv2.waitKey(0)
    cv2.destroyWindow(win)

    if len(points) < 3:
        raise RuntimeError(f'Only {len(points)} points clicked; need >= 3 for a polygon.')
    mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    cv2.imwrite(cache_path, mask)
    print(f'wrote {cache_path} ({int((mask > 0).sum())} px)')
    return mask


def _running_median_residual(sig, win=21):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(sig) < win:
        return sig - np.median(sig)
    pad = np.pad(sig, (win // 2, win // 2), mode='edge')
    med = np.median(sliding_window_view(pad, win), axis=1)[:len(sig)]
    return sig - med


def summarize(times, poses_world):
    t = np.asarray(times) - times[0]
    xyz = np.array([p[:3, 3] for p in poses_world])
    rot = R.from_matrix([p[:3, :3] for p in poses_world])
    body_axis_world = rot.apply(CONE_BODY_AXIS)
    tilt = np.degrees(np.arccos(np.clip(body_axis_world[:, 2], -1, 1)))
    dstep = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())
    jitter = {ax: float(_running_median_residual(xyz[:, k]).std() * 1000)
              for k, ax in enumerate('xyz')}
    out = {
        'n_frames': len(t),
        'jitter_mm': jitter,
        'per_step_rot_deg_mean': float(dstep.mean()),
        'per_step_rot_deg_p95': float(np.percentile(dstep, 95)),
        'flips_gt5deg': int((dstep > 5).sum()),
        'flips_gt10deg': int((dstep > 10).sum()),
        'tilt_first10_deg': float(tilt[:10].mean()),
        'tilt_last10_deg': float(tilt[-10:].mean()),
        'tilt_corr_t': float(np.corrcoef(t, tilt)[0, 1]) if len(t) > 2 else 0.0,
        'z_below_plate_frames': int((xyz[:, 2] < 0).sum()),
        'z_below_5mm_frames': int((xyz[:, 2] < -0.005).sum()),
        'min_z_mm': float(xyz[:, 2].min() * 1000),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump_dir')
    ap.add_argument('--tag', default='replay')
    ap.add_argument('--config', default=None)
    ap.add_argument('--no-mask', action='store_true')
    ap.add_argument('--no-symmetry', action='store_true')
    ap.add_argument('--no-reregistration', action='store_true')
    ap.add_argument('--manual_mask', action='store_true',
                    help='hand-click the frame-0 object outline instead of using '
                         'the WORLD_T_CONE-rendered silhouette (use this if '
                         'extrinsics/color_tf_world.npy was not freshly '
                         'calibrated for this session -- the auto mask may not '
                         'line up with where the object actually is)')
    ap.add_argument('--redo_mask', action='store_true',
                    help='redraw the manual mask even if a cached one exists')
    ap.add_argument('--est_refine_iter', type=int, default=5)
    ap.add_argument('--track_refine_iter', type=int, default=2)
    args = ap.parse_args()

    set_logging_format()
    set_seed(0)

    dump = args.dump_dir
    cfg_path = args.config or op.join(dump, 'state_estimation_params.yaml')
    params = load_params(cfg_path if op.isfile(cfg_path) else None)

    cam_K = np.load(op.join(dump, 'cam_K.npy'))
    world_to_cam = np.load(op.join(dump, 'world_to_cam.npy'))
    camera_T_cone = np.load(op.join(dump, 'camera_T_cone.npy'))
    manifest = [json.loads(l) for l in open(op.join(dump, 'frames.jsonl'))]
    color_files = sorted(glob.glob(op.join(dump, 'color', '*.png')))
    assert len(color_files) == len(manifest), (len(color_files), len(manifest))

    mesh = trimesh.load(op.join(CODE_DIR, 'demo_data', 'cone.obj'), force='mesh')

    sym = (None if args.no_symmetry
           else make_symmetry_tfs(params.get_path('object.symmetry_count'),
                                  params.get_path('object.symmetry_axis')))
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
        scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
        glctx=glctx, symmetry_tfs=sym,
        hardcoded_initial_rot_mat=camera_T_cone[:3, :3],
        debug=0, debug_dir='/tmp/replay_debug',
    )

    masker = WorkspaceMasker(params, mesh_diameter=est.diameter)
    conf_est = ConfidenceEstimator(params)
    rereg = ReregistrationController(params)
    if args.no_reregistration:
        rereg.enabled = False
    use_mask = not args.no_mask

    out_dir = op.join(dump, f'replay_{args.tag}')
    os.makedirs(op.join(out_dir, 'ob_in_cam'), exist_ok=True)

    H = int(params.get_path('camera.process_height'))
    W = int(params.get_path('camera.process_width'))

    manual_mask = None
    if args.manual_mask:
        first_bgr = cv2.imread(color_files[0])
        manual_mask = pick_manual_mask(
            first_bgr, op.join(dump, 'manual_mask.png'), redo=args.redo_mask)

    prev_pose = prev_sil = last_good = pending = None
    times, poses_world, confs, events = [], [], [], []

    for k, (cfile, rec) in enumerate(zip(color_files, manifest)):
        color = cv2.imread(cfile)[..., ::-1].copy()
        depth = (cv2.imread(op.join(dump, 'depth', op.basename(cfile)),
                            cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0)
        depth[(depth < 0.1) | ~np.isfinite(depth)] = 0
        observed_depth = depth.copy()
        now = rec.get('wall_time', k / 18.0)
        is_repos = rec.get('is_repositioning')
        if is_repos is None:
            is_repos = True
        event = 'track'

        if k == 0:
            if manual_mask is not None:
                mask = manual_mask
                event = 'register(manual_mask)'
            else:
                mask, _ = render_silhouette(est, camera_T_cone, cam_K, H, W)
                event = 'register(auto_mask)'
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask,
                                iteration=args.est_refine_iter)
        else:
            do_rereg = pending is not None and pending.reregister
            hold = (pending is not None and pending.hold_last_good_pose
                    and last_good is not None)
            if do_rereg:
                reg_mask, _ = render_silhouette(est, prev_pose, cam_K, H, W)
                if int(reg_mask.sum()) >= 100:
                    pose = est.register(K=cam_K, rgb=color, depth=depth,
                                        ob_mask=reg_mask,
                                        iteration=args.est_refine_iter)
                    rereg.note_reregistered(now)
                    event = 'reregister:' + pending.reason
                else:
                    pose = est.track_one(rgb=color, depth=depth, K=cam_K,
                                         iteration=args.track_refine_iter)
            elif hold:
                pose = last_good.copy()
                T_c = est.get_tf_to_centered_mesh().detach().cpu().numpy()
                est.pose_last = torch.from_numpy(
                    (last_good @ np.linalg.inv(T_c)).astype(np.float32)).cuda()
                event = 'hold:' + pending.reason
            else:
                if use_mask:
                    md, mc, _ = masker.apply(
                        depth, color, cam_K, world_to_cam,
                        silhouette=prev_sil,
                        last_obj_xyz_world=((world_to_cam @ prev_pose)[:3, 3]
                                            if prev_pose is not None else None))
                else:
                    md, mc = depth, color
                pose = est.track_one(rgb=mc, depth=md, K=cam_K,
                                     iteration=args.track_refine_iter)

        cur_sil, cur_rd = render_silhouette(est, pose, cam_K, H, W)
        conf = conf_est.evaluate(cur_rd, observed_depth)
        obj_world = world_to_cam @ pose

        np.savetxt(op.join(out_dir, 'ob_in_cam', f'{k:06d}.txt'), pose)
        times.append(now)
        poses_world.append(obj_world)
        confs.append(conf.confidence)
        events.append(event)

        if k > 0:
            pending = rereg.update(
                now=now, obj_pose_world=obj_world, confidence=conf.confidence,
                mesh_min_z_world=mesh_min_z_in_world(mesh, obj_world),
                is_repositioning=is_repos)

        prev_pose = pose.copy()
        prev_sil = cur_sil
        if conf.confidence >= params.get_path(
                'reregistration.health_checks.low_confidence.min_confidence') \
                or last_good is None:
            last_good = pose.copy()

        if k % 50 == 0:
            print(f'  frame {k}/{len(manifest)}  {event}  conf {conf.confidence:.2f}')

    metrics = summarize(times, poses_world)
    metrics['tag'] = args.tag
    metrics['flags'] = {'mask': use_mask, 'symmetry': not args.no_symmetry,
                        'reregistration': rereg.enabled}
    metrics['n_reregister'] = sum(1 for e in events if e.startswith('reregister'))
    metrics['n_hold'] = sum(1 for e in events if e.startswith('hold'))
    with open(op.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n=== replay [{args.tag}]  mask={use_mask} '
          f'symmetry={not args.no_symmetry} rereg={rereg.enabled} ===')
    print(json.dumps(metrics, indent=2))
    print(f'\nwrote {out_dir}/')


if __name__ == '__main__':
    main()
