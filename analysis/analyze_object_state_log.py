"""Regression / diagnostics tool for the SC3 cone-demo state estimate.

Decodes ``OBJECT_STATE`` (+ ``OBJECT_STATE_CONFIDENCE`` and, if the dairlib LCM
types are importable, ``C3_ACTUAL``) from a hardware log and reports the metrics
that the FoundationPose changes are meant to move:

  * per-axis high-frequency position jitter (std of the estimate minus a short
    running median),
  * per-step orientation change + count of large single-frame flips,
  * cone body-axis tilt from vertical over time (drift),
  * count of frames whose pose penetrates the build plate (z < 0),
  * confidence vs. those error signals.

Usage:
    python analysis/analyze_object_state_log.py /path/to/hwlog-000003 [--out DIR]

Baseline for comparison: ~/3d_printer/logs/2026/08_30_26/000003/hwlog-000003
"""

import argparse
import os
import os.path as op
import sys

import numpy as np
from lcm import EventLog
from scipy.spatial.transform import Rotation as R

CODE_DIR = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from lcm_systems.lcm_types.lcm_pose import lcmt_object_state
from lcm_systems.lcm_types.drake import lcmt_drake_signal

# Cone body axis (CAD +x points through the apex). WORLD_T_CONE maps CAD +x onto
# world +z at the demo start, so "tilt" = angle between the estimated body +x in
# world and world +z.
CONE_BODY_AXIS = np.array([1.0, 0.0, 0.0])
PLATE_Z = 0.0


def _try_import_c3_state():
    try:
        import file_utils
        file_utils.add_dair_lcmtypes_to_path()
        import dairlib
        return dairlib.lcmt_c3_state
    except Exception as exc:  # dairlib not built / not on this machine
        print(f'(C3_ACTUAL decoding unavailable: {exc})')
        return None


def _running_median_residual(sig, win):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(sig) < win:
        return sig - np.median(sig)
    pad = np.pad(sig, (win // 2, win // 2), mode='edge')
    med = np.median(sliding_window_view(pad, win), axis=1)[:len(sig)]
    return sig - med


def load(path):
    c3_state_type = _try_import_c3_state()
    obj_t, obj_q, obj_xyz = [], [], []
    conf_t, conf = [], []
    ee_t, ee_xyz = [], []
    log = EventLog(path, 'r')
    for e in log:
        if e.channel == 'OBJECT_STATE':
            m = lcmt_object_state.decode(e.data)
            obj_t.append(e.timestamp / 1e6)
            obj_q.append([m.position[1], m.position[2], m.position[3], m.position[0]])
            obj_xyz.append([m.position[4], m.position[5], m.position[6]])
        elif e.channel == 'OBJECT_STATE_CONFIDENCE':
            m = lcmt_drake_signal.decode(e.data)
            d = dict(zip(m.coord, m.val))
            conf_t.append(e.timestamp / 1e6)
            conf.append(d.get('confidence', np.nan))
        elif e.channel == 'C3_ACTUAL' and c3_state_type is not None:
            m = c3_state_type.decode(e.data)
            s = dict(zip(m.state_names, m.state))
            if 'end_effector_x' in s:
                ee_t.append(e.timestamp / 1e6)
                ee_xyz.append([s['end_effector_x'], s['end_effector_y'], s['end_effector_z']])
    return {
        'obj_t': np.array(obj_t),
        'obj_q': np.array(obj_q),
        'obj_xyz': np.array(obj_xyz),
        'conf_t': np.array(conf_t),
        'conf': np.array(conf),
        'ee_t': np.array(ee_t),
        'ee_xyz': np.array(ee_xyz),
    }


def analyze(d):
    t = d['obj_t'] - d['obj_t'][0]
    xyz = d['obj_xyz']
    q = d['obj_q'] / np.linalg.norm(d['obj_q'], axis=1, keepdims=True)
    rot = R.from_quat(q)

    rate = len(t) / (t[-1] - t[0])

    jitter = {ax: float(_running_median_residual(xyz[:, k], 21).std() * 1000)
              for k, ax in enumerate('xyz')}

    body_axis_world = rot.apply(CONE_BODY_AXIS)
    tilt_deg = np.degrees(np.arccos(np.clip(body_axis_world[:, 2], -1, 1)))

    dstep = np.degrees((rot[:-1].inv() * rot[1:]).magnitude())

    below = xyz[:, 2] < PLATE_Z
    below_5mm = xyz[:, 2] < PLATE_Z - 0.005

    print(f'\n=== {len(t)} OBJECT_STATE msgs, {t[-1]-t[0]:.1f} s, {rate:.1f} Hz ===')
    print(f'high-freq position jitter (mm):  x {jitter["x"]:.2f}   '
          f'y {jitter["y"]:.2f}   z {jitter["z"]:.2f}')
    print(f'per-step rotation (deg):  mean {dstep.mean():.2f}   '
          f'p95 {np.percentile(dstep, 95):.2f}   max {dstep.max():.2f}')
    print(f'large single-frame flips:  >5deg {int((dstep > 5).sum())}   '
          f'>10deg {int((dstep > 10).sum())}')
    print(f'cone-axis tilt from vertical (deg):  first 60s {tilt_deg[t < 60].mean():.1f}   '
          f'last 60s {tilt_deg[t > t[-1]-60].mean():.1f}   '
          f'corr(t, tilt) {np.corrcoef(t, tilt_deg)[0, 1]:.2f}')
    print(f'plate penetration:  z<0 {int(below.sum())} frames   '
          f'z<-5mm {int(below_5mm.sum())} frames   min z {xyz[:, 2].min()*1000:.1f} mm')
    if len(d['conf']):
        c = d['conf']
        print(f'confidence:  n {len(c)}   mean {np.nanmean(c):.2f}   '
              f'frac<0.3 {np.mean(c < 0.3):.2f}')

    return {'t': t, 'xyz': xyz, 'tilt_deg': tilt_deg, 'dstep': dstep,
            'jitter': jitter, 'rate': rate}


def plot(d, a, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    t, xyz, tilt = a['t'], a['xyz'], a['tilt_deg']

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for k, name in enumerate('xyz'):
        ax[0].plot(t, xyz[:, k] * 1000, label=name, lw=0.8)
    ax[0].set_ylabel('position (mm)')
    ax[0].legend(loc='upper right')
    ax[0].set_title('object position (world)')

    ax[1].plot(t, xyz[:, 2] * 1000, lw=0.8)
    ax[1].axhline(0, color='r', lw=1, ls='--', label='build plate')
    ax[1].set_ylabel('z (mm)')
    ax[1].legend(loc='upper right')
    ax[1].set_title('vertical position vs build plate')

    ax[2].plot(t[:-1], a['dstep'], lw=0.6)
    ax[2].axhline(5, color='r', lw=1, ls='--')
    ax[2].set_ylabel('per-step rot (deg)')
    ax[2].set_title('frame-to-frame orientation change')

    ax[3].plot(t, tilt, lw=0.8)
    ax[3].set_ylabel('cone-axis tilt (deg)')
    ax[3].set_xlabel('time (s)')
    ax[3].set_title('cone body-axis tilt from vertical (drift)')

    if len(d['conf']):
        ct = d['conf_t'] - d['obj_t'][0]
        axc = ax[3].twinx()
        axc.plot(ct, d['conf'], color='tab:green', lw=0.7, alpha=0.6)
        axc.set_ylabel('confidence', color='tab:green')
        axc.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    path = op.join(out_dir, 'object_state_analysis.png')
    fig.savefig(path, dpi=130)
    print(f'\nwrote {path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--out', default=None,
                    help='directory for plots (default: alongside the log)')
    args = ap.parse_args()

    data = load(args.log)
    if len(data['obj_t']) < 10:
        raise SystemExit('No / too few OBJECT_STATE messages in the log')
    stats = analyze(data)
    out_dir = args.out or op.join(op.dirname(op.abspath(args.log)),
                                  'object_state_analysis')
    plot(data, stats, out_dir)
