"""Decide, each frame, whether to re-register instead of tracking.

Hard rule (never violated): re-registration only happens while the end effector
is repositioning AND the object estimate has been still for a short window --
running ``register`` while the object could be moving would anchor to a smear.

Health checks (plate penetration, one-frame orientation jump, sustained low
confidence) never bypass those hard gates; they only let a re-registration fire
*before* ``min_interval_s`` has elapsed once the gates are open. Each health
check has its own on/off flag.

When a health check trips but the gates are closed, the caller is told to hold
the last good pose instead (``hold_last_good_pose_on_bad_health``).
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ReregDecision:
    reregister: bool
    reason: str
    gates_open: bool
    health_bad: bool
    hold_last_good_pose: bool


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a[:3, :3].T @ R_b[:3, :3]
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class ReregistrationController:
    def __init__(self, params):
        r = params.reregistration
        self.enabled = bool(r['enabled'])
        self.require_repositioning = bool(r['require_repositioning'])
        self.still_window_s = float(r['still_window_s'])
        self.still_trans_m = float(r['still_trans_m'])
        self.still_rot_deg = float(r['still_rot_deg'])
        self.min_interval_s = float(r['min_interval_s'])
        self.hold_last_good_pose_on_bad_health = bool(r['hold_last_good_pose_on_bad_health'])

        hc = r['health_checks']
        self.hc_pen = bool(hc['penetration']['enabled'])
        self.pen_z_below = float(hc['penetration']['z_below_plate_m'])
        self.pen_frames = int(hc['penetration']['sustained_frames'])
        self.hc_jump = bool(hc['orientation_jump']['enabled'])
        self.jump_deg = float(hc['orientation_jump']['threshold_deg'])
        self.hc_lowconf = bool(hc['low_confidence']['enabled'])
        self.lowconf_min = float(hc['low_confidence']['min_confidence'])
        self.lowconf_frames = int(hc['low_confidence']['sustained_frames'])

        self._history = deque()          # (t, xyz_world, R_world)
        self._last_reregister_t = -1e9
        self._prev_pose_world = None
        self._pen_count = 0
        self._lowconf_count = 0

    # -- called once per frame with the freshly estimated pose ---------------
    def update(self, now: float, obj_pose_world: np.ndarray,
               confidence: float, mesh_min_z_world: float,
               is_repositioning: bool) -> ReregDecision:
        if not self.enabled:
            return ReregDecision(False, 'disabled', False, False, False)

        xyz = obj_pose_world[:3, 3].copy()
        R = obj_pose_world[:3, :3].copy()

        # --- rolling still-window history
        self._history.append((now, xyz, R))
        while self._history and now - self._history[0][0] > self.still_window_s:
            self._history.popleft()

        # --- health-check counters
        if mesh_min_z_world < -self.pen_z_below:
            self._pen_count += 1
        else:
            self._pen_count = 0
        if confidence is not None and np.isfinite(confidence) \
                and confidence < self.lowconf_min:
            self._lowconf_count += 1
        else:
            self._lowconf_count = 0

        one_frame_jump_deg = 0.0
        if self._prev_pose_world is not None:
            one_frame_jump_deg = _rotation_angle_deg(self._prev_pose_world, obj_pose_world)
        self._prev_pose_world = obj_pose_world.copy()

        # --- hard gates
        still = self._is_still()
        repositioning_ok = (not self.require_repositioning) or is_repositioning
        gates_open = still and repositioning_ok

        # --- health state
        health_reasons = []
        if self.hc_pen and self._pen_count >= self.pen_frames:
            health_reasons.append(f'penetration({self._pen_count}f)')
        if self.hc_jump and one_frame_jump_deg > self.jump_deg:
            health_reasons.append(f'orientation_jump({one_frame_jump_deg:.0f}deg)')
        if self.hc_lowconf and self._lowconf_count >= self.lowconf_frames:
            health_reasons.append(f'low_confidence({self._lowconf_count}f)')
        health_bad = bool(health_reasons)

        elapsed = now - self._last_reregister_t
        timing_ok = elapsed >= self.min_interval_s

        if gates_open and (timing_ok or health_bad):
            reason = ('interval' if timing_ok else 'health:' + ','.join(health_reasons))
            return ReregDecision(True, reason, True, health_bad, False)

        hold = health_bad and not gates_open and self.hold_last_good_pose_on_bad_health
        if not gates_open:
            why = []
            if not still:
                why.append('object_moving')
            if not repositioning_ok:
                why.append('c3_mode')
            reason = 'gates_closed:' + ','.join(why)
        else:
            reason = f'waiting_interval({elapsed:.1f}/{self.min_interval_s:.1f}s)'
        if health_bad:
            # Surface *why* health looks bad even though we can't act on it yet
            # (gates closed) -- otherwise a stuck "hold" is unexplainable from
            # the logs.
            reason += ' health:' + ','.join(health_reasons)
        return ReregDecision(False, reason, gates_open, health_bad, hold)

    def note_reregistered(self, now: float):
        self._last_reregister_t = now
        self._pen_count = 0
        self._lowconf_count = 0

    # ---------------------------------------------------------------------
    def _is_still(self) -> bool:
        if len(self._history) < 2:
            return False
        # The window must actually span most of still_window_s, otherwise "still"
        # is trivially true right after a history reset.
        if self._history[-1][0] - self._history[0][0] < 0.5 * self.still_window_s:
            return False
        _, xyz0, R0 = self._history[0]
        max_trans = max(np.linalg.norm(x - xyz0) for _, x, _ in self._history)
        max_rot = max(_rotation_angle_deg(R0, R) for _, _, R in self._history)
        return max_trans <= self.still_trans_m and max_rot <= self.still_rot_deg
