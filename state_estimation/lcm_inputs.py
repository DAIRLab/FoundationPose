"""Background LCM listener for the C3 controller mode.

Opportunistic re-registration is only allowed while the end effector is
repositioning (i.e. not in C3 contact mode). The sampling controller publishes a
flat ``is_c3_mode`` boolean on ``SAMPLING_C3_DEBUG`` (``lcmt_sampling_c3_debug``,
vendored under ``lcm_systems/lcm_types/dairlib``), which is far cheaper to decode
than the trajectory-wrapped ``IS_C3_MODE`` channel.

The listener runs its own LCM instance on a daemon thread so it never blocks the
estimation loop.
"""

import threading
import time

import lcm

from lcm_systems.lcm_types.dairlib import lcmt_sampling_c3_debug


class ControllerModeListener:
    def __init__(self, channel: str, stale_after_s: float):
        self._channel = channel
        self._stale_after_s = float(stale_after_s)
        self._lock = threading.Lock()
        self._is_c3_mode = None
        self._last_rx_wall = 0.0
        self._lc = lcm.LCM()
        self._lc.subscribe(self._channel, self._on_msg)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='ControllerModeListener')

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            # handle_timeout takes milliseconds; short timeout keeps shutdown snappy.
            try:
                self._lc.handle_timeout(200)
            except Exception:
                time.sleep(0.05)

    def _on_msg(self, channel, data):
        try:
            msg = lcmt_sampling_c3_debug.decode(data)
        except ValueError:
            return
        with self._lock:
            self._is_c3_mode = bool(msg.is_c3_mode)
            self._last_rx_wall = time.time()

    def is_repositioning(self) -> bool:
        """True only if we have a fresh message saying we are NOT in C3 mode."""
        with self._lock:
            fresh = (time.time() - self._last_rx_wall) <= self._stale_after_s
            mode = self._is_c3_mode
        return bool(fresh and mode is False)

    def status(self) -> str:
        with self._lock:
            age = time.time() - self._last_rx_wall
            mode = self._is_c3_mode
        if mode is None:
            return 'no SAMPLING_C3_DEBUG received yet'
        return f'is_c3_mode={mode} (age {age:.2f}s)'
