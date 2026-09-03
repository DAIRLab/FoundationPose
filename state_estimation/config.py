"""Load and lightly validate ``config/state_estimation_params.yaml``."""

import os.path as op

import yaml

CODE_DIR = op.dirname(op.dirname(op.abspath(__file__)))
DEFAULT_CONFIG_PATH = op.join(CODE_DIR, 'config', 'state_estimation_params.yaml')

_MISSING = object()


class Params(dict):
    """Dict with attribute access and a ``.get_path`` for nested keys.

    ``p.reregistration['enabled']`` and ``p['reregistration']['enabled']`` both
    work; ``p.get_path('reregistration.health_checks.penetration.enabled')``
    walks the tree and raises a clear error if a key is missing.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get_path(self, dotted, default=_MISSING):
        node = self
        for key in dotted.split('.'):
            if not isinstance(node, dict) or key not in node:
                if default is _MISSING:
                    raise KeyError(
                        f'Missing config key {dotted!r} '
                        f'(stopped at {key!r})'
                    )
                return default
            node = node[key]
        return node


def _wrap(obj):
    if isinstance(obj, dict):
        return Params({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_params(path: str = None) -> Params:
    path = path or DEFAULT_CONFIG_PATH
    if not op.isfile(path):
        raise FileNotFoundError(f'State-estimation config not found: {path}')
    with open(path) as f:
        raw = yaml.safe_load(f)
    params = _wrap(raw)

    # Fail fast on the handful of values that would otherwise blow up much
    # later, mid-run.
    sym = params.get_path('object.symmetry_count')
    if not isinstance(sym, int) or sym < 1:
        raise ValueError(f'object.symmetry_count must be a positive int, got {sym!r}')
    for axis_key in ('mask.workspace_box_world.x',
                     'mask.workspace_box_world.y',
                     'mask.workspace_box_world.z'):
        lo, hi = params.get_path(axis_key)
        if not lo < hi:
            raise ValueError(f'{axis_key} must be [low, high] with low < high, got {[lo, hi]}')
    return params
