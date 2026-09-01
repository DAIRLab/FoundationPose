"""Apply accuracy-oriented capture settings to a RealSense D435/D435i.

Kept in one place so ``run_live_demo.py`` just calls ``configure_streams`` /
``apply_device_options`` / ``build_post_processing`` with the loaded config.
All pyrealsense2 imports are local so the rest of the package stays importable
without the camera SDK.
"""

import logging
import time

VISUAL_PRESETS = {
    'Default': 'Default',
    'Hand': 'Hand',
    'High Accuracy': 'High Accuracy',
    'High Density': 'High Density',
    'Medium Density': 'Medium Density',
}


def configure_streams(config, cam_cfg):
    """Enable depth + color streams per the ``camera:`` config block."""
    import pyrealsense2 as rs

    config.enable_stream(rs.stream.depth,
                         int(cam_cfg['depth_width']), int(cam_cfg['depth_height']),
                         rs.format.z16, int(cam_cfg['depth_fps']))
    config.enable_stream(rs.stream.color,
                         int(cam_cfg['color_width']), int(cam_cfg['color_height']),
                         rs.format.rgb8, int(cam_cfg['color_fps']))


def apply_device_options(profile, cam_cfg):
    """Set the visual preset, emitter and laser power on the depth sensor."""
    import pyrealsense2 as rs

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    preset = (cam_cfg.get('visual_preset') or '').strip()
    if preset:
        applied = _apply_visual_preset(device, depth_sensor, preset)
        logging.info(f'realsense: visual_preset -> {applied}')

    if 'emitter_enabled' in cam_cfg and cam_cfg['emitter_enabled'] is not None:
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled,
                                    1.0 if cam_cfg['emitter_enabled'] else 0.0)

    laser = cam_cfg.get('laser_power')
    if laser is not None and depth_sensor.supports(rs.option.laser_power):
        rng = depth_sensor.get_option_range(rs.option.laser_power)
        depth_sensor.set_option(rs.option.laser_power,
                                float(min(max(laser, rng.min), rng.max)))
        logging.info(f'realsense: laser_power -> {depth_sensor.get_option(rs.option.laser_power)}')


def _apply_visual_preset(device, depth_sensor, preset_name):
    import pyrealsense2 as rs

    # Preferred path: advanced mode with a named preset.
    try:
        adv = rs.rs400_advanced_mode(device)
        if not adv.is_enabled():
            adv.toggle_advanced_mode(True)
            time.sleep(2.0)  # device re-enumerates
    except Exception as exc:  # not all builds expose advanced mode
        logging.warning(f'realsense: advanced mode unavailable ({exc})')

    if depth_sensor.supports(rs.option.visual_preset):
        rng = depth_sensor.get_option_range(rs.option.visual_preset)
        for i in range(int(rng.min), int(rng.max) + 1):
            try:
                name = depth_sensor.get_option_value_description(
                    rs.option.visual_preset, i)
            except Exception:
                continue
            if name.lower() == preset_name.lower():
                depth_sensor.set_option(rs.option.visual_preset, float(i))
                return name
    logging.warning(f'realsense: visual preset {preset_name!r} not found; '
                    f'leaving device default')
    return 'device default'


def build_post_processing(cam_cfg):
    """Return an ordered list of pyrealsense2 filters to apply to depth frames."""
    import pyrealsense2 as rs

    pp = cam_cfg.get('post_processing', {}) or {}
    filters = []
    # Depth->disparity, filter in disparity space, back to depth: recommended
    # order when spatial/temporal are used.
    use_disparity = pp.get('spatial') or pp.get('temporal')
    if use_disparity:
        filters.append(rs.disparity_transform(True))
    if pp.get('spatial'):
        filters.append(rs.spatial_filter())
    if pp.get('temporal'):
        filters.append(rs.temporal_filter())
    if use_disparity:
        filters.append(rs.disparity_transform(False))
    if pp.get('hole_filling'):
        filters.append(rs.hole_filling_filter())
    return filters


def run_post_processing(depth_frame, filters):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame
