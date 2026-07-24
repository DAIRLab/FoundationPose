# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Instructions for using a new camera calibration:

    1. Change the name of the latest extrinsics file to 'extrinsics_thru_'
       followed by the last day that the extrinsics were relevant.
    2. Put the new extrinsics file in the 'extrinsics' directory with the name
       'extrinsics_starting_' followed by the first day that the extrinsics are
       relevant (note:  these have to be different dates).
    3. Change the 'else' block to an 'elif' and add a new 'else' block to the
       'get_world_T_cam' function.
"""

import datetime
import pyrealsense2 as rs
from estimater import *
from datareader import *
from FoundationPose.mask import *
import argparse
from FoundationPose.lcm_systems.pose_publisher import PosePublisher
# imports for reading camera extrinsics
import yaml
import numpy as np
import os.path as op
from scipy.spatial.transform import Rotation as R
import time


WORLD_ROT_MAT_RB_AGAINST_ROBOT_PLATFORM = np.array([
    [-0.4122421 ,  0.81648502, -0.40423838],
    [ 0.70478414,  0.00462011, -0.70940678],
    [-0.57735238, -0.57734814, -0.57735029]])
WORLD_ROT_MAT_PUSH_T = np.eye(3)
WORLD_ROT_MAT_CUBE = np.eye(3)
WORLD_ROT_MAT_CONE = np.array([
    [ 0.70710678,  0,  -0.70710678        ],
    [ 0,    1, 0.        ],
    [ 0.70710678,  0.        ,  0.70710678,        ]])

DIST_CAM_TO_X_AXIS = 0.85
CAM_CAL_SWITCH_HYSTERESIS = 0.04


def get_extrinsic(filename):
  # Centralize the extrinsics path construction so the rest of the script
  # only deals with logical file names.
    print(f'get_extrinsic: resolving {filename} from extrinsics/')
    print(f'Loading {filename}')
    return op.join('extrinsics', filename)

def get_world_T_cam(dist_from_cam: float = None, was_near: bool = None):
  # This helper currently returns a fixed calibration, but it is the single
  # place where distance-aware camera calibration logic is meant to live.
    print(f'get_world_T_cam: dist_from_cam={dist_from_cam}, was_near={was_near}')
    is_near = None

    # Handle dates and changing extrinsics.
    today = datetime.date.today()
    print(f'get_world_T_cam: today={today.isoformat()}')

  # The active transform is loaded from the calibration file on disk and
  # inverted so downstream code can reason in world-to-camera coordinates.
    print('get_world_T_cam: loading active camera-to-world transform')
    cam_to_world = np.load(
        get_extrinsic('06_16_25_color_tf_world.npy'))
    world_to_cam = np.linalg.inv(cam_to_world)
    print('get_world_T_cam: transform loaded and inverted')

    return world_to_cam, is_near


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  # Resolve assets relative to this file so launch location does not matter.
  code_dir = os.path.dirname(os.path.realpath(__file__))
  print(f'run_live_demo: code_dir={code_dir}')
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/colored_jacktoy_data/mesh/jack_colored.obj')
  # parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/colored_jacktoy_data')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--system', type=str, default=None)
  parser.add_argument('--hardcode_quat', type=int, default=1)
  parser.add_argument('--lcm_publish', type=int, default=1)
  args = parser.parse_args()
  print(f'run_live_demo: parsed args={args}')

  # Global runtime setup for deterministic behavior and readable logs.
  print('run_live_demo: initializing logging and random seed')
  set_logging_format()
  set_seed(0)

  # Select the object model based on the requested demo system.
  mesh_file = f'{code_dir}/demo_data/colored_jacktoy_data/mesh/jack_colored.obj'
  if args.system == 'jack':
    print('run_live_demo: selected system jack')
    pass
  elif args.system == 't':
    print('run_live_demo: selected system t')
    mesh_file = f'{code_dir}/demo_data/push_t_data/mesh/push_t_bicolor.obj'
  elif args.system == 'cube':
    print('run_live_demo: selected system cube')
    mesh_file = f'{code_dir}/demo_data/cube_data/mesh/cube.obj'
  elif args.system == 'cone':
    print('run_live_demo: selected system cone')
    mesh_file = f'{code_dir}/demo_data/cone_data/cone.obj'
  elif args.system == None:
    raise ValueError('Need to specify system: "jack" or "t" or "cube" or "cone"')
  else:
    raise ValueError(f'Unknown system: {args.system} -- can only handle ' + \
                     f'"jack" or "t" or "cube" or "cone"')

  print("This is the mesh file: " + mesh_file)
  print('run_live_demo: loading mesh')
  mesh = trimesh.load(mesh_file, force='mesh')
  print("LOADED MESH FILE")

  debug = args.debug
  debug_dir = args.debug_dir
  print(f'run_live_demo: debug={debug}, debug_dir={debug_dir}')
  # Clear stale debug artifacts so each run writes a fresh trace of the
  # current tracking session.
  print('run_live_demo: clearing old debug artifacts')
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
    
  # Precompute geometry metadata used later for visualization and alignment.
  print('run_live_demo: computing mesh bounds')
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh,ordered=True)
  bbox = mesh.bounds
  print(f'run_live_demo: bbox={bbox}')

  # Get camera information.
  # Make sure to update this value according to the current intrinsics from the
  # camera. ros2 topic echo /camera/aligned_depth/camera_info from host machine.
  old_cam = np.array([[381.8276672363281, 0.0, 320.3140869140625],
                    [0.0, 381.4604187011719, 244.2602081298828],
                    [0.0, 0.0, 1.0]])
  cam_K = np.array([[604.05114746,   0,         326.85733032],
                    [  0,         603.39227295, 253.49771118],
                    [  0,           0,           1,        ]])
  print(f'run_live_demo: cam_K={cam_K.tolist()}')

  # Get camera extrinsics.
  print('run_live_demo: loading world-to-camera extrinsics')
  world_to_cam, is_near = get_world_T_cam(dist_from_cam=0)
  print(f'run_live_demo: initial is_near={is_near}')

  # Optional hard-coded orientation provides a stable first pose guess for
  # the target object when the scene is initialized by hand.
  hardcoded_initial_rot_mat = None
  if args.hardcode_quat != 0:
    if args.system == 'jack':
      print('run_live_demo: computing hardcoded initial rotation for jack')
      input('\nEnsure the blue and red capsules are touching the robot ' + \
            'platform, with the red contact further in the world y ' + \
            'direction.' + \
            '\nPress enter to continue.\nNote: A GUI window will' + \
            ' pop up to show the pose estimate.  Press \'q\' to close the ' + \
            'window and enable faster publishing without the GUI. ')
      hardcoded_initial_rot_mat = np.linalg.inv(world_to_cam[:3, :3]) @ \
        WORLD_ROT_MAT_RB_AGAINST_ROBOT_PLATFORM
    elif args.system == 't':
      print('run_live_demo: computing hardcoded initial rotation for t')
      input('\nEnsure the push T is flat on the table with the top of the ' + \
            'T up against the robot platform.' + \
            '\nPress enter to continue.\nNote: A GUI window will' + \
            ' pop up to show the pose estimate.  Press \'q\' to close the ' + \
            'window and enable faster publishing without the GUI. ')
      hardcoded_initial_rot_mat = np.linalg.inv(world_to_cam[:3, :3]) @ \
        WORLD_ROT_MAT_PUSH_T
    elif args.system == 'cone':
      print('run_live_demo: computing hardcoded initial rotation for cone')
      input('\nEnsure the cone is flat on the table.' + \
            '\nPress enter to continue.\nNote: A GUI window will' + \
            ' pop up to show the pose estimate.  Press \'q\' to close the ' + \
            'window and enable faster publishing without the GUI. ')
      hardcoded_initial_rot_mat = np.linalg.inv(world_to_cam[:3, :3]) @ WORLD_ROT_MAT_CONE
  print(f'run_live_demo: hardcoded_initial_rot_mat is None? {hardcoded_initial_rot_mat is None}')

  print('run_live_demo: creating predictors and rasterizer context')
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=debug_dir,
    debug=debug,
    glctx=glctx,
    hardcoded_initial_rot_mat=hardcoded_initial_rot_mat,
  )
  logging.info("estimator initialization done")
  print('run_live_demo: estimator initialization done')

  # The foreground mask is generated once and then resized to the current
  # frame size before the initial registration step.
  print('run_live_demo: creating foreground mask')
  create_mask()
  mask = cv2.imread('mask.png')
  print(f'run_live_demo: loaded mask with shape {None if mask is None else mask.shape}')

  # Create a RealSense pipeline and configure color + depth streaming.
  print('run_live_demo: creating RealSense pipeline')
  pipeline = rs.pipeline()

  # The config object defines the camera streams before the pipeline starts.
  print('run_live_demo: creating RealSense config')
  config = rs.config()

  # Resolve the device first so we can verify that a color-capable sensor is
  # present before starting the stream.
  print('run_live_demo: resolving pipeline/device')
  pipeline_wrapper = rs.pipeline_wrapper(pipeline)
  pipeline_profile = config.resolve(pipeline_wrapper)
  device = pipeline_profile.get_device()
  device_product_line = str(device.get_info(rs.camera_info.product_line))
  print(f'run_live_demo: device_product_line={device_product_line}')

  found_rgb = False
  for s in device.sensors:
      if s.get_info(rs.camera_info.name) == 'RGB Camera':
          found_rgb = True
          break
  print(f'run_live_demo: found_rgb={found_rgb}')
  if not found_rgb:
      print("The demo requires Depth camera with Color sensor")
      exit(0)

  # The demo assumes 640x480 synchronized streams for both depth and color.
  print('run_live_demo: enabling depth stream 640x480 z16@30')
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  print('run_live_demo: enabling color stream 640x480 rgb8@30')
  config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

  # Start streaming only after configuration and device validation succeed.
  print('run_live_demo: starting RealSense pipeline')
  profile = pipeline.start(config)

  # Depth scale converts raw depth units into metric depth values.
  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print("Depth Scale is: " , depth_scale)
  print(f'run_live_demo: depth_scale={depth_scale}')

  # Anything beyond this distance is treated as background in the local demo
  # logic.
  clipping_distance_in_meters = 1 #1 meter
  clipping_distance = clipping_distance_in_meters / depth_scale
  print(f'run_live_demo: clipping_distance_in_meters={clipping_distance_in_meters}, clipping_distance={clipping_distance}')

  # Align depth into the color camera frame so pose estimation consumes
  # consistent RGB-D pairs.
  print('run_live_demo: creating align object to color stream')
  align_to = rs.stream.color
  align = rs.align(align_to)

  i = 0
  print('run_live_demo: entering streaming loop setup')

  ################## HERE ##################

  if args.lcm_publish > 0:
    # LCM publishing is optional and only initialized when requested.
    print(f'run_live_demo: initializing PosePublisher for system={args.system}')
    lcm_pose_publisher = PosePublisher(system_name=args.system)

  # Estimation begins after a short delay so the camera stream can stabilize.
  Estimating = True
  keep_gui_window_open = True
  print('run_live_demo: sleeping before starting estimation loop')
  time.sleep(3)
  # Streaming loop.
  print('run_live_demo: starting streaming loop')
  try:
    while Estimating:
      start_time = time.perf_counter()
      print(f'run_live_demo: frame {i} start')
      # Fetch the next synchronized RGB-D frame pair from the camera.
      frames = pipeline.wait_for_frames()
      print('run_live_demo: frames received')

      # Reproject depth into the color frame so both arrays index the same
      # pixels.
      aligned_frames = align.process(frames)
      print('run_live_demo: frames aligned')

      # Extract the aligned depth and color frames from the synchronized set.
      aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
      color_frame = aligned_frames.get_color_frame()
      print(f'run_live_demo: depth_frame_ok={bool(aligned_depth_frame)}, color_frame_ok={bool(color_frame)}')

      # Skip frames that did not arrive cleanly.
      if not aligned_depth_frame or not color_frame:
        print('run_live_demo: skipping invalid frame pair')
        continue

      # Convert camera buffers into numpy arrays used by the estimator.
      depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
      color_image = np.asanyarray(color_frame.get_data())
      print(f'run_live_demo: depth_image.shape={depth_image.shape}, color_image.shape={color_image.shape}')

      # Convert the normalized depth image into the estimator's expected unit
      # scale.
      depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
      print('run_live_demo: depth image scaled')

      # cv2.imshow('color', color_image)
      # cv2.imshow('depth', depth_image)

      if cv2.waitKey(1) == 13:
        print('run_live_demo: enter key detected, stopping estimation')
        Estimating = False
        break

      # Per-frame trace marker for log correlation.
      logging.info(f'i:{i}')
      print(f'run_live_demo: processing frame {i}')

      # Resize to the working resolution expected by the pose estimator.
      H, W = cv2.resize(color_image, (640,480)).shape[:2]
      color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
      depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
      print(f'run_live_demo: resized frame to H={H}, W={W}')

      depth[(depth<0.1) | (depth>=np.inf)] = 0
      print('run_live_demo: invalid depth values zeroed')

      if i == 0:
        # The first frame uses the foreground mask and full registration to
        # bootstrap the object pose.
        print('run_live_demo: first frame registration path')
        if len(mask.shape) == 3:
          for c in range(3):
            if mask[...,c].sum() > 0:
              mask = mask[...,c]
              print(f'run_live_demo: selected mask channel {c}')
              break
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        print(f'run_live_demo: resized mask shape={mask.shape}')

        pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask,
                            iteration=args.est_refine_iter)
        print('run_live_demo: initial pose registered')

        if debug >= 3:
          # Extra diagnostics dump the aligned model and observed scene for
          # offline inspection when the debug level is high enough.
          print('run_live_demo: writing high-debug registration artifacts')
          m = mesh.copy()
          m.apply_transform(pose)
          m.export(f'{debug_dir}/model_tf.obj')
          xyz_map = depth2xyzmap(depth, cam_K)
          valid = depth >= 0.1
          pcd = toOpen3dCloud(xyz_map[valid], color[valid])
          o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

      else:
        # After initialization, tracking updates the pose frame-to-frame.
        print('run_live_demo: tracking update path')
        pose = est.track_one(rgb=color, depth=depth, K=cam_K,
                             iteration=args.track_refine_iter)
        print('run_live_demo: pose tracked')

      # Persist each estimated object pose so a run can be replayed later.
      print(f'run_live_demo: saving pose for frame {i}')
      os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
      np.savetxt(f'{debug_dir}/ob_in_cam/{i}.txt', pose.reshape(4,4))

      # Publish the pose over LCM.
      if args.lcm_publish > 0:
        # Convert the tracked object pose into the world frame before
        # publishing to downstream consumers.
        print('run_live_demo: publishing pose over LCM')
        cam_to_object = pose
        world_to_cam, is_near = get_world_T_cam(
            dist_from_cam=pose[2, 3], was_near=is_near)
        obj_pose_in_world = world_to_cam @ cam_to_object
        lcm_pose_publisher.publish_pose("Jack", obj_pose_in_world)
        print('run_live_demo: LCM publish complete')

      if keep_gui_window_open:
        # Overlay the estimated pose on the live image for fast operator
        # feedback.
        print('run_live_demo: rendering debug visualization')
        vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=pose, bbox=bbox)
        vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow("debug", vis[...,::-1])
        key = cv2.waitKey(1)
        print(f'run_live_demo: debug window key={key}')

        if debug <= 1 and keep_gui_window_open and (key==ord("q")):
          # The GUI can be closed once the operator is satisfied and only
          # background publishing needs to continue.
          print('run_live_demo: closing debug window on q press')
          cv2.destroyWindow("debug")
          keep_gui_window_open = False

      if debug >= 2:
        # Save visual tracking snapshots for post-run inspection.
        print(f'run_live_demo: writing track_vis image for frame {i}')
        os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
        imageio.imwrite(f'{debug_dir}/track_vis/{i}.png', vis)

      i += 1
      # Wall-clock timing is logged so slow frames can be identified quickly.
      print(f"duration: {time.perf_counter() - start_time}")
      print(f'run_live_demo: frame {i - 1} complete')

  finally:
    print('run_live_demo: stopping pipeline')
    pipeline.stop()

