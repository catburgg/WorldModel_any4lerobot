from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

def decode_images(camera_key, input_images, bgr2rgb: bool = False):
    if "depth" not in camera_key:
        rgb_images = []
        camera_rgb_images = input_images
        for camera_rgb_image in camera_rgb_images:
            camera_rgb_image = np.array(camera_rgb_image)
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if rgb is None:
                rgb = np.frombuffer(camera_rgb_image, dtype=np.uint8)
                if rgb.size == 2764800:
                    rgb = rgb.reshape(720, 1280, 3)
                elif rgb.size == 921600:
                    rgb = rgb.reshape(480, 640, 3)
            if bgr2rgb:
                rgb = rgb[..., ::-1]
            rgb_images.append(rgb)
        rgb_images = np.asarray(rgb_images)
        return rgb_images
    else:
        depth_images = []
        camera_depth_images = input_images
        for camera_depth_image in camera_depth_images:
            if isinstance(camera_depth_image, np.ndarray):
                depth_array = camera_depth_image
            else:
                depth_array = np.frombuffer(camera_depth_image, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            if depth is None:
                if depth_array.size == 921600:
                    depth = depth_array.reshape(720, 1280)
                elif depth_array.size == 307200:
                    depth = depth_array.reshape(480, 640)
            depth_images.append(depth)
        depth_images = np.asarray(depth_images)[..., None]
        return depth_images

intrinsic_matrices = {}
intrinsic_matrices["agilex_3rgb"] = np.array([
    [455.47412109375, 0.0, 327.1614074707031],
    [0.0, 455.47412109375, 236.7096710205078],
    [0.0, 0.0, 1.0]], dtype=np.float32)
intrinsic_matrices["franka_1rgb"] = np.array([
    [608.78, 0.0, 322.79],
    [0.0, 608.537, 257.3597],
    [0.0, 0.0, 1.0]], dtype=np.float32)
intrinsic_matrices["franka_3rgb"] = np.array([
    [909.6201171875, 0.0, 635.703125],
    [0.0, 908.7317504882812, 364.509368896844],
    [0.0, 0.0, 1.0]], dtype=np.float32)
intrinsic_matrices["ur_1rgb"] = np.array([
      [609.3246327846327, 0.0, 317.7893242938894],
      [0.0, 609.7463827932434, 257.3246343287439],
      [0.0, 0.0, 1.0]], dtype=np.float32)
intrinsic_matrices["franka_fr3_dual"] = np.array([
      [693.0001220703125, 0.0, 642.3272094726562],
      [0.0, 692.7867431640625, 358.94342041015625],
      [0.0, 0.0, 1.0]], dtype=np.float32)

def load_local_dataset(episode_path: Path, config: dict, save_depth: bool, bgr2rgb: bool = False, robot_type: str = None):
    try:
        images = {}
        states = {}
        actions = {}
        camera_key = None
        with h5py.File(episode_path, "r") as file:
            if robot_type == "agilex_3rgb":
                if "observations/rgb_images/camera_front" in file:
                    camera_key = "observations/rgb_images/camera_front"
                
                if "puppet/end_effector_left" in file:
                    eef_left = np.array(file["puppet/end_effector_left"], dtype=np.float32)
                    states["eef.left.wrist"] = eef_left[:, :6]
                    states["eef.left.hand"] = eef_left[:, 6:7]
                
                if "puppet/end_effector_right" in file:
                    eef_right = np.array(file["puppet/end_effector_right"], dtype=np.float32)
                    states["eef.right.wrist"] = eef_right[:, :6]
                    states["eef.right.hand"] = eef_right[:, 6:7]

                if "puppet/joint_position_left" in file and "puppet/joint_position_right" in file:
                    jp_left = np.array(file["puppet/joint_position_left"], dtype=np.float32)
                    jp_right = np.array(file["puppet/joint_position_right"], dtype=np.float32)
                    states["proprioception_raw"] = np.concatenate([jp_left, jp_right], axis=-1)
            elif robot_type == "franka_1rgb" or robot_type == "franka_3rgb":
                if "observations/rgb_images/camera_top" in file:
                    camera_key = "observations/rgb_images/camera_top"

                if "puppet/end_effector" in file:
                    eef = np.array(file["puppet/end_effector"], dtype=np.float32)
                    states["eef.left.wrist"] = eef[:, :6]
                    states["eef.right.wrist"] = eef[:, :6]

                if "puppet/joint_position" in file:
                    jp = np.array(file["puppet/joint_position"], dtype=np.float32)
                    states["eef.left.hand"] = jp[:, 7:8]
                    states["eef.right.hand"] = jp[:, 7:8]
                    states["proprioception_raw"] = jp
            elif robot_type == "simulation" or robot_type == "sim_franka_3rgb":
                if "observations/rgb_images/camera_front_external" in file:
                    camera_key = "observations/rgb_images/camera_front_external"

                if "franka/end_effector" in file:
                    eef = np.array(file["franka/end_effector"], dtype=np.float32)
                    # Convert (x, y, z, qx, qy, qz, qw) to (x, y, z, r, p, y)
                    pos = eef[:, :3]
                    quat = eef[:, 3:]
                    euler = Rotation.from_quat(quat).as_euler("xyz")
                    states["eef.left.wrist"] = np.concatenate([pos, euler], axis=-1).astype(np.float32)
                    states["eef.right.wrist"] = np.concatenate([pos, euler], axis=-1).astype(np.float32)

                if "franka/joint_position" in file:
                    jp = np.array(file["franka/joint_position"], dtype=np.float32)
                    states["eef.left.hand"] = jp[:, 7:8]
                    states["eef.right.hand"] = jp[:, 7:8]
                    states["proprioception_raw"] = jp
                
                # 解决图像和状态原始采样率不一致的问题
                if camera_key and camera_key in file:
                    n_img = len(file[camera_key])
                    if states:
                        n_state = len(next(iter(states.values())))
                        # logging.info(f"Downsampling check: n_state={n_state}, n_img={n_img}")
                        if n_state > n_img:
                            step = n_state // n_img
                            for k in states:
                                states[k] = states[k][::step][:n_img]
                            # logging.info(f"Downsampled states to length {len(next(iter(states.values())))}")
            elif robot_type == "tienkung_gello_1rgb" or robot_type == "tienkung_prod1_gello_1rgb":
                if "observations/rgb_images/camera_top" in file:
                    camera_key = "observations/rgb_images/camera_top"
                
                if "puppet/joint_position" in file:
                    jp = np.array(file["puppet/joint_position"], dtype=np.float32)
                    states["eef.left.hand"] = jp[:, 7: 8]
                    states["eef.right.hand"] = jp[:, 15: 16]
                    states["proprioception_raw"] = jp
                    
                    # Fill missing wrist data with zeros
                    num_frames = jp.shape[0]
                    states["eef.left.wrist"] = np.zeros((num_frames, 6), dtype=np.float32)
                    states["eef.right.wrist"] = np.zeros((num_frames, 6), dtype=np.float32)
            elif robot_type == "tienkung_xsens_1rgb":
                if "observations/rgb_images/camera_top" in file:
                    camera_key = "observations/rgb_images/camera_top"

                if "puppet/joint_position" in file and "puppet/end_effector" in file:
                    jp = np.array(file["puppet/joint_position"], dtype=np.float32)
                    hands = np.array(file["puppet/end_effector"], dtype=np.float32)
                    states["proprioception_raw"] = np.concatenate([jp, hands], axis=-1)

                    # Fill missing wrist and hand data with zeros
                    num_frames = jp.shape[0]
                    states["eef.left.wrist"] = np.zeros((num_frames, 6), dtype=np.float32)
                    states["eef.right.wrist"] = np.zeros((num_frames, 6), dtype=np.float32)
                    states["eef.left.hand"] = np.zeros((num_frames, 1), dtype=np.float32)
                    states["eef.right.hand"] = np.zeros((num_frames, 1), dtype=np.float32)
            elif robot_type == "ur_1rgb":
                if "observations/rgb_images/camera_top" in file:
                    camera_key = "observations/rgb_images/camera_top"

                if "puppet/end_effector" in file:
                    eef = np.array(file["puppet/end_effector"], dtype=np.float32)
                    states["eef.left.wrist"] = eef
                    states["eef.right.wrist"] = eef
                
                if "puppet/joint_position" in file:
                    jp = np.array(file["puppet/joint_position"], dtype=np.float32)
                    states["proprioception_raw"] = jp
                    states["eef.left.hand"] = jp[:, 6:7]
                    states["eef.right.hand"] = jp[:, 6:7]
            elif robot_type == "franka_fr3_dual":
                if "observations/rgb_images/camera_top" in file:
                    camera_key = "observations/rgb_images/camera_top"

                if "puppet/end_effector" in file:
                    eef = np.array(file["puppet/end_effector"], dtype=np.float32)
                    states["eef.left.wrist"] = eef[:, :6]
                    states["eef.right.wrist"] = eef[:, 6:12]
                
                if "puppet/joint_position" in file:
                    jp = np.array(file["puppet/joint_position"], dtype=np.float32)
                    states["eef.left.hand"] = jp[:, 7:8]
                    states["eef.right.hand"] = jp[:, 15:16]
                    states["proprioception_raw"] = jp
            elif robot_type == "sim_tienkung_1rgb":
                if "observations/rgb_images/camera_head" in file:
                    camera_key = "observations/rgb_images/camera_head"
                
                if "tiangong/left_arm_joint_pos_seq" in file and "tiangong/left_hand_joint_pos_seq" in file and "tiangong/right_arm_joint_pos_seq" in file and "tiangong/right_hand_joint_pos_seq" in file:
                    left_arm_jp = np.array(file["tiangong/left_arm_joint_pos_seq"], dtype=np.float32)
                    left_hand_jp = np.array(file["tiangong/left_hand_joint_pos_seq"], dtype=np.float32)
                    right_arm_jp = np.array(file["tiangong/right_arm_joint_pos_seq"], dtype=np.float32)
                    right_hand_jp = np.array(file["tiangong/right_hand_joint_pos_seq"], dtype=np.float32)
                    
                    min_len = min(left_arm_jp.shape[0], left_hand_jp.shape[0], right_arm_jp.shape[0], right_hand_jp.shape[0])
                    left_arm_jp = left_arm_jp[:min_len]
                    left_hand_jp = left_hand_jp[:min_len]
                    right_arm_jp = right_arm_jp[:min_len]
                    right_hand_jp = right_hand_jp[:min_len]

                    states["proprioception_raw"] = np.concatenate([left_arm_jp, right_arm_jp, left_hand_jp, right_hand_jp], axis=-1)

                    # Fill missing wrist and hand data with zeros
                    num_frames = min_len
                    states["eef.left.wrist"] = np.zeros((num_frames, 6), dtype=np.float32)
                    states["eef.right.wrist"] = np.zeros((num_frames, 6), dtype=np.float32)
                    states["eef.left.hand"] = np.zeros((num_frames, 1), dtype=np.float32)
                    states["eef.right.hand"] = np.zeros((num_frames, 1), dtype=np.float32)
            else:
                raise NotImplementedError(f"Robot type {robot_type} not supported.")

            if not states:
                 raise ValueError(f"No state data found for {robot_type} in {episode_path}. Keys found: {list(file.keys())}")

            num_frames = len(next(iter(states.values())))
            
            if camera_key is not None:
                images["observation.images.top_head"] = decode_images(camera_key, file[camera_key], bgr2rgb)
                num_frames = min(num_frames, len(images["observation.images.top_head"]))
            
            # Truncate states to match num_frames
            for k in states:
                states[k] = states[k][:num_frames]

            # states["camera.intrinsic"] = np.zeros((num_frames, 9), dtype=np.float32)
            # states["camera.extrinsic"] = np.zeros((num_frames, 16), dtype=np.float32)
            
            if camera_key is not None:
                if robot_type in intrinsic_matrices:
                    states["camera.intrinsic"] = intrinsic_matrices[robot_type].reshape((1, 9)).repeat(num_frames, axis=0)

            if "states" in config:
                for state_key, state_cfg in config["states"].items():
                    if state_key in ["task_index", "timestamp"]:
                        continue
                    if state_key not in states:
                        shape = state_cfg["shape"]
                        states[state_key] = np.zeros((num_frames, *shape), dtype=np.float32)

            if "images" in config:
                for image_key, image_cfg in config["images"].items():
                    full_image_key = f"observation.images.{image_key}"
                    if full_image_key not in images:
                        shape = image_cfg["shape"]
                        images[full_image_key] = np.zeros((num_frames, *shape), dtype=np.uint8)

        num_frames = len(next(iter(states.values())))
        frames = [
            {
                **{key: value[i] for key, value in images.items() if save_depth or "depth" not in key},
                **{key: value[i] for key, value in states.items()},
                **{key: value[i] for key, value in actions.items()},
            }
            for i in range(num_frames)
        ]
        return True, frames, ""

    except (FileNotFoundError, OSError, KeyError) as e:
        return False, [], e
