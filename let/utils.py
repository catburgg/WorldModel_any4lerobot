from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from scipy.interpolate import interp1d


def se3_to_6d(se3_obj):
    """
    Input: sophus.SE3 object
    Output: (x, y, z, roll, pitch, yaw)
    """
    
    t = np.array(se3_obj.translation()).flatten()
    x, y, z = t[0], t[1], t[2]

    R = se3_obj.rotation().to_matrix()
    rot = Rotation.from_matrix(R)
    roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

    return np.array((x, y, z, roll, pitch, yaw))

def resample_poses(
    source_poses: np.ndarray, 
    target_num_frames: int
) -> np.ndarray:
    
    source_num_frames = len(source_poses)
    
    source_times_norm = np.linspace(0, 1, source_num_frames)
    target_times_norm = np.linspace(0, 1, target_num_frames)
    
    trans_src = source_poses[:, :3, 3] 
    rot_src_matrices = source_poses[:, :3, :3]
    rot_src_obj = Rotation.from_matrix(rot_src_matrices)
    
    interp_trans = interp1d(source_times_norm, trans_src, axis=0, kind='linear')
    trans_interp = interp_trans(target_times_norm)
    
    slerp = Slerp(source_times_norm, rot_src_obj)
    rot_interp_obj = slerp(target_times_norm)
    rot_interp_matrices = rot_interp_obj.as_matrix()
    
    new_poses = np.eye(4).reshape(1, 4, 4).repeat(target_num_frames, axis=0)
    new_poses[:, :3, :3] = rot_interp_matrices
    new_poses[:, :3, 3] = trans_interp
    
    return new_poses

def resample_motion(gripper_data, target_length):
    data = np.array(gripper_data)
    if data.ndim == 1:
        data = data[:, None]
        
    source_length = len(data)
    
    source_times = np.linspace(0, 1, source_length)
    target_times = np.linspace(0, 1, target_length)
    
    interpolator = interp1d(source_times, data, axis=0, kind='linear', fill_value="extrapolate")
    
    resampled_data = interpolator(target_times)
    
    return resampled_data.flatten().astype(np.float32)

def convert_cam_to_world(pts, poses):
    N, J, _ = pts.shape
    
    ones = np.ones((N, J, 1))
    pts_homo = np.concatenate([pts, ones], axis=-1)
    
    pts_world_homo = np.einsum('nij, nkj -> nki', poses, pts_homo)
    pts_world = pts_world_homo[..., :3]
    
    return pts_world

def resize_intrinsics(K, orig_w, orig_h, new_w, new_h):
    
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    K_new = K.copy()
    
    K_new[0, 0] *= scale_x  # fx
    K_new[1, 1] *= scale_y  # fy
    
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 2] *= scale_y  # cy
    
    return K_new

def resample_6d(source_data, target_len):
    """
    Resamples an (N, 6) trajectory (Translation + Rotation) to a new length.
    Handles rotation wrapping (3.14 -> -3.14) automatically.
    """
    # 1. Ensure input is a Numpy Array (Fixes your TypeError)
    source_data = np.array(source_data)
    
    source_len = len(source_data)
    
    # 2. Setup Time steps
    source_times = np.linspace(0, 1, source_len)
    target_times = np.linspace(0, 1, target_len)
    
    # 3. Handle Rotation Wrapping (Indices 3, 4, 5)
    # This prevents the robot from spinning 360 degrees unnecessarily
    data_unwrapped = np.copy(source_data)
    
    # Assuming indices 3, 4, 5 are the rotation angles (radians)
    data_unwrapped[:, 3:] = np.unwrap(data_unwrapped[:, 3:], axis=0)
    
    # 4. Interpolate Everything (Linear)
    interpolator = interp1d(source_times, data_unwrapped, axis=0, kind='linear', fill_value="extrapolate")
    resampled_data = interpolator(target_times)
    
    # 5. Re-wrap angles to [-pi, pi] (Optional, but good for drivers)
    resampled_data[:, 3:] = (resampled_data[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    
    return resampled_data.astype(np.float32)

def process_frames(frame_list):
    processed = []
    for frame in frame_list:
        # Check if Channel-First (e.g., 3, 128, 128) -> Convert to (128, 128, 3)
        if frame.shape[0] == 3: 
            frame = np.transpose(frame, (1, 2, 0))
            
        # Check if Float (0-1) -> Convert to Int (0-255)
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        processed.append(frame)
    return processed