from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.stream_id import StreamId
import h5py

class AriaCamera:
    """Get data from the original aria camera"""
    
    def __init__(self, hot3d_data_provider):
        self.hot3d_data_provider = hot3d_data_provider
        self.device_data_provider = hot3d_data_provider.device_data_provider
        self.stream_id = StreamId("214-1")
        self.stream_label = "camera-rgb"
        self.target_res = (512,512)
        self.target_depth = 240.0
        self.timestamps = self.device_data_provider.get_sequence_timestamps()
        self.vrs_provider = self.device_data_provider._vrs_data_provider
        self.device_calib = self.vrs_provider.get_device_calibration()
        self.real_src_calib = self.device_calib.get_camera_calib(self.stream_label)
        self.pinhole_calib = calibration.get_linear_camera_calibration(
            self.target_res[0], 
            self.target_res[1], 
            self.target_depth,
            self.stream_label, 
            self.real_src_calib.get_transform_device_camera()
        )
        
    def get_calibration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        return the (camera_device) extrinsics & intrinsics of rgb camera
        Modified for k_rotation=3 (90 degree Clockwise rotation)
        """
        focal_lengths = self.pinhole_calib.get_focal_lengths()
        principal_point = self.pinhole_calib.get_principal_point()
        
        w_old = self.pinhole_calib.get_image_size()[0]
        h_old = self.pinhole_calib.get_image_size()[1]

        K = np.eye(3)
        K[0, 0] = focal_lengths[1] 
        K[1, 1] = focal_lengths[0]
        
        K[0, 2] = h_old - principal_point[1]
        K[1, 2] = principal_point[0]

        transform_se3 = self.pinhole_calib.get_transform_device_camera()
        T_original = transform_se3.to_matrix()

        R_correction = np.array([
            [ 0,  1,  0,  0],
            [-1,  0,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ], dtype=float)

        T_new = T_original @ R_correction

        return T_new, K

    def undistort_image(self, raw_image) -> np.ndarray:
        """undistort the image from original aria rgb camera"""
        return calibration.distort_by_calibration(
            raw_image, 
            self.pinhole_calib, 
            self.real_src_calib 
        )


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

def resample_motion(A: np.ndarray, target_length: int) -> np.ndarray:

    current_length = A.shape[0]
    
    zoom_factor = target_length / current_length
    
    factors = (zoom_factor, 1, 1)
    
    output = zoom(A, factors, order=1) 
    
    return output

def convert_cam_to_world(pts, poses):
    N, J, _ = pts.shape
    
    ones = np.ones((N, J, 1))
    pts_homo = np.concatenate([pts, ones], axis=-1)
    
    pts_world_homo = np.einsum('nij, nkj -> nki', poses, pts_homo)
    pts_world = pts_world_homo[..., :3]
    
    return pts_world


def load_hdf5_to_dict(h5_input):
    data = {}
    if not hasattr(h5_input, 'items'):
        raise TypeError(f"Input must be an h5py object or file path, not {type(h5_input)}")

    for key, item in h5_input.items():
        if isinstance(item, h5py.Group):
            data[key] = load_hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            value = item[()]
            if isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            data[key] = value
            
    return data