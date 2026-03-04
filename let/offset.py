from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

DATASET_CONFIGS = {
    "default": {
        "permutation": [0, 1, 2],
        "signs": [1, 1, 1],
        "rotation": [0, 0, 0] 
    },
    "let": {
        "permutation": [2, 1, 0],
        "signs": [-1, 1, -1],
        "rotation": [0, 0, 0] 
    },
}

def get_transform_matrix(ds_id):
    cfg = DATASET_CONFIGS.get(ds_id, DATASET_CONFIGS["default"])
    perm = np.zeros((4, 4))
    perm[3, 3] = 1
    p = cfg['permutation']
    s = cfg['signs']
    for i in range(3):
        perm[i, p[i]] = s[i]
    rot = np.eye(4)
    if any(x != 0 for x in cfg['rotation']):
        rot[:3, :3] = R.from_euler('xyz', cfg['rotation'], degrees=True).as_matrix()
    # print(p)
    return rot @ perm

def apply_transform_to_matrix(pose_6d, transform_matrix):
    """
    Returns the 4x4 homogenous matrix. 
    Does NOT convert back to Euler angles to avoid 'Non-positive determinant' errors.
    """
    if pose_6d is None: return None
    
    t = pose_6d[:3]
    mat = np.eye(4)
    # Convert input Euler -> Matrix (Input is usually valid)
    mat[:3, :3] = R.from_euler('xyz', pose_6d[3:6], degrees=False).as_matrix()
    mat[:3, 3] = t
    
    # Apply transform (which might mirror it)
    new_mat = mat @ transform_matrix
    return new_mat

def apply_transform(pose_6d, ds_id):
    transform_mat = get_transform_matrix(ds_id)
    new_mat = apply_transform_to_matrix(pose_6d, transform_mat)
    
    if new_mat is None:
        return None

    # 1. Extract the new translation
    new_translation = new_mat[:3, 3]

    # 2. Extract the new rotation
    # We take the upper-left 3x3 rotation matrix
    rot_mat = new_mat[:3, :3]
    
    if np.linalg.det(rot_mat) < 0:
        rot_mat = rot_mat.copy()
        rot_mat[:, 1] *= -1

    new_euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=False)

    # 3. Concatenate to return 6D pose
    return np.concatenate([new_translation, new_euler]).astype(np.float32)