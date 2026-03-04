"""
kinematics.py
Wrapper for Pinocchio to handle Forward Kinematics.
"""
import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation as R

class KinematicsSolver:
    def __init__(self, hand):
        urdf_path = "/mnt/home/lvjiangran/zhuwenxuan/let/kuavo-ros-opensource/src/kuavo_assets/models/biped_s47/urdf/biped_s47.urdf"
        if hand == "left":
            wrist_link_name = "zarm_l7_end_effector"
        else:
            wrist_link_name = "zarm_r7_end_effector"
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        print(wrist_link_name)
        self.wrist_frame_id = self.model.getFrameId(wrist_link_name)
        print(self.wrist_frame_id)
        #print("--- JOINT NAMES IN URDF ---")
        #for i, name in enumerate(self.model.names):
            # Pinocchio adds a 'universe' joint at 0, so indices shift by 1 compared to q
        #    print(f"Index {i}: {name}")

    def compute_eef_pose(self, joint_positions):
        """
        Computes (x, y, z, r, p, y) for the given joint positions.
        """
        # Pinocchio expects the full configuration vector. 
        # If the robot has fewer joints in the config than the URDF (e.g. gripper fingers),
        # we pad with zeros.
        q = np.zeros(self.model.nq)
        n = min(len(joint_positions), self.model.nq)
        q[:n] = joint_positions[:n]

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get frame placement
        transform = self.data.oMf[self.wrist_frame_id]
        
        pos = transform.translation
        rot_mat = transform.rotation
        
        # Convert rotation matrix to Euler XYZ (extrinsic)
        r = R.from_matrix(rot_mat)
        euler = r.as_euler('xyz', degrees=False)
        
        # Return format: x, y, z, r, p, y
        return (np.concatenate([pos, euler])).astype(np.float32)