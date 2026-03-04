import torch
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
# Update this path to your local dataset location
LOCAL_DATASET_PATH = "/mnt/project/world_model/data/HumanData/HOI4D" 
EPISODE_INDEX = 36
OUTPUT_FILENAME = "wrist_pose_viz.mp4"
AXIS_LENGTH = 0.1  # Length of axes in meters (Adjust if lines are too big/small)

def get_pose_matrix(pose_6d):
    """
    Converts a 6D pose (x, y, z, roll, pitch, yaw) into a 4x4 transformation matrix.
    Assumes Euler angles are in radians and order is XYZ.
    """
    # Detach if it's a tensor, otherwise ensure numpy
    if isinstance(pose_6d, torch.Tensor):
        pose_6d = pose_6d.numpy()
        
    x, y, z, r, p, y_ang = pose_6d
    
    # Translation
    t = np.array([x, y, z])
    
    # Rotation (Euler -> Matrix)
    # Note: 'xyz' is standard for roll-pitch-yaw. If axes look flipped, try 'zyx'.
    rotation = R.from_euler('xyz', [r, p, y_ang], degrees=False)
    rot_mat = rotation.as_matrix()
    
    # 4x4 Matrix
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = t
    return T

def project_points(points_3d, intrinsics, extrinsics):
    """
    Projects 3D points (Nx3) into 2D pixel coordinates (Nx2).
    """
    # 1. Convert to Homogeneous (Nx4)
    ones = np.ones((points_3d.shape[0], 1))
    points_homog = np.hstack([points_3d, ones]) 

    # 2. World -> Camera (Inverse of Camera Pose)
    world_to_cam = np.linalg.inv(extrinsics) 
    points_cam = (world_to_cam @ points_homog.T).T 

    # 3. Camera -> Image Plane
    xyz_cam = points_cam[:, :3]
    uv_z = (intrinsics @ xyz_cam.T).T 

    # 4. Normalize (Perspective Divide)
    z = uv_z[:, 2:3]
    z[z == 0] = 1e-5 # Avoid div/0
    uv = uv_z[:, :2] / z
    
    return uv.astype(int)

def draw_pose_axes(img_bgr, pose_6d, K, E, axis_len=0.1):
    """
    Draws RGB axes for a specific wrist pose.
    """
    # 1. Get Wrist -> World Matrix
    wrist_to_world = get_pose_matrix(pose_6d)

    # 2. Define Local Axes (Origin, X, Y, Z)
    local_axes = np.array([
        [0.0, 0.0, 0.0],      # Origin
        [axis_len, 0.0, 0.0], # X
        [0.0, axis_len, 0.0], # Y
        [0.0, 0.0, axis_len]  # Z
    ])

    # 3. Transform Local -> World
    # (Manual multiplication for points batch)
    ones = np.ones((4, 1))
    local_hom = np.hstack([local_axes, ones])
    world_axes = (wrist_to_world @ local_hom.T).T[:, :3]

    # 4. Project World -> Pixels
    pixels = project_points(world_axes, K, E)

    # 5. Draw Lines
    origin = tuple(pixels[0])
    
    # Check bounds briefly to avoid drawing crazy lines if tracking is lost
    h, w = img_bgr.shape[:2]
    if 0 <= origin[0] < w and 0 <= origin[1] < h:
        # X Axis (Red)
        cv2.line(img_bgr, origin, tuple(pixels[1]), (0, 0, 255), 2)
        # Y Axis (Green)
        cv2.line(img_bgr, origin, tuple(pixels[2]), (0, 255, 0), 2)
        # Z Axis (Blue)
        cv2.line(img_bgr, origin, tuple(pixels[3]), (255, 0, 0), 2)

def main():
    # 1. Load Local Dataset
    print(f"Loading local dataset from: {LOCAL_DATASET_PATH}")
    dataset = LeRobotDataset(
        root=LOCAL_DATASET_PATH, 
        repo_id="dummy_id", 
        episodes=[EPISODE_INDEX]
    )

    print(f"Processing Episode {EPISODE_INDEX} (Total frames: {len(dataset)})...")

    writer = None

    for frame_idx in range(min(len(dataset), 1000)):
        frame_data = dataset[frame_idx]
        item = dataset[frame_idx]

        # --- A. Prepare Image ---
        img_tensor = item["observation.images.top_head"]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Scale float [0,1] -> uint8 [0,255]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Init Writer
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, 10.0, (w, h))

        # --- B. Get Matrices ---
        K = item["camera.intrinsic"].numpy()   # (3, 3)
        E = item["camera.extrinsic"].numpy()   # (4, 4)

        # --- C. Draw Wrist Axes (The new part) ---
        if "eef.left.wrist" in item:
            draw_pose_axes(frame_bgr, item["eef.left.wrist"], K, E, axis_len=AXIS_LENGTH)
        
        if "eef.right.wrist" in item:
            draw_pose_axes(frame_bgr, item["eef.right.wrist"], K, E, axis_len=AXIS_LENGTH)

        # --- D. Draw Hand Keypoints (Your original dots) ---
        # Optional: Comment out if it looks too cluttered
        # Define a font for the numbers
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (255, 255, 255) # White text
        thickness = 1

        if "eef.left.hand" in item:
            hand_3d = item["eef.left.hand"].numpy()
            pixels = project_points(hand_3d, K, E)
            
            # Iterate with index 'i' to get the joint number (0-20)
            for i, (x, y) in enumerate(pixels):
                # 1. Draw the dot (Yellow)
                cv2.circle(frame_bgr, (int(x), int(y)), 3, (0, 255, 255), -1)
                
                # 2. Draw the ID number next to it
                # Offset the text slightly (+4 pixels) so it doesn't overlap the dot
                cv2.putText(frame_bgr, str(i), (int(x)+4, int(y)-4), 
                            font, font_scale, font_color, thickness, cv2.LINE_AA)

        if "eef.right.hand" in item:
            hand_3d = item["eef.right.hand"].numpy()
            pixels = project_points(hand_3d, K, E)
            
            for i, (x, y) in enumerate(pixels):
                # 1. Draw the dot (Magenta)
                cv2.circle(frame_bgr, (int(x), int(y)), 3, (255, 0, 255), -1)
                
                # 2. Draw the ID number
                cv2.putText(frame_bgr, str(i), (int(x)+4, int(y)-4), 
                            font, font_scale, font_color, thickness, cv2.LINE_AA)

        # --- E. Write Frame ---
        writer.write(frame_bgr)

        if (frame_idx) % 50 == 0:
            print(f"Processed frame {frame_idx}...")

    if writer:
        writer.release()
    print(f"Done! Video saved to: {Path(OUTPUT_FILENAME).absolute()}")

if __name__ == "__main__":
    main()