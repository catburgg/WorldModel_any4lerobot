# AI-generated and just for debugging. Correctness not guaranteed
import torch
import numpy as np
import cv2
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Configuration ---
# Path to your local dataset folder (the folder containing info.json)
LOCAL_DATASET_PATH = "/mnt/afs/lvjiangran/zhuwenxuan/holoassist/result" 
EPISODE_INDEX = 3
OUTPUT_FILENAME = "episode_viz.mp4"

def project_points(points_3d, intrinsics, extrinsics):
    """
    Projects 3D points (Nx3) into 2D pixel coordinates (Nx2).
    Assumes extrinsics are Camera-to-World (Pose).
    """
    # 1. Convert points to Homogeneous coordinates (Nx4)
    ones = np.ones((points_3d.shape[0], 1))
    points_homog = np.hstack([points_3d, ones]) # Shape: (N, 4)

    # 2. Get World-to-Camera Matrix (Inverse of Pose)
    # If your extrinsics are already View Matrix, remove np.linalg.inv()
    world_to_cam = np.linalg.inv(extrinsics) 

    # 3. Transform World -> Camera Frame
    points_cam = (world_to_cam @ points_homog.T).T # Shape: (N, 4)

    # 4. Project Camera -> Image Plane (using Intrinsics)
    # Only keep x, y, z
    points_cam_xyz = points_cam[:, :3] 
    points_img_homog = (intrinsics @ points_cam_xyz.T).T # Shape: (N, 3)

    # 5. Normalize by Z (depth) to get u, v
    # Avoid division by zero
    z = points_img_homog[:, 2:3]
    z[z == 0] = 1e-5
    pixel_coords = points_img_homog[:, :2] / z
    
    return pixel_coords

def main():
    # 1. Load Local Dataset
    print(f"Loading local dataset from: {LOCAL_DATASET_PATH}")
    dataset = LeRobotDataset(root=LOCAL_DATASET_PATH, repo_id="dummy_id") 
    # Note: repo_id is required by arg parser but ignored if root is valid and local

    # 2. Get Episode Range
    from_idx = dataset.episode_data_index["from"][EPISODE_INDEX]
    to_idx = dataset.episode_data_index["to"][EPISODE_INDEX]
    print(to_idx)
    
    print(f"Processing Episode {EPISODE_INDEX} (Frames {from_idx} to {to_idx})...")

    # 3. Initialize Video Writer
    writer = None

    for frame_idx in range(from_idx, to_idx):
        item = dataset[frame_idx]

        # --- A. Get Image ---
        # Convert Torch (C, H, W) -> Numpy (H, W, C) -> BGR for OpenCV
        img_tensor = item["observation.images.top_head"]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Ensure range 0-255 uint8
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Initialize writer once we know image dimensions
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, 10.0, (w, h))

        # --- B. Get Matrices ---
        K = item["camera.intrinsic"].numpy()   # (3, 3)
        E = item["camera.extrinsic"].numpy()   # (4, 4)

        # --- C. Project and Draw Left Hand ---
        if "eef.left.hand" in item:
            hand_3d = item["eef.left.hand"].numpy() # (21, 3)
            pixels = project_points(hand_3d, K, E)

            for x, y in pixels:
                # Draw only if within image bounds
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame_bgr, (int(x), int(y)), 3, (0, 0, 255), -1) # Red

        # --- D. Project and Draw Right Hand ---
        if "eef.right.hand" in item:
            hand_3d = item["eef.right.hand"].numpy() # (21, 3)
            pixels = project_points(hand_3d, K, E)

            for x, y in pixels:
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame_bgr, (int(x), int(y)), 3, (0, 255, 0), -1) # Green

        # --- E. Write Frame ---
        writer.write(frame_bgr)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}...")

    writer.release()
    print(f"Done! Video saved to: {Path(OUTPUT_FILENAME).absolute()}")

if __name__ == "__main__":
    main()