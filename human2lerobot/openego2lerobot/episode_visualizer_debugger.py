# AI-generated and just for debugging. Correctness not guaranteed
import cv2
import numpy as np
import torch
import os

def to_numpy(data):
    """Helper to ensure data is numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def project_points(points_3d, K, world_to_camera_transform):
    """
    Projects 3D world points to 2D image pixels.
    
    Args:
        points_3d: (N, 3) points in World Frame
        K: (3, 3) Intrinsic matrix
        world_to_camera_transform: (4, 4) Extrinsic matrix (World -> Camera)
    """
    if points_3d is None or len(points_3d) == 0:
        return []

    # 1. Convert to Homogeneous World Coords (N, 4)
    ones = np.ones((points_3d.shape[0], 1))
    points_hom = np.hstack([points_3d, ones]) # (N, 4)

    # 2. Transform to Camera Coordinate System
    # T_w2c * P_w
    points_cam = (world_to_camera_transform @ points_hom.T).T  # (N, 4)

    # 3. Project to Image Plane (Homogeneous 2D)
    # Only use x, y, z
    points_cam_xyz = points_cam[:, :3]
    points_img_hom = (K @ points_cam_xyz.T).T # (N, 3)

    # 4. Normalize by Z (Perspective Divide)
    u = points_img_hom[:, 0] / (points_img_hom[:, 2] + 1e-6)
    v = points_img_hom[:, 1] / (points_img_hom[:, 2] + 1e-6)
    
    return np.stack([u, v], axis=1)

def visualize_dataset(data_dict, output_path="/mnt/afs/lvjiangran/zhuwenxuan/any4lerobot/human2lerobot/openego2lerobot/debug_output.mp4", max_frames=400):
    # 1. Setup Video Reader
    video_path = data_dict["video_paths"]["observation.images.top_head"]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 2. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 3. Get Static Data (Intrinsics)
    K = to_numpy(data_dict["camera.intrinsic"])
    
    # Check if K is batched (T, 3, 3) or static (3, 3)
    if K.ndim == 3: 
        K_is_static = False
    else:
        K_is_static = True

    print(f"Processing video... Output: {output_path}")

    # 4. Loop through frames
    num_frames = len(data_dict["camera.extrinsic"])
    limit = min(num_frames, max_frames)
    
    for t in range(limit):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get Frame Data
        K_curr = K[t] if not K_is_static else K
        extrinsic = to_numpy(data_dict["camera.extrinsic"][t])
        
        # CRITICAL: Check Extrinsic Direction
        # If your matrix is Camera-to-World (common in datasets), invert it.
        # If it is World-to-Camera, leave it.
        # Try running once; if points fly away, uncomment the line below:
        extrinsic = np.linalg.inv(extrinsic) 
        
        # Get Hands
        hands_info = [
            ("left", (255, 0, 0)),   # Blue for Left
            ("right", (0, 0, 255))   # Red for Right
        ]
        
        for side, color in hands_info:
            key = f"eef.{side}.hand"
            if key in data_dict and data_dict[key] is not None:
                hand_kps = to_numpy(data_dict[key][t]) # (21, 3)
                
                # Project
                pixels = project_points(hand_kps, K_curr, extrinsic)
                # pixels_2d = data_dict[f"eef.{side}.2d"][t]

                #if t==157:
                #    print(pixels-pixels_2d)
                
                # Draw
                for (x, y) in pixels:
                    # Check bounds
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        out.write(frame)
        if t % 50 == 0:
            print(f"Processed frame {t}/{limit}")

    cap.release()
    out.release()
    print("Done.")

# --- RUN IT ---
# Assuming 'data_dict' is loaded in your memory
# visualize_dataset(data_dict)