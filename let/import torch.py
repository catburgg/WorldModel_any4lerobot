import torch
import numpy as np
import cv2
import argparse
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation as R

# ==============================================================================
#  DATASET ALIGNMENT CONFIGURATION
# ==============================================================================
# TARGET DEFINITION:
#   X (Red)   -> Forward (Along Gripper)
#   Y (Green) -> Down (To Ground)
#   Z (Blue)  -> Right (To the side)
DATASET_CONFIGS = {
    "default": {
        "permutation": [0, 1, 2],
        "signs": [1, 1, 1],
        "rotation": [0, 0, 0] 
    }
}
# ==============================================================================
# --- Constants ---
PLOT_HISTORY = 120
HUD_SIZE = 220
PLOT_PANEL_HEIGHT = 150
PADDING = 20
DISPLAY_IMG_H = 520 

# Colors
C_X = (0, 0, 255)   # Red
C_Y = (0, 255, 0)   # Green
C_Z = (255, 0, 0)   # Blue

def to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)
    return data

def find_video_file(dataset_root, episode_index, preferred_camera=None):
    video_dir = dataset_root / "videos"
    if not video_dir.exists(): return None, "No video dir"
    target = f"episode_{episode_index:06d}.mp4"
    files = list(video_dir.rglob(target))
    if not files: return None, "No video found"
    
    sel = files[0]
    if preferred_camera:
        for f in files:
            if preferred_camera in str(f):
                sel = f
                break
    return sel, "Found"

def get_transform_matrix(ds_id):
    # cfg = DATASET_CONFIGS.get(ds_id, DATASET_CONFIGS["default"])
    cfg = DATASET_CONFIGS["default"]
    perm = np.zeros((4, 4))
    perm[3, 3] = 1
    p = cfg['permutation']
    s = cfg['signs']
    for i in range(3):
        perm[i, p[i]] = s[i]
    rot = np.eye(4)
    if any(x != 0 for x in cfg['rotation']):
        rot[:3, :3] = R.from_euler('xyz', cfg['rotation'], degrees=True).as_matrix()
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

def draw_hud(canvas, rect, pose_mat, grip, title, view_mode="TOP"):
    x, y, s, _ = rect
    cx, cy = x + s//2, y + s//2
    scale = s // 3 

    cv2.rectangle(canvas, (x, y), (x+s, y+s), (40,40,40), -1)
    cv2.rectangle(canvas, (x, y), (x+s, y+s), (100,100,100), 1)

    if pose_mat is None:
        cv2.putText(canvas, "NO DATA", (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        return

    # --- 1. Position Projection ---
    pos = pose_mat[:3, 3] # Extract translation from matrix
    
    if view_mode == "TOP":
        u_pos = int(-pos[1] * scale * 2.5) 
        v_pos = int(-pos[0] * scale * 2.5)
        label = "TOP"
        proj_u = np.array([0, -1, 0]) 
        proj_v = np.array([-1, 0, 0])
        proj_d = np.array([0, 0, -1]) 
    else: # FRONT
        u_pos = int(-pos[1] * scale * 2.5)
        v_pos = int(-(pos[2]-0.3) * scale * 2.5)
        label = "FRONT"
        proj_u = np.array([0, -1, 0]) 
        proj_v = np.array([0, 0, -1]) 
        proj_d = np.array([-1, 0, 0])

    u_pos = np.clip(u_pos, -s//2 + 10, s//2 - 10)
    v_pos = np.clip(v_pos, -s//2 + 10, s//2 - 10)
    center = (cx + u_pos, cy + v_pos)
    cv2.circle(canvas, center, 4, (200, 200, 200), -1)

    # --- 2. Orientation Axes (Using Matrix Directly) ---
    # We use the columns of the rotation matrix directly as the X, Y, Z vectors.
    # This works even if the matrix is mirrored/skewed.
    rot_mat = pose_mat[:3, :3]
    L = 35 
    
    axes_depths = []
    for i in range(3):
        vec = rot_mat[:, i]
        depth_val = np.dot(vec, proj_d)
        axes_depths.append((i, depth_val))
    
    axes_depths.sort(key=lambda x: x[1])

    for i, depth_val in axes_depths:
        color = [C_X, C_Y, C_Z][i]
        vec = rot_mat[:, i]
        
        du = np.dot(vec, proj_u) * L
        dv = np.dot(vec, proj_v) * L
        pt2 = (int(center[0] + du), int(center[1] + dv))
        
        cv2.line(canvas, center, pt2, color, 2)
        
        if depth_val > 0: # Pointing OUT
            cv2.circle(canvas, pt2, 4, color, -1)
            cv2.circle(canvas, pt2, 5, (0,0,0), 1)
        else: # Pointing IN
            cv2.circle(canvas, pt2, 4, (0,0,0), -1)
            cv2.circle(canvas, pt2, 4, color, 2)

    cv2.putText(canvas, label, (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
    cv2.putText(canvas, title, (x+5, y+s-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    g_col = (0, 255, 0) if grip > 0.5 else (0, 0, 255)
    cv2.circle(canvas, (x+s-10, y+10), 4, g_col, -1)

def main(root, ep, cam):
    ds_id = "droid"
    path = Path(root)
    vid_path, _ = find_video_file(path, ep, cam)
    if not vid_path: return print("Video not found")
    
    print(root)
    ds = LeRobotDataset(root=path, repo_id="", episodes=[ep])
    
    cap = cv2.VideoCapture(str(vid_path))
    vw, vh = int(cap.get(3)), int(cap.get(4))
    scale = DISPLAY_IMG_H / vh
    dw = int(vw * scale)
    
    HUD_COL_W = HUD_SIZE + PADDING
    W = dw + 2*HUD_COL_W + 40
    H = DISPLAY_IMG_H + 40
    
    out_name = f"check_ep{ep}.mp4"
    writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (W, H))
    transform_mat = get_transform_matrix(ds_id)
    
    print(f"Saving to {out_name}...")
    frames_total = min(len(ds), int(cap.get(7)))
    
    for i in range(frames_total):
        ret, frame = cap.read()
        if not ret: break
        
        item = ds[i]
        canvas = np.full((H, W, 3), 20, dtype=np.uint8)
        
        img = cv2.resize(frame, (dw, DISPLAY_IMG_H))
        canvas[20:20+DISPLAY_IMG_H, 20:20+dw] = img
        
        l_w = to_numpy(item.get("eef.left.wrist"))
        r_w = to_numpy(item.get("eef.right.wrist"))
        l_h = to_numpy(item.get("eef.left.hand", 0))
        r_h = to_numpy(item.get("eef.right.hand", 0))
        if isinstance(l_h, np.ndarray): l_h = l_h.item()
        if isinstance(r_h, np.ndarray): r_h = r_h.item()

        # --- KEY CHANGE: Get Matrix, NOT Pose ---
        l_mat = apply_transform_to_matrix(l_w, transform_mat)
        r_mat = apply_transform_to_matrix(r_w, transform_mat)
        
        # --- HUDs (Pass Matrix) ---
        lx = 20 + dw + 10
        draw_hud(canvas, (lx, 20, HUD_SIZE, HUD_SIZE), l_mat, l_h, "LEFT - Top", "TOP")
        draw_hud(canvas, (lx, 20+HUD_SIZE+10, HUD_SIZE, HUD_SIZE), l_mat, l_h, "LEFT - Front", "FRONT")

        rx = lx + HUD_SIZE + 10
        draw_hud(canvas, (rx, 20, HUD_SIZE, HUD_SIZE), r_mat, r_h, "RIGHT - Top", "TOP")
        draw_hud(canvas, (rx, 20+HUD_SIZE+10, HUD_SIZE, HUD_SIZE), r_mat, r_h, "RIGHT - Front", "FRONT")

        cv2.putText(canvas, "SOLID Tip = OUT", (lx, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(canvas, "HOLLOW Tip = IN", (rx, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        writer.write(canvas)
        if i % 20 == 0: print(f"Frame {i}/{frames_total}", end="\r")

    writer.release()
    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/home/lvjiangran/zhuwenxuan/let/results")
    parser.add_argument("--ep", type=int, default=0)
    parser.add_argument("--cam", type=str, default="top_head")
    args = parser.parse_args()
    main(args.root, args.ep, args.cam)