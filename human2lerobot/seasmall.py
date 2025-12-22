"""
File: seasmall.py
Function: Convert SEA-Small format to LeRobot format
"""

import struct
import shutil
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import warnings
import argparse 

warnings.filterwarnings("ignore")

class SeaBinaryLoader:
    @staticmethod
    def load_frames(frames_path: Path) -> List[Tuple[int, np.ndarray]]:
        frames = []
        header_size = 8 + 8 + 4
        if not frames_path.exists(): return []
        with open(frames_path, "rb") as f:
            while True:
                header = f.read(header_size)
                if len(header) < header_size: break
                ts, _, size = struct.unpack(">qqi", header)
                data = f.read(size)
                if len(data) != size: break
                arr = np.frombuffer(data, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    frames.append((ts, img_rgb))
        return frames

    @staticmethod
    def load_trajectory(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_list = []
        buf_size = 8 + 7 * 4
        if not path.exists(): return np.array([]), np.array([]), np.array([])
        with open(path, "rb") as f:
            while True:
                buf = f.read(buf_size)
                if len(buf) < buf_size: break
                ts = struct.unpack("<q", buf[:8])[0]
                vals = struct.unpack("<7f", buf[8:])
                data_list.append((ts, vals[:3], vals[3:]))
        if not data_list: return np.array([]), np.array([]), np.array([])
        ts_arr = np.array([x[0] for x in data_list])
        pos_arr = np.array([x[1] for x in data_list])
        quat_arr = np.array([x[2] for x in data_list])
        return ts_arr, pos_arr, quat_arr

    @staticmethod
    def load_hand_data(path: Path):
        frames = []
        if not path.exists(): return []
        with open(path, "rb") as f:
            while True:
                ts_buf = f.read(8)
                if len(ts_buf) < 8: break
                ts = struct.unpack("<q", ts_buf)[0]
                def read_kp():
                    cnt_buf = f.read(4)
                    if len(cnt_buf) < 4: return None
                    cnt = struct.unpack("<i", cnt_buf)[0]
                    if cnt <= 0: return None
                    bytes_needed = cnt * 7 * 4
                    raw = f.read(bytes_needed)
                    if len(raw) != bytes_needed: return None
                    floats = struct.unpack("<" + "f" * (cnt * 7), raw)
                    return np.array(floats).reshape(cnt, 7)
                left_kps = read_kp()
                right_kps = read_kp()
                frames.append((ts, left_kps, right_kps))
        return frames

    @staticmethod
    def load_intrinsics(path: Path):
        K = np.eye(3, dtype=np.float32)
        if path.exists():
            try:
                parts = path.read_text().split()
                if len(parts) >= 6:
                    fx, fy = float(parts[2]), float(parts[3])
                    cx, cy = float(parts[4]), float(parts[5])
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            except: pass
        return K

class SeaSmallConverter:
    def __init__(self, raw_dir: Path, output_dir: Path, repo_id: str, fps: int = 10):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.fps = fps
        self.mano_indices = [1] + list(range(2, 6)) + list(range(7, 11)) + list(range(12, 16)) + list(range(17, 21)) + list(range(22, 26))
        
        self.features = {
            "eef.left.wrist": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "eef.right.wrist": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "roll", "pitch", "yaw"]},
            "eef.left.hand": {"dtype": "float32", "shape": (63,)},
            "eef.right.hand": {"dtype": "float32", "shape": (63,)},
            
            "eef.left.hand_raw": {"dtype": "float32", "shape": (189,)},
            "eef.right.hand_raw": {"dtype": "float32", "shape": (189,)},
            "camera.intrinsic": {"dtype": "float32", "shape": (3, 3)},
            "camera.extrinsic": {"dtype": "float32", "shape": (4, 4)},
            "annotation.language.instruction": {"dtype": "string", "shape": (1,)},
            "annotation.language.task_index": {"dtype": "int64", "shape": (1,)},
            "observation.images.top_head": {"dtype": "video", "shape": (3, 960, 1280), "names": ["channel", "height", "width"]},
        }

    def process(self):
        episode_paths = sorted([p for p in self.raw_dir.iterdir() if p.is_dir()])
        print(f"Found {len(episode_paths)} episodes. Starting conversion...")
        dataset = LeRobotDataset.create(repo_id=self.repo_id, fps=self.fps, root=self.output_dir, features=self.features, image_writer_processes=10, image_writer_threads=2)
        try:
            for ep_idx, ep_path in enumerate(tqdm(episode_paths, desc="Processing Episodes")):
                try:
                    self._process_single_episode(dataset, ep_path, ep_idx)
                except Exception as e:
                    print(f"\nSkipping episode {ep_path.name} due to error: {e}")
                    traceback.print_exc()
        finally:
            print("Finalizing dataset...")
            if hasattr(dataset, "finalize"): dataset.finalize()
            else: print("Dataset object finalized.")

    def _process_single_episode(self, dataset, ep_path: Path, ep_idx: int):
        paths = {
            'img': ep_path / "stereo" / "left_frames.dat",
            'hand': ep_path / "hand_data.bin",
            'traj': ep_path / "stereo" / "left_trajectory.bin",
            'intr': ep_path / "stereo" / "left_intrinsics.txt"
        }
        raw_frames = SeaBinaryLoader.load_frames(paths['img'])
        if len(raw_frames) < 10: return
        raw_hands = SeaBinaryLoader.load_hand_data(paths['hand'])
        traj_ts, traj_pos, traj_quat = SeaBinaryLoader.load_trajectory(paths['traj'])
        K = SeaBinaryLoader.load_intrinsics(paths['intr'])

        t_start_raw = raw_frames[0][0]
        time_divisor = 1e9 if t_start_raw > 1e16 else 1e3
        if not raw_hands: return
        hand_ts_arr = np.array([x[0] for x in raw_hands])
        duration_sec = (raw_frames[-1][0] - raw_frames[0][0]) / time_divisor
        src_fps = len(raw_frames) / duration_sec
        step = max(1, int(round(src_fps / self.fps)))
        
        raw_indices = list(range(0, len(raw_frames), step))
        valid_indices = []
        for idx in raw_indices:
            img_ts = raw_frames[idx][0]
            h_idx = np.searchsorted(hand_ts_arr, img_ts)
            if h_idx >= len(hand_ts_arr): h_idx = len(hand_ts_arr) - 1
            if abs(img_ts - hand_ts_arr[h_idx]) / time_divisor > 0.05: continue
            _, l_kps, r_kps = raw_hands[h_idx]
            if (l_kps is not None and len(l_kps) > 0) or (r_kps is not None and len(r_kps) > 0):
                valid_indices.append(idx)
        if not valid_indices: return

        FIX_CAM_MAT = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0], 
            [0,  0,  1, 0], 
            [0,  0,  0, 1]
        ], dtype=np.float32)

        for idx in valid_indices:
            img_ts, img = raw_frames[idx]
            h_idx = np.searchsorted(hand_ts_arr, img_ts)
            if h_idx >= len(hand_ts_arr): h_idx = len(hand_ts_arr) - 1
            _, left_kps_raw, right_kps_raw = raw_hands[h_idx]

            extrinsic = np.eye(4, dtype=np.float32)
            if len(traj_ts) > 0:
                t_idx = np.searchsorted(traj_ts, img_ts)
                if t_idx >= len(traj_ts): t_idx = len(traj_ts) - 1
                pos_c, quat_c = traj_pos[t_idx], traj_quat[t_idx]
                T_c2w = np.eye(4, dtype=np.float32)
                T_c2w[:3, :3] = R.from_quat(quat_c).as_matrix()
                T_c2w[:3, 3] = pos_c
                extrinsic = FIX_CAM_MAT @ np.linalg.inv(T_c2w)

            l_wrist, l_hand, l_raw = self._process_hand_side(left_kps_raw, "left")
            r_wrist, r_hand, r_raw = self._process_hand_side(right_kps_raw, "right")

            frame_dict = {
                "observation.images.top_head": img, 
                "annotation.language.instruction": "Interact with objects", 
                "annotation.language.task_index": np.array([0], dtype=np.int64),
                "camera.intrinsic": K.astype(np.float32),
                "camera.extrinsic": extrinsic.astype(np.float32),
                "eef.left.wrist": l_wrist,
                "eef.left.hand": l_hand, 
                "eef.left.hand_raw": l_raw,
                "eef.right.wrist": r_wrist,
                "eef.right.hand": r_hand, 
                "eef.right.hand_raw": r_raw,
            }
            dataset.add_frame(frame_dict, task="Interact with objects")
        dataset.save_episode()

    def _process_hand_side(self, kps_raw, side_name):
        wrist_6d = np.zeros(6, dtype=np.float32)
        hand_21 = np.zeros((21, 3), dtype=np.float32)
        hand_raw = np.zeros(189, dtype=np.float32) 
        
        if kps_raw is not None and len(kps_raw) > 0:
            raw_flat = kps_raw.flatten()
            valid_len = min(len(raw_flat), 189)
            hand_raw[:valid_len] = raw_flat[:valid_len].astype(np.float32)
            
            if kps_raw.shape[0] > 1:
                wrist_pos = kps_raw[1, :3]
                wrist_quat = kps_raw[1, 3:] 
                wrist_rpy = R.from_quat(wrist_quat).as_euler('xyz')
                wrist_6d = np.concatenate([wrist_pos, wrist_rpy]).astype(np.float32)
            
            valid_indices = [i for i in self.mano_indices if i < kps_raw.shape[0]]
            if valid_indices:
                extracted = kps_raw[valid_indices, :3]
                count = extracted.shape[0]
                if count == 21:
                    hand_21 = extracted.astype(np.float32)
                else:
                    hand_21[:count] = extracted.astype(np.float32)
        
        return wrist_6d, hand_21.flatten(), hand_raw

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SEA-Small dataset to LeRobot format.")
    
    parser.add_argument(
        "--raw-dir", 
        type=Path, 
        required=True, 
        help="Path to the raw sea-small dataset folder (containing episode folders)."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        required=True, 
        help="Path where the output LeRobot dataset will be saved."
    )
    parser.add_argument(
        "--repo-id", 
        type=str, 
        default="local/sea-small-converted",
        help="HuggingFace repo ID (default: local/sea-small-converted)."
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10, 
        help="Target FPS for the dataset (default: 10)."
    )

    args = parser.parse_args()

    if args.output_dir.exists():
        print(f"⚠️  Cleaning up old output directory: {args.output_dir}")
        try:
            shutil.rmtree(args.output_dir)
        except Exception as e:
            print(f"Error removing directory: {e}")
            exit(1)
            
    converter = SeaSmallConverter(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps
    )
    
    converter.process()
    print("🎉 Conversion Complete!")