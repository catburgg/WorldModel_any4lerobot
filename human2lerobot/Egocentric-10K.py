import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import cv2
import argparse
import importlib
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames

# -----------------------------
# Helpers
# -----------------------------
def safe_load_pickle(p: Path):
    with open(p, 'rb') as f:
        return pickle.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def symlink_or_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)

# -----------------------------
# Main Conversion Class
# -----------------------------

class Egocentric2LeRobot:
    def __init__(self, input_root: Path, output_root: Path, fps: int, repo_id: str):
        self.features = {
            "eef.left.wrist": {
                "dtype": "float32",
                "shape": (6,),
            },
            "eef.right.wrist": {
                "dtype": "float32",
                "shape": (6,),
            },
            "camera.intrinsic": {
                "dtype": "float32",
                "shape": (9, ),
            },
            "camera.extrinsic": {
                "dtype": "float32",
                "shape": (16, ),
            },
            "observation.images.top_head": {
                "dtype": "video",
                "shape": [3, 1080, 1920],
                "names": ["channel", "height", "width"],
                "info": {
                    "video.fps": 30,
                    "video.height": 1080,
                    "video.width": 1920,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },    
        }
        self.input_root = input_root
        self.output_root = output_root
        self.fps = fps
        self.repo_id = repo_id

    def create_lerobot_dataset(self) -> LeRobotDataset:
        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type="human_hand",
            features=self.features,
            root=self.output_root,
            use_videos=True,
            tolerance_s=0.0001,
            image_writer_processes=10,
            image_writer_threads=5,
            video_backend="ffmpeg",
        )
        return dataset
    
    def convert(self):
        egocentric_root = self.input_root
        output_root = self.output_root

        dataset = self.create_lerobot_dataset()

        # Iterate over each episode
        '''
        directory structure: 
        root/factory_001/workers/worker_001/factory001_worker001_00000.mp4
        '''
        episodes = list(egocentric_root.glob("*/workers/*/*.mp4"))
        # only for testing!
        # episodes = episodes[:10]
        # ends
        for ep_dir in tqdm(episodes, desc="Converting Egocentric-10K to LeRobot"):
            video_path = ep_dir
            print(f"Processing video: {video_path}")
            # Process single dataset
            dataset = self.process_single_episode(
                dataset,
                video_path,
                triplet_name
            )

        print(f"Saving converted dataset to {output_root}")

    def process_single_episode(
        self,
        dataset: LeRobotDataset,
        video_path: Path,
        triplet_name: str,
    ) -> LeRobotDataset:

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            return
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"Processing episode {triplet_name} with {num_frames} frames.")
        extract = 30 // self.fps
        if 30 % self.fps != 0:
            print(f"Warning: fps {self.fps} does not divide 30 evenly, some frames may be dropped.")

        def get_frames(video_file: Path) -> torch.Tensor:
            timestamps = [t/30 for t in range(0, num_frames, extract)]
            # 用 lerobot 提供的工具，按照时间戳提取帧。返回形状 (T, C, H, W) ，值被归一化到 float32 [0, 1]
            frames = decode_video_frames(video_file, timestamps, tolerance_s=1.0/self.fps)
            # 转回 uint8 [0, 255]
            frames = (frames*255).to(torch.uint8)
            return frames

        frames = get_frames(video_path)

        for i in range(0, num_frames, extract):
            frame = {
                "eef.left.wrist": np.zeros((6,), dtype=np.float32),  # Placeholder
                "eef.right.wrist": np.zeros((6,), dtype=np.float32),  # Placeholder
                "camera.intrinsic": np.zeros((9, ), dtype=np.float32),  # Placeholder
                "camera.extrinsic": np.zeros((16, ), dtype=np.float32),  # Placeholder
                "observation.images.top_head": frames[i // extract].cpu().numpy(),  # (C, H, W) numpy array
            }
            dataset.add_frame(frame, task=f"work in the factory")

        dataset.save_episode()

        return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Factory dataset to LeRobot v2.1 format")
    parser.add_argument('--input_root', type=str, required=True, help='Path to the input Factory dataset root directory')
    parser.add_argument('--output_root', type=str, required=True, help='Path to save the converted LeRobot dataset')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the videos')
    return parser.parse_args()

def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    fps = args.fps

    converter = Egocentric2LeRobot(input_root, output_root, fps, repo_id="lerobot/world-model-dataset-egocentric-10k")
    converter.convert()

if __name__ == "__main__":
    main()