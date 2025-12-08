"""
Usage:
  - run: python factory.py \
        --input_root /path/to/factory/dataset \
        --output_root /path/to/save/LeRobot_v2.1_dataset \
        --fps fps
"""

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
from datasets import Dataset, DatasetDict
from dataclasses import dataclass, asdict

# ===================== constants ======================
ROBOT_TYPE = "human_hand" 
CODEBASE_VERSION = "v2.1"

# ====================== helper functions ======================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def symlink_or_copy(src: Path, dst: Path):
    try:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except Exception:
        # fallback to copy
        shutil.copy2(src, dst)

def load_json(file_path: Path) -> Any:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Any, file_path: Path):
    with open(file_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

# ===================== main processing class ======================
@dataclass
class LeRobotDatasetMetadata:
    info: Dict
    episodes: List
    episode_stats: List
    tasks: List

class FactoryDatasetConverter:
    def __init__(self, input_root: Path, output_root: Path, fps: int, chunk_size: int = 1000):
        self.input_root = input_root # 数据集根目录
        self.output_root = output_root
        self.fps = fps
        self.chunk_size = chunk_size  
        self.metadata = None
        self.num_frames = 0
        self.num_episodes = 0

    def process_episodes(self):
        # 已知self.input_path是数据集根目录，子目录形如factory_001/workers/worker_001/factory001_worker001_00000.mp4，要遍历所有的episodes
        episode_dirs = list(self.input_root.glob("factory_*/workers/worker_*/factory*_worker*_*.json"))
        episode_dirs.sort()
        h, w = 1080, 1920  
        codec = "h264"
        for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
            h, w, codec = self.process_single_episode(self.input_root/episode_dir, self.num_episodes)
            self.num_episodes += 1
        return h, w, codec

    def process_single_episode(self, path: Path, episode_id: int):
        chunk_id = episode_id // self.chunk_size
        # 现在path后缀是json，要删除json，把后缀变成mp4才是视频路径
        episode_input_path = path.parent
        video_path = path.with_suffix('.mp4')
        print(f"video path is {video_path}")
        video_output_path = self.output_root / "videos" / f"chunk-{chunk_id:03d}" / "observation.images.top_head" /  f"episode_{episode_id:06d}.mp4"
        data_output_dir = self.output_root / "data" / f"chunk-{chunk_id:03d}"
        data_output_path = self.output_root / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_id:06d}.parquet"
        ensure_dir(video_output_path.parent)
        symlink_or_copy(video_path, video_output_path) # 复制视频文件
        arrays = {}
        fields = []    
        video_info = load_json(path)
        h, w = video_info["height"], video_info["width"]
        codec = video_info["codec"]
        frame_count = int(self.fps * video_info["duration_sec"])
        arrays["observation.images.top_head"] = pa.array([str(video_output_path)]*frame_count, type=pa.string())
        fields.append(pa.field("observation.images.top_head", pa.string()))
        arrays["observation.state"] = pa.array([[]]*frame_count, type=pa.list_(pa.float32())); fields.append(pa.field("observation.state", pa.list_(pa.float32())))
        arrays["action"] = pa.array([[]]*frame_count, type=pa.list_(pa.float32())); fields.append(pa.field("action", pa.list_(pa.float32())))
        # 补充定义
        arrays["camera.intrinsic"] = pa.array([[[0,0,0],[0,0,0],[0,0,0]]]*frame_count, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("camera.intrinsic", pa.list_(pa.list_(pa.float32()))))
        arrays["camera.extrinsic"] = pa.array([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]]*frame_count, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("camera.extrinsic", pa.list_(pa.list_(pa.float32()))))
        arrays["eef.left.hand"] = pa.array([[[0,0,0]]*21]*frame_count, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("eef.left.hand", pa.list_(pa.list_(pa.float32()))))
        arrays["eef.right.hand"] = pa.array([[[0,0,0]]*21]*frame_count, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("eef.right.hand", pa.list_(pa.list_(pa.float32()))))
        arrays["eef.left.wrist"] = pa.array([[0,0,0,0,0,0]]*frame_count, type=pa.list_(pa.float32())); fields.append(pa.field("eef.left.wrist", pa.list_(pa.float32())))
        arrays["eef.right.wrist"] = pa.array([[0,0,0,0,0,0]]*frame_count, type=pa.list_(pa.float32())); fields.append(pa.field("eef.right.wrist", pa.list_(pa.float32())))
        arrays["annotation.language.action_text"] = pa.array([0]*frame_count, type=pa.int64()); fields.append(pa.field("annotation.language.action_text", pa.int64()))
        arrays["annotation.language.task_text"] = pa.array([0]*frame_count, type=pa.int64()); fields.append(pa.field("annotation.language.task_text", pa.int64()))
        # 补充定义结束
        timestamps = [i/self.fps for i in range(frame_count)]
        frame_indices = [i for i in range(frame_count)]
        episode_indices = [episode_id]*frame_count
        global_indices = [i+self.num_frames for i in range(frame_count)]
        arrays["task_index"] = pa.array(episode_indices, type=pa.int64()); fields.append(pa.field("task_index", pa.int64()))
        arrays["episode_index"] = pa.array(episode_indices, type=pa.int64()); fields.append(pa.field("episode_index", pa.int64()))
        arrays["frame_index"] = pa.array(frame_indices, type=pa.int64()); fields.append(pa.field("frame_index", pa.int64()))
        arrays["timestamp"] = pa.array(timestamps, type=pa.float32()); fields.append(pa.field("timestamp", pa.float32()))
        arrays["next.done"] = pa.array([False]*(frame_count-1)+[True], type=pa.bool_()); fields.append(pa.field("next.done", pa.bool_()))
        arrays["index"] = pa.array(global_indices, type=pa.int64()); fields.append(pa.field("index", pa.int64()))

        schema = pa.schema(fields)
        table = pa.Table.from_arrays([arrays[k] for k in arrays.keys()], schema=schema)
        ensure_dir(data_output_dir)
        pq.write_table(table, data_output_path)
        self.num_frames += frame_count

        ep_stats = {
            "episode_index": episode_id,
            "stats": {
                "action":{
                    "min": [],
                    "max": [],
                    "mean": [],
                    "std": [],
                    "count": [frame_count]
                },
                "observation.state": {
                    "min": [],
                    "max": [],
                    "mean": [],
                    "std": [],
                    "count": [frame_count]
                },
                "timestamp": {
                    "min": [min(timestamps)],
                    "max": [max(timestamps)],
                    "mean": [np.mean(timestamps)],
                    "std": [np.std(timestamps)],
                    "count": [frame_count]
                },
                "frame_index": {
                    "min": [min(frame_indices)],
                    "max": [max(frame_indices)],
                    "mean": [np.mean(frame_indices)],
                    "std": [np.std(frame_indices)],
                    "count": [frame_count]
                },
                "episode_index": {
                    "min": [min(episode_indices)],
                    "max": [max(episode_indices)],
                    "mean": [np.mean(episode_indices)],
                    "std": [np.std(episode_indices)],
                    "count": [frame_count]
                },
                "index": {
                    "min": [min(global_indices)],
                    "max": [max(global_indices)],
                    "mean": [np.mean(global_indices)],
                    "std": [np.std(global_indices)],
                    "count": [frame_count]
                },
                "task_index": {
                    "min": [0],
                    "max": [0],
                    "mean": [0.0],
                    "std": [0.0],
                    "count": [frame_count]
                }
            }
        }
        ep = {
            "episode_index": episode_id,
            "length": frame_count,
            "dataset_from_index": self.num_frames - frame_count,
            "dataset_to_index": self.num_frames - 1,
            "tasks": [""],
            "parquet_path": str(data_output_path.resolve()),   
        }
        self.metadata.episodes.append(ep)
        self.metadata.episode_stats.append(ep_stats)
        self.metadata.tasks.append({
            "task_index": episode_id,
            "task": "",
            "task_description": ""
        })
        return h, w, codec

    def save_metadata(self, h, w, codec):

        self.metadata.info["codebase_version"] = CODEBASE_VERSION
        self.metadata.info["fps"] = self.fps
        self.metadata.info["robot_type"] = ROBOT_TYPE
        self.metadata.info["total_episodes"] = self.num_episodes
        self.metadata.info["total_tasks"] = self.num_episodes
        self.metadata.info["total_videos"] = self.num_episodes
        self.metadata.info["total_frames"] = self.num_frames
        self.metadata.info["total_episodes"] = self.num_episodes
        self.metadata.info["total_tasks"] = len(self.metadata.tasks)
        self.metadata.info["chunks_size"] = self.chunk_size
        self.metadata.info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        self.metadata.info["video_path"] = "videos/chunk-{episode_chunk:03d}/observation.images.top_head/episode_{episode_index:06d}.mp4"
        self.metadata.info["splits"] = {
            "train": "0:10000000000", 
        }

        # deal with features
        features = {}
        features["observation.images.top_head"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "info": {
                "video.fps": 30,
                "video.height": h,
                "video.width": w,
                "video.channels": 3,
                "video.codec": codec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
        features["observation.state"] = {"dtype": "list[float32]", "shape": [None]} 
        features["action"] = {"dtype": "list[float32]", "shape": [None]}
        # 补充定义
        features["camera.intrinsic"] = {"dtype": "list[list[float32]]", "shape": [3,3]}
        features["camera.extrinsic"] = {"dtype": "list[list[float32]]", "shape": [4,4]}
        features["eef.left.hand"] = {"dtype": "list[list[float32]]", "shape": [21,3]} 
        features["eef.right.hand"] = {"dtype": "list[list[float32]]", "shape": [21,3]}
        features["eef.left.wrist"] = {"dtype": "list[float32]", "shape": [6]}
        features["eef.right.wrist"] = {"dtype": "list[float32]", "shape": [6]}
        features["annotation.language.action_text"] = {"dtype": "int64", "shape": [1]}
        features["annotation.language.task_text"] = {"dtype": "int64", "shape": [1]}
        # 补充定义结束
        features["episode_index"] = {"dtype": "int64", "shape": [1]}
        features["frame_index"] = {"dtype": "int64", "shape": [1]}
        features["timestamp"] = {"dtype": "float32", "shape": [1]}
        features["next.done"] = {"dtype": "bool", "shape": [1]}
        features["index"] = {"dtype": "int64", "shape": [1]}
        self.metadata.info["features"] = features

        metadata_output_path = self.output_root / "meta"
        ensure_dir(metadata_output_path)
        with open(metadata_output_path / "info.json", 'w') as f:
            json.dump(self.metadata.info, f, indent=2)
        save_json(self.metadata.episode_stats, metadata_output_path / "episodes_stats.jsonl")
        save_json(self.metadata.episodes, metadata_output_path / "episodes.jsonl")
        save_json(self.metadata.tasks, metadata_output_path / "tasks.jsonl")
        print("conversion finished.\n")

    def convert(self):
        ensure_dir(self.output_root)
        h, w, codec = self.process_episodes()
        self.save_metadata(h, w, codec)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Factory dataset to LeRobot v2.1 format")
    parser.add_argument('--input_root', type=str, required=True, help='Path to the input Factory dataset root directory')
    parser.add_argument('--output_root', type=str, required=True, help='Path to save the converted LeRobot dataset')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the videos')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Number of episodes per chunk')
    return parser.parse_args()

def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    fps = args.fps
    chunk_size = args.chunk_size

    converter = FactoryDatasetConverter(input_root, output_root, fps, chunk_size)
    converter.metadata = LeRobotDatasetMetadata(info={}, episodes=[], episode_stats=[], tasks=[])
    converter.convert()

if __name__ == "__main__":
    main()