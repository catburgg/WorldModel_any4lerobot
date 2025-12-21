import cv2
import numpy as np
import tqdm
import shutil
import json
import torch
import argparse
import concurrent.futures
import re
import os
import multiprocessing
import queue
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

def get_video_frames(video_file, target_fps):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Failed to open {video_file}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps <= 0:
        return None
    
    duration = frame_count / fps
    num_frames = int(duration * target_fps)
    
    # 用 lerobot 提供的工具，按照时间戳提取帧。返回形状 (T, C, H, W) ，值被归一化到 float32 [0, 1]
    timestamps = [t / target_fps for t in range(0, num_frames)]
    frames = decode_video_frames(video_file, timestamps, tolerance_s=1.0 / target_fps)
        
    # 转回 uint8 [0, 255]
    frames = (frames * 255).to(torch.uint8)
    for j, frame in enumerate(frames):
        yield frame, timestamps[j]

def process_action_group(action, videos, input_root, output_root, fps, repo_id, writer_processes, writer_threads, progress_queue=None):
    # Sanitize action name for directory
    safe_action = re.sub(r'[^a-zA-Z0-9]', '_', action)
    dataset_root = output_root / safe_action
    action_repo_id = f"{repo_id}_{safe_action}"
    
    features = {
        "eef.left.wrist": {
            "dtype": "float32",
            "shape": (6,),
        },
        "eef.right.wrist": {
            "dtype": "float32",
            "shape": (6,),
        },
        "eef.left.hand": {
            "dtype": "float32",
            "shape": (21, 3),
        },
        "eef.right.hand": {
            "dtype": "float32",
            "shape": (21, 3),
        },
        "camera.intrinsic": {
            "dtype": "float32",
            "shape": (3, 3),
        },
        "camera.extrinsic": {
            "dtype": "float32",
            "shape": (4, 4),
        },
        "observation.images.top_head": {
            "dtype": "video",
            "shape": (3, 960, 1280),
            "names": ["channel", "height", "width"],
        },
    }

    # if dataset_root.exists():
    #     shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=action_repo_id,
        fps=fps,
        robot_type="dex",
        features=features,
        root=dataset_root,
        use_videos=True,
        image_writer_processes=writer_processes,
        image_writer_threads=writer_threads
    )

    for video_id, meta in videos:
        try:
            video_file = input_root / "Video" / video_id
            if not video_file.exists():
                continue
            
            frames_gen = get_video_frames(video_file, target_fps=fps)
            if frames_gen is None:
                continue

            fine_steps = meta.get("Fine-level", [])

            for frame_img, timestamp in frames_gen:
                fine_text = ""
                for step in fine_steps:
                    desc = step.get("Step discription", "")
                    ts = step.get("Step timestamp", [0, 0])
                    if ts and len(ts) >= 2:
                        start, end = ts[0], ts[1]
                        if start <= timestamp <= end:
                            fine_text = desc
                            break

                frame = {
                    "eef.left.wrist": np.zeros((6,), dtype=np.float32),
                    "eef.right.wrist": np.zeros((6,), dtype=np.float32),
                    "eef.left.hand": np.zeros((21, 3), dtype=np.float32),
                    "eef.right.hand": np.zeros((21, 3), dtype=np.float32),
                    "camera.intrinsic": np.zeros((3, 3), dtype=np.float32),
                    "camera.extrinsic": np.zeros((4, 4), dtype=np.float32),
                    "observation.images.top_head": frame_img,
                }
                dataset.add_frame(frame, task=fine_text)

            dataset.save_episode()
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
        finally:
            if progress_queue is not None:
                progress_queue.put(1)
    
    return f"Finished action: {action}"

class EgoMe2LeRobot:
    def __init__(self, input_root: Path, output_root: Path, fps: int, repo_id: str, num_cpus: int):
        self.input_root = input_root
        self.output_root = output_root
        self.fps = fps
        self.repo_id = repo_id
        self.num_cpus = num_cpus

    def load_annotations(self):
        ann_path = self.input_root / "Annotation"
        
        p = ann_path / "total.json"
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "annotations" in data:
                    return data["annotations"]
                return data
        
        return {}
            
    def convert(self):
        # 加载标注
        annotations = self.load_annotations()
        # 筛选 Ego 视角视频
        ego_videos = {k: v for k, v in annotations.items() if isinstance(v, dict) and v.get("View") == "Ego"}
        print(f"Found {len(ego_videos)} ego videos.")

        # Group by Action
        action_groups = {}
        for video_id, meta in ego_videos.items():
            action = meta.get("Action", "Unknown")
            if action not in action_groups:
                action_groups[action] = []
            action_groups[action].append((video_id, meta))
            
        print(f"Found {len(action_groups)} unique actions.")

        # Calculate workers
        # Each worker uses 1 process for itself + writer_processes
        writer_processes = 2
        writer_threads = 2
        processes_per_worker = 1 + writer_processes
        
        max_workers = max(1, self.num_cpus // processes_per_worker)
        
        print(f"Total CPUs available: {self.num_cpus}")
        print(f"Allocating {max_workers} worker processes for actions.")
        print(f"Each worker will use {writer_processes} subprocesses for image writing.")

        # Multiprocessing
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        
        total_videos = sum(len(v) for v in action_groups.values())
        print(f"Total videos to process: {total_videos}")

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        try:
            futures = []
            for action, videos in action_groups.items():
                futures.append(executor.submit(
                    process_action_group, 
                    action, 
                    videos, 
                    self.input_root, 
                    self.output_root, 
                    self.fps, 
                    self.repo_id,
                    writer_processes,
                    writer_threads,
                    progress_queue
                ))
            
            # Monitor progress
            with tqdm.tqdm(total=total_videos, desc="Processing Videos") as pbar:
                finished_videos = 0
                while finished_videos < total_videos:
                    # Check if all futures are done (in case of errors/early exit)
                    if all(f.done() for f in futures):
                        # If all done but count mismatch, break to avoid hang
                        if progress_queue.empty():
                            break
                    
                    try:
                        # Wait for progress update
                        _ = progress_queue.get(timeout=1.0)
                        pbar.update(1)
                        finished_videos += 1
                    except queue.Empty:
                        continue
            
            # Check for exceptions in futures
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Task failed: {e}")

        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt. Terminating workers...")
            # Force kill workers
            if hasattr(executor, '_processes'):
                for p in executor._processes.values():
                    p.terminate()
            executor.shutdown(wait=False)
            print("Workers terminated.")
            raise
        finally:
            executor.shutdown(wait=True)
            manager.shutdown()
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to the EgoMe dataset root directory."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to save the converted LeRobot dataset."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate for the LeRobot dataset."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="user/EgoMe2LeRobot_dataset",
        help="Huggingface repo ID for the LeRobot dataset."
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=16,
        help="Total number of CPUs to use."
    )
    args = parser.parse_args()
    return args.input_root, args.output_root, args.fps, args.repo_id, args.num_cpus

def main():
    input_root, output_root, fps, repo_id, num_cpus = parse_args()
    input_root = Path(input_root)
    output_root = Path(output_root)

    # 删除已有的数据集，应当只在测试时保留
    # if output_root.exists():
    #     shutil.rmtree(output_root)
    #     print(f"Removed existing dataset at {output_root}")
    # output_root.mkdir(parents=True, exist_ok=True)
        
    converter = EgoMe2LeRobot(
        input_root=input_root,
        output_root=output_root,
        fps=fps,
        repo_id=repo_id,
        num_cpus=num_cpus
    )
    converter.convert()
    
if __name__ == "__main__":
    main()
