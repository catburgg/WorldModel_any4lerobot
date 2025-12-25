import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pickle
from tqdm import tqdm
import torch
import cv2
from scipy.spatial.transform import Rotation as R
import argparse
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
        # fallback to copy
        shutil.copy2(src, dst)

def list_video_files(dirpath: Path) -> Dict[str, str]:
    """Return mapping cam_key -> absolute path string for .mp4 files in dirpath"""
    res = {}
    if not dirpath.exists():
        return res
    for fp in dirpath.glob("*.mp4"):
        res[fp.stem] = str(fp.resolve())
    return res

# -----------------------------
# Main Conversion Class
# -----------------------------

class Taco2LeRobot:
    def __init__(self, input_root: Path, output_root: Path, fps: int, repo_id: str):
        self.features = {
            "eef.hand_raw": {
                "dtype": "float32",
                "shape": (122,),
            },
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
                "shape": (3, 3),
            },
            "camera.extrinsic": {
                "dtype": "float32",
                "shape": (4, 4),
            },
            "annotation.language.action_text": {
                "dtype": "int64",
                "shape": (1,),
            },
            "annotation.language.task_text": {
                "dtype": "string",
                "shape": (1,),
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
        taco_root = self.input_root
        output_root = self.output_root

        dataset = self.create_lerobot_dataset()

        # Iterate over each episode in TACO
        '''
        directory structure: 
        taco_root/Egocentric_RGB_Videos/triplet_name/sequence_name/color.mp4
        taco_root/Hand_Poses/triplet_name/sequence_name/left_hand.pkl or left_hand_shape.pkl
        taco_root/Egocentric_Camera_Parameters/triplet_name/sequence_name/egocentric_intrinsic.txt or egocentric_frame_extrinsic.npy
        '''
        video_root = taco_root / "Egocentric_RGB_Videos"
        hand_pose_root = taco_root / "Hand_Poses"
        camera_param_root = taco_root / "Egocentric_Camera_Parameters"
        # 我现在要找到video root下面所有的triplet_name/sequence_name目录
        episodes = list(video_root.glob("*/*"))
        # only for testing!
        # episodes = episodes[:10]
        # ends
        for ep_dir in tqdm(episodes, desc="Converting TACO to LeRobot"):
            # Load data: all we need is camera_intrinsics, camera_extrinsics, left/right_hand_shape.pkl, left/right_hand.pkl, videos
            triplet_name = ep_dir.parent.name
            sequence_name = ep_dir.name
            video_path = video_root / triplet_name / sequence_name / "color.mp4"
            left_hand_path = hand_pose_root / triplet_name / sequence_name / "left_hand.pkl"
            right_hand_path = hand_pose_root / triplet_name / sequence_name / "right_hand.pkl"
            left_hand_shape_path = hand_pose_root / triplet_name / sequence_name / "left_hand_shape.pkl"
            right_hand_shape_path = hand_pose_root / triplet_name / sequence_name / "right_hand_shape.pkl"
            intrinsic_path = camera_param_root / triplet_name / sequence_name / "egocentric_intrinsic.txt"
            extrinsic_path = camera_param_root / triplet_name / sequence_name / "egocentric_frame_extrinsic.npy"
            
            left_hand = safe_load_pickle(left_hand_path)
            right_hand = safe_load_pickle(right_hand_path)
            left_hand_shape = safe_load_pickle(left_hand_shape_path)
            right_hand_shape = safe_load_pickle(right_hand_shape_path)
            intrinsic = np.loadtxt(intrinsic_path)
            extrinsic = np.load(extrinsic_path)

            # Process single dataset
            dataset = self.process_single_episode(
                dataset,
                left_hand, 
                right_hand, 
                left_hand_shape,
                right_hand_shape,
                intrinsic,
                extrinsic,
                video_path,
                triplet_name
            )

        print(f"Saving converted dataset to {output_root}")

    def process_single_episode(
        self,
        dataset: LeRobotDataset,
        left_hand: Dict[str, Any],
        right_hand: Dict[str, Any],
        left_hand_shape: Dict[str, Any],
        right_hand_shape: Dict[str, Any],
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        video_path: Path,
        triplet_name: str,
    ) -> LeRobotDataset:
        
        w1, w2, w3 = str(triplet_name).strip("()").split(", ")
        
        num_frames = len(left_hand)
        # 一个小patch：torchcodec识别出来的帧数和left_hand里面不太一样
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Failed to open {video_path}")
            return
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        num_frames = min(num_frames, frame_count)

        print(f"Processing episode {triplet_name} with {num_frames} frames.")
        extract = 30 // self.fps
        if 30 % self.fps != 0:
            print(f"Warning: fps {self.fps} does not divide 30 evenly, some frames may be dropped.")

        def get_wrist_pose(pose_tensor, trans_tensor): 
            pose = pose_tensor.numpy()  # (48,)
            wrist_axisangle = pose[:3]
            R_mat = R.from_rotvec(wrist_axisangle)
            rpy_euler = R_mat.as_euler('ZYX', degrees=False)
            yaw, pitch, roll = rpy_euler
            # 提取手腕的绝对位置
            wrist_pos = trans_tensor.numpy()  # (3,)
            x, y, z = wrist_pos
            return np.array([x, y, z, roll, pitch, yaw]).astype(np.float32)

        def get_frames(video_file: Path) -> torch.Tensor:
            timestamps = [t/30 for t in range(0, num_frames, extract)]
            # 用 lerobot 提供的工具，按照时间戳提取帧。返回形状 (T, C, H, W) ，值被归一化到 float32 [0, 1]
            frames = decode_video_frames(video_file, timestamps, tolerance_s=1.0/self.fps)
            # 转回 uint8 [0, 255]
            frames = (frames*255).to(torch.uint8)
            return frames

        frames = get_frames(video_path)

        for i in range(0, num_frames, extract):
            left_pose = left_hand[f"{i+1:05d}"]["hand_pose"]
            left_trans = left_hand[f"{i+1:05d}"]["hand_trans"]
            right_pose = right_hand[f"{i+1:05d}"]["hand_pose"]
            right_trans = right_hand[f"{i+1:05d}"]["hand_trans"]
            left_wrist = get_wrist_pose(left_pose, left_trans)
            right_wrist = get_wrist_pose(right_pose, right_trans)
            frame = {
                "eef.hand_raw": np.concatenate([
                    left_hand[f"{i+1:05d}"]["hand_pose"].numpy(),
                    left_hand[f"{i+1:05d}"]["hand_trans"].numpy(),
                    left_hand_shape["hand_shape"].numpy(),
                    right_hand[f"{i+1:05d}"]["hand_pose"].numpy(),
                    right_hand[f"{i+1:05d}"]["hand_trans"].numpy(),
                    right_hand_shape["hand_shape"].numpy(),
                ], axis=0).astype(np.float32), # 直接暴力合并
                "eef.left.wrist": left_wrist,
                "eef.right.wrist": right_wrist,
                "camera.intrinsic": intrinsic.astype(np.float32).reshape(3, 3),
                "camera.extrinsic": extrinsic[i].astype(np.float32).reshape(4, 4),
                "annotation.language.task_text": f"use the {w2} to {w1} the {w3}",
                "annotation.language.action_text": np.array([0], dtype=np.int64),
                "observation.images.top_head": frames[i // extract].cpu().numpy(),  # (C, H, W) numpy array
            }
            dataset.add_frame(frame, task=f"use the {w2} to {w1} the {w3}")

        dataset.save_episode()

        return dataset

def parse_args():  # input_root, output_root, fps, repo_id
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to the TACO dataset root directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to save the converted LeRobot dataset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate for video sampling.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="TACO_LeRobot_v2.1",
        help="HuggingFace repo ID to save the dataset.",
    )
    args = parser.parse_args()
    return args.input_root, args.output_root, args.fps, args.repo_id

def main():
    global input_root, output_root, fps, repo_id 
    input_root, output_root, fps, repo_id = parse_args()
    # if the output_root already exists, we will overwrite it
    shutil.rmtree(output_root, ignore_errors=True)
    root_path = Path(output_root)
    print(f"Converting TACO dataset from {input_root} to LeRobot dataset at {output_root} with fps={fps}")
    converter = Taco2LeRobot(
        input_root=Path(input_root),
        output_root=root_path,
        fps=fps,
        repo_id=repo_id,
    )
    converter.convert()

if __name__ == "__main__":
    main()