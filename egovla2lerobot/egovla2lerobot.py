#!/usr/bin/env python3
"""Convert EgoVLA finetune data to LeRobot format."""

import argparse
import io
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision
from datasets import Dataset, load_from_disk
from lerobot.common.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

torchvision.set_video_backend("pyav")

DEFAULT_SOURCE_FPS = 30
DEFAULT_TARGET_FPS = 10

VIDEO_KEY = "observation.images.top_head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
EEF_LEFT_WRIST_KEY = "eef.left.wrist"
EEF_RIGHT_WRIST_KEY = "eef.right.wrist"
EEF_STATE_KEY = "eef.state"
EEF_LEFT_HAND_KEY = "eef.left.hand"
EEF_RIGHT_HAND_KEY = "eef.right.hand"
CAMERA_INTRINSIC_KEY = "camera.intrinsic"
CAMERA_EXTRINSIC_KEY = "camera.extrinsic"

ORIGINAL_HAND_KEYS = [
    "current_left_mano_trans",
    "current_left_mano_rot",
    "current_left_mano_kps3d",
    "current_left_mano_parameters",
    "current_left_mano_ee_2d",
    "current_left_finger_tip_cam_pos",
    "current_right_mano_trans",
    "current_right_mano_rot",
    "current_right_mano_kps3d",
    "current_right_mano_parameters",
    "current_right_mano_ee_2d",
    "current_right_finger_tip_cam_pos",
    "curent_qpos",
]

OTV_MAIN_INTRINSICS = np.array(
    [[146.6, 0.0000, 192.0000], [0.0000, 260.6, 192.0000], [0.0000, 0.0000, 1.0000]],
    dtype=np.float32,
)

OTV_MAIN_CAM_QUAT_XYZW = np.array([0.24184, -0.24184, -0.664464, 0.66446], dtype=np.float32)
OTV_MAIN_CAM_TRANS = np.array([0.09, 0.0, 1.7], dtype=np.float32)
_rot = R.from_quat(OTV_MAIN_CAM_QUAT_XYZW)
_rot_matrix = _rot.as_matrix()
_gt_cam_quat_xyzw = np.array([0.0, 0.42261826174069944, 0.0, 0.9063077870366499], dtype=np.float32)
_gt_rotmat = R.from_quat(_gt_cam_quat_xyzw).as_matrix()
ISAAC_LAB_CAMERA_FRAME_CHANGE = _gt_rotmat @ np.linalg.inv(_rot_matrix)
OTV_MAIN_CAM_TRANSFORMATION = np.eye(4, dtype=np.float32)
OTV_MAIN_CAM_TRANSFORMATION[:3, :3] = ISAAC_LAB_CAMERA_FRAME_CHANGE @ _rot_matrix
OTV_MAIN_CAM_TRANSFORMATION[:3, 3] = OTV_MAIN_CAM_TRANS


def sample_images(input):
    if type(input) is str:
        reader = torchvision.io.VideoReader(input, stream="video")
        frames = [frame["data"] for frame in reader]
        frames_array = torch.stack(frames).numpy()
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = auto_downsample_height_width(frames_array[idx])
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    elif type(input) is np.ndarray:
        frames_array = input[:, None, :, :]
        sampled_indices = sample_indices(len(frames_array))
        images = None
        for i, idx in enumerate(sampled_indices):
            img = auto_downsample_height_width(frames_array[idx])
            if images is None:
                images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)
            images[i] = img
    return images


def compute_episode_stats(episode_data: dict, features: dict) -> dict:
    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1
        ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
        if features[key]["dtype"] in ["image", "video"]:
            value_norm = 1.0 if "depth" in key else 255.0
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / value_norm, axis=0)
                for k, v in ep_stats[key].items()
            }
    return ep_stats


class EgoVLADatasetMetadata(LeRobotDatasetMetadata):
    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict,
        action_config: list,
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()
        write_info(self.info, self.root)
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
            "action_config": action_config,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)
        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)


class EgoVLADataset(LeRobotDataset):
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "EgoVLADataset":
        obj = cls.__new__(cls)
        obj.meta = EgoVLADatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)
        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()
        features = {key: value for key, value in self.features.items() if key in self.hf_features}
        validate_frame(frame, features)
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
        frame_index = self.episode_buffer["size"]
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)
        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'.")
            self.episode_buffer[key].append(value)
        self.episode_buffer["size"] += 1

    def save_episode(self, videos: dict, action_config: list, episode_data: dict | None = None) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]
        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])
        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
            episode_buffer[key] = str(video_path)
        ep_stats = compute_episode_stats(episode_buffer, self.features)
        for key in self.meta.video_keys:
            video_path = episode_buffer[key]
            episode_buffer[key] = [video_path] * episode_length
        self._save_episode_table(episode_buffer, episode_index)
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, action_config)
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )
        self.episode_buffer = None


def images_to_video(images: List[np.ndarray], output_path: Path, fps: float = 10.0) -> Path:
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)
    out.release()
    return output_path


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    return np.array(img)


def axis_angle_to_rpy(axis_angle: np.ndarray) -> np.ndarray:
    if np.linalg.norm(axis_angle) < 1e-6:
        return np.zeros(3, dtype=np.float32)
    rot = R.from_rotvec(axis_angle)
    rpy = rot.as_euler('xyz', degrees=False)
    return rpy.astype(np.float32)


def extract_state_from_sample(sample: dict) -> np.ndarray:
    state_parts = []
    if "current_left_mano_ee_2d" in sample and "current_right_mano_ee_2d" in sample:
        left_2d = np.array(sample["current_left_mano_ee_2d"], dtype=np.float32).reshape(-1)
        right_2d = np.array(sample["current_right_mano_ee_2d"], dtype=np.float32).reshape(-1)
        state_parts.append(np.concatenate([left_2d[:2], right_2d[:2]]))
    else:
        # state_parts.append(np.zeros(4, dtype=np.float32))
        raise ValueError
    if "current_left_mano_trans" in sample and "current_right_mano_trans" in sample:
        left_3d = np.array(sample["current_left_mano_trans"], dtype=np.float32).reshape(-1)
        right_3d = np.array(sample["current_right_mano_trans"], dtype=np.float32).reshape(-1)
        state_parts.append(np.concatenate([left_3d[:3], right_3d[:3]]))
    else:
        # state_parts.append(np.zeros(6, dtype=np.float32))
        raise ValueError
    if "current_left_mano_rot" in sample and "current_right_mano_rot" in sample:
        left_rot = np.array(sample["current_left_mano_rot"], dtype=np.float32).reshape(-1)
        right_rot = np.array(sample["current_right_mano_rot"], dtype=np.float32).reshape(-1)
        state_parts.append(np.concatenate([left_rot[:3], right_rot[:3]]))
    else:
        # state_parts.append(np.zeros(6, dtype=np.float32))
        raise ValueError
    if "current_left_mano_parameters" in sample and "current_right_mano_parameters" in sample:
        left_dof = np.array(sample["current_left_mano_parameters"], dtype=np.float32).reshape(-1)
        right_dof = np.array(sample["current_right_mano_parameters"], dtype=np.float32).reshape(-1)
        state_parts.append(np.concatenate([left_dof[:15], right_dof[:15]]))
    else:
        # state_parts.append(np.zeros(30, dtype=np.float32))
        raise ValueError
    return np.concatenate(state_parts)


def extract_action_from_sample(sample: dict, next_sample: Optional[dict] = None) -> np.ndarray:
    if next_sample is None:
        return np.zeros(46, dtype=np.float32)
    action_parts = []
    if "current_left_mano_ee_2d" in sample and "current_left_mano_ee_2d" in next_sample:
        curr_left_2d = np.array(sample["current_left_mano_ee_2d"], dtype=np.float32).reshape(-1)
        next_left_2d = np.array(next_sample["current_left_mano_ee_2d"], dtype=np.float32).reshape(-1)
        curr_right_2d = np.array(sample["current_right_mano_ee_2d"], dtype=np.float32).reshape(-1)
        next_right_2d = np.array(next_sample["current_right_mano_ee_2d"], dtype=np.float32).reshape(-1)
        action_parts.append(np.concatenate([next_left_2d[:2] - curr_left_2d[:2], next_right_2d[:2] - curr_right_2d[:2]]))
    else:
        action_parts.append(np.zeros(4, dtype=np.float32))
    if "current_left_mano_trans" in sample and "current_left_mano_trans" in next_sample:
        curr_left_3d = np.array(sample["current_left_mano_trans"], dtype=np.float32).reshape(-1)
        next_left_3d = np.array(next_sample["current_left_mano_trans"], dtype=np.float32).reshape(-1)
        curr_right_3d = np.array(sample["current_right_mano_trans"], dtype=np.float32).reshape(-1)
        next_right_3d = np.array(next_sample["current_right_mano_trans"], dtype=np.float32).reshape(-1)
        action_parts.append(np.concatenate([next_left_3d[:3] - curr_left_3d[:3], next_right_3d[:3] - curr_right_3d[:3]]))
    else:
        action_parts.append(np.zeros(6, dtype=np.float32))
    if "current_left_mano_rot" in sample and "current_left_mano_rot" in next_sample:
        curr_left_rot = np.array(sample["current_left_mano_rot"], dtype=np.float32).reshape(-1)
        next_left_rot = np.array(next_sample["current_left_mano_rot"], dtype=np.float32).reshape(-1)
        curr_right_rot = np.array(sample["current_right_mano_rot"], dtype=np.float32).reshape(-1)
        next_right_rot = np.array(next_sample["current_right_mano_rot"], dtype=np.float32).reshape(-1)
        action_parts.append(np.concatenate([next_left_rot[:3] - curr_left_rot[:3], next_right_rot[:3] - curr_right_rot[:3]]))
    else:
        action_parts.append(np.zeros(6, dtype=np.float32))
    if "current_left_mano_parameters" in sample and "current_left_mano_parameters" in next_sample:
        curr_left_dof = np.array(sample["current_left_mano_parameters"], dtype=np.float32).reshape(-1)
        next_left_dof = np.array(next_sample["current_left_mano_parameters"], dtype=np.float32).reshape(-1)
        curr_right_dof = np.array(sample["current_right_mano_parameters"], dtype=np.float32).reshape(-1)
        next_right_dof = np.array(next_sample["current_right_mano_parameters"], dtype=np.float32).reshape(-1)
        action_parts.append(np.concatenate([next_left_dof[:15] - curr_left_dof[:15], next_right_dof[:15] - curr_right_dof[:15]]))
    else:
        action_parts.append(np.zeros(30, dtype=np.float32))
    return np.concatenate(action_parts)


def extract_eef_wrist_hand(sample: dict) -> Dict[str, np.ndarray]:
    result = {}
    if "current_left_ee_cam_pose" in sample and "current_right_ee_cam_pose" in sample:
        left_cam_pose = np.array(sample["current_left_ee_cam_pose"], dtype=np.float32).reshape(4, 4)
        right_cam_pose = np.array(sample["current_right_ee_cam_pose"], dtype=np.float32).reshape(4, 4)
        left_trans, left_rot_matrix = left_cam_pose[:3, 3], left_cam_pose[:3, :3]
        right_trans, right_rot_matrix = right_cam_pose[:3, 3], right_cam_pose[:3, :3]
        left_rpy = R.from_matrix(left_rot_matrix).as_euler('xyz', degrees=False).astype(np.float32)
        right_rpy = R.from_matrix(right_rot_matrix).as_euler('xyz', degrees=False).astype(np.float32)
        left_wrist = np.concatenate([left_trans, left_rpy])
        right_wrist = np.concatenate([right_trans, right_rpy])
        result[EEF_STATE_KEY] = np.concatenate([left_wrist, right_wrist])
    else:
        raise ValueError
    if "current_left_mano_kps3d" in sample:
        left_kps = np.array(sample["current_left_mano_kps3d"], dtype=np.float32).reshape(-1)
        result[EEF_LEFT_HAND_KEY] = left_kps[:63].reshape(21, 3)
    else:
        result[EEF_LEFT_HAND_KEY] = np.zeros((21, 3), dtype=np.float32)
    if "current_right_mano_kps3d" in sample:
        right_kps = np.array(sample["current_right_mano_kps3d"], dtype=np.float32).reshape(-1)
        result[EEF_RIGHT_HAND_KEY] = right_kps[:63].reshape(21, 3)
    else:
        result[EEF_RIGHT_HAND_KEY] = np.zeros((21, 3), dtype=np.float32)
    return result


def extract_original_hand_data(sample: dict) -> Dict[str, np.ndarray]:
    result = {}
    for key in ORIGINAL_HAND_KEYS:
        if key in sample:
            value = sample[key]
            if key == "curent_qpos":
                result["current_qpos"] = np.array(value, dtype=np.float32)
            elif isinstance(value, (list, tuple)):
                result[key] = np.array(value, dtype=np.float32)
            elif isinstance(value, np.ndarray):
                result[key] = value.astype(np.float32)
            else:
                result[key] = np.array([value], dtype=np.float32)
        else:
            if "kps3d" in key:
                result[key] = np.zeros(63, dtype=np.float32)
            elif "parameters" in key:
                result[key] = np.zeros(15, dtype=np.float32)
            elif "finger_tip" in key:
                result[key] = np.zeros(15, dtype=np.float32)
            elif "ee_2d" in key:
                result[key] = np.zeros(2, dtype=np.float32)
            elif "trans" in key or "rot" in key:
                result[key] = np.zeros(3, dtype=np.float32)
            elif key == "curent_qpos":
                result["current_qpos"] = np.zeros(50, dtype=np.float32)
            else:
                result[key] = np.zeros(1, dtype=np.float32)
    return result


def build_features(state_dim: int, action_dim: int, image_shape: tuple = (384, 384, 3), target_fps: float = DEFAULT_TARGET_FPS) -> dict:
    video_feature = {
        "dtype": "video",
        "shape": [int(x) for x in image_shape],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(target_fps),
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }
    features = {
        VIDEO_KEY: video_feature,
        STATE_KEY: {"dtype": "float32", "shape": (state_dim,)},
        ACTION_KEY: {"dtype": "float32", "shape": (action_dim,)},
        EEF_STATE_KEY: {"dtype": "float32", "shape": (12,)},
        EEF_LEFT_WRIST_KEY: {"dtype": "float32", "shape": (6,)},
        EEF_RIGHT_WRIST_KEY: {"dtype": "float32", "shape": (6,)},
        EEF_LEFT_HAND_KEY: {"dtype": "float32", "shape": (21, 3)},
        EEF_RIGHT_HAND_KEY: {"dtype": "float32", "shape": (21, 3)},
        CAMERA_INTRINSIC_KEY: {"dtype": "float32", "shape": (9,)},
        CAMERA_EXTRINSIC_KEY: {"dtype": "float32", "shape": (16,)},
    }
    for key in ORIGINAL_HAND_KEYS:
        if "kps3d" in key:
            features[key] = {"dtype": "float32", "shape": (63,)}
        elif "parameters" in key:
            features[key] = {"dtype": "float32", "shape": (15,)}
        elif "finger_tip" in key:
            features[key] = {"dtype": "float32", "shape": (15,)}
        elif "ee_2d" in key:
            features[key] = {"dtype": "float32", "shape": (2,)}
        elif "trans" in key or "rot" in key:
            features[key] = {"dtype": "float32", "shape": (3,)}
        elif key == "curent_qpos":
            features["current_qpos"] = {"dtype": "float32", "shape": (50,)}
        else:
            features[key] = {"dtype": "float32", "shape": (1,)}
    return features


def group_samples_by_episode(dataset: Dataset) -> Dict[str, List[int]]:
    seq_names = dataset["seq_name"]
    frame_counts = dataset["frame_count"]
    episodes = defaultdict(list)
    for idx in tqdm(range(len(dataset))):
        seq_name = seq_names[idx]
        if isinstance(seq_name, (list, tuple)):
            seq_name = ''.join(seq_name)
        episodes[seq_name].append(idx)
    for seq_name in tqdm(episodes):
        indices = episodes[seq_name]
        indexed_frames = [(frame_counts[idx], idx) for idx in indices]
        indexed_frames.sort()
        episodes[seq_name] = [idx for _, idx in indexed_frames]
    return episodes


def convert_episode_to_lerobot(
    lerobot_dataset: EgoVLADataset,
    dataset: Dataset,
    img_dataset: Dataset,
    image_mapping: dict,
    episode_indices: List[int],
    output_root: Path,
    episode_index: int,
    target_fps: float = DEFAULT_TARGET_FPS,
    source_fps: float = DEFAULT_SOURCE_FPS,
) -> None:
    if len(episode_indices) == 0:
        return
    first_sample = dataset[episode_indices[0]]
    seq_name = ''.join(first_sample["seq_name"]) if isinstance(first_sample["seq_name"], (list, tuple)) else first_sample["seq_name"]
    stride = int(source_fps / target_fps)
    all_images = []
    for idx in episode_indices:
        sample = dataset[idx]
        frame_count = sample["frame_count"]
        if seq_name in image_mapping and frame_count in image_mapping[seq_name]:
            img_idx = image_mapping[seq_name][frame_count]
            img_bytes = img_dataset[img_idx]["rgb_obs"]
            all_images.append(decode_image_bytes(img_bytes))
        else:
            all_images.append(np.zeros((384, 384, 3), dtype=np.uint8))
    frames = []
    sampled_indices = episode_indices[::stride]
    for i, idx in enumerate(sampled_indices):
        sample = dataset[idx]
        next_sampled_idx = sampled_indices[min(i + 1, len(sampled_indices) - 1)]
        next_sample = dataset[next_sampled_idx] if next_sampled_idx != idx else None
        frame_data = {
            STATE_KEY: extract_state_from_sample(sample),
            ACTION_KEY: extract_action_from_sample(sample, next_sample),
            **extract_eef_wrist_hand(sample),
            **extract_original_hand_data(sample),
            CAMERA_INTRINSIC_KEY: OTV_MAIN_INTRINSICS.flatten(),
            CAMERA_EXTRINSIC_KEY: OTV_MAIN_CAM_TRANSFORMATION.flatten(),
        }
        frames.append((frame_data, sample.get("language_label", "")))
    videos_dir = output_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_path = videos_dir / f"episode_{episode_index:06d}.mp4"
    images_to_video(all_images, video_path, fps=source_fps)
    for i, (frame_data, instruction) in enumerate(frames):
        lerobot_dataset.add_frame(frame_data, task=instruction, timestamp=i / target_fps)
    lerobot_dataset.save_episode(videos={VIDEO_KEY: str(video_path)}, action_config=[])


def convert_egovla_to_lerobot(
    data_path: str,
    image_path: str,
    image_mapping_path: str,
    output_path: str,
    target_fps: float = DEFAULT_TARGET_FPS,
    source_fps: float = DEFAULT_SOURCE_FPS,
    max_episodes: Optional[int] = None,
) -> None:
    dataset = load_from_disk(data_path)
    img_dataset = load_from_disk(image_path)
    with open(image_mapping_path, 'rb') as f:
        image_mapping = pickle.load(f)
    episodes = group_samples_by_episode(dataset)
    output_root = Path(output_path)
    if len(episodes) == 0:
        return
    first_episode_indices = list(episodes.values())[0]
    first_sample = dataset[first_episode_indices[0]]
    seq_name = ''.join(first_sample["seq_name"]) if isinstance(first_sample["seq_name"], (list, tuple)) else first_sample["seq_name"]
    state_dim = len(extract_state_from_sample(first_sample))
    action_dim = len(extract_action_from_sample(first_sample, None))
    first_frame_count = first_sample["frame_count"]
    if seq_name in image_mapping and first_frame_count in image_mapping[seq_name]:
        img_idx = image_mapping[seq_name][first_frame_count]
        first_img_bytes = img_dataset[img_idx]["rgb_obs"]
        image_shape = decode_image_bytes(first_img_bytes).shape
    else:
        image_shape = (384, 384, 3)
    features = build_features(state_dim, action_dim, image_shape, target_fps=target_fps)
    lerobot_dataset = EgoVLADataset.create(
        repo_id="egovla_dataset",
        root=str(output_root),
        fps=target_fps,
        robot_type="dex",
        features=features,
    )
    for episode_idx, (seq_name, indices) in enumerate(tqdm(list(episodes.items())[:max_episodes])):
        convert_episode_to_lerobot(
            lerobot_dataset=lerobot_dataset,
            dataset=dataset,
            img_dataset=img_dataset,
            image_mapping=image_mapping,
            episode_indices=indices,
            output_root=output_root,
            episode_index=episode_idx,
            target_fps=target_fps,
            source_fps=source_fps,
        )


def main():
    parser = argparse.ArgumentParser(description="Convert EgoVLA finetune data to LeRobot format")
    parser.add_argument("--root", type=str, default=None, help="Root directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output path")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS, help="Target FPS")
    parser.add_argument("--source-fps", type=float, default=DEFAULT_SOURCE_FPS, help="Source FPS")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes")
    args = parser.parse_args()
    root_path = Path(args.root)
    if not root_path.exists():
        raise ValueError(f"Root path does not exist: {args.root}")
    convert_egovla_to_lerobot(
        data_path=str(root_path / "HF_hand_FIXED_SET_MIX_train"),
        image_path=str(root_path / "HF_images"),
        image_mapping_path=str(root_path / "hf_images_mapping.pkl"),
        output_path=args.output_path,
        target_fps=args.target_fps,
        source_fps=args.source_fps,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
