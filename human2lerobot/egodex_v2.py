from __future__ import annotations

import time
import concurrent.futures
import argparse
import json
import os
import shutil
import sys
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_process_scripts.egodex.utils.skeleton_tfs import DEFAULT_TFS  # uwm

VIDEO_TOP_KEY = "observation.images.top_head"
STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"
EEF_STATE_KEY = "eef.state"
EEF_ACTION_KEY = "eef.action"
ANNOTATION_TEXT_KEY = "annotation.language.action_text"
TIMESTAMP_KEY = "timestamp"
CAMERA_INTRINSIC_KEY = "camera.intrinsic"
CAMERA_EXTRINSIC_KEY = "camera.extrinsic"

DEFAULT_VIDEO_SHAPE = (1080, 1920, 3)
DEFAULT_FPS = 30
DEFAULT_TARGET_FPS = 10
DEFAULT_ROBOT_TYPE = "dex"

TRANSFORM_KEYS = sorted(DEFAULT_TFS)
EEF_KEYS = ("leftHand", "rightHand")
GRIPPER_CLOSED_VALUE = 0.0

CAMERA_ALIGN_FIX = np.array(
    [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float32,
)
PIXEL_EPS = 1e-9
AXIS_COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
TEXT_COLOR = (255, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
TEXT_LINE_STEP = 18
POINT_COLORS = {"leftHand": (0, 165, 255), "rightHand": (255, 0, 255)}
AXIS_LENGTH = 0.1
AXIS_THICKNESS = 2
POINT_RADIUS = 5

DATASET_PATH_PREFIX = "data"
VIEW_NAME = "front_view"
DEFAULT_DATASET_NAME = "EgoDex_Preprocessed"


@dataclass
class EpisodeData:
    source_path: Path
    data_id: str
    task_name: str
    instruction: str
    state_vectors: np.ndarray
    eef_states: np.ndarray
    eef_actions: np.ndarray
    video_path: Path | None
    camera_tf: np.ndarray
    camera_intrinsics: np.ndarray
    transforms_camera_frame: Dict[str, np.ndarray]
    confidences: Dict[str, np.ndarray]
    transforms_world: Dict[str, np.ndarray]

    @property
    def num_frames(self) -> int:
        return self.state_vectors.shape[0]


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = float(np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        pitch = float(np.arctan2(-rotation[2, 0], sy))
        yaw = 0.0
    return roll, pitch, yaw


def transform_to_pose7(mat: np.ndarray) -> np.ndarray:
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = mat[:3, 3]
    roll, pitch, yaw = rotation_matrix_to_rpy(mat[:3, :3])
    pose[3] = roll
    pose[4] = pitch
    pose[5] = yaw
    pose[6] = GRIPPER_CLOSED_VALUE
    return pose


def compute_eef_states_actions(transforms_camera_frame: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    missing = [key for key in EEF_KEYS if key not in transforms_camera_frame]
    if missing:
        raise KeyError(f"缺少末端执行器变换：{', '.join(missing)}")

    num_frames = transforms_camera_frame[EEF_KEYS[0]].shape[0]
    eef_dim = len(EEF_KEYS) * 7
    eef_states = np.empty((num_frames, eef_dim), dtype=np.float32)
    eef_actions = np.zeros((num_frames, eef_dim), dtype=np.float32)

    for frame_idx in range(num_frames):
        offset = 0
        for key in EEF_KEYS:
            tf = transforms_camera_frame[key][frame_idx]
            eef_states[frame_idx, offset : offset + 7] = transform_to_pose7(tf)
            offset += 7

    if num_frames > 1:
        for frame_idx in range(num_frames - 1):
            offset = 0
            for key in EEF_KEYS:
                curr_tf = transforms_camera_frame[key][frame_idx]
                fut_tf = transforms_camera_frame[key][frame_idx + 1]
                rel_tf = np.linalg.inv(curr_tf) @ fut_tf
                delta_trans = rel_tf[:3, 3].astype(np.float32)
                delta_rpy = rotation_matrix_to_rpy(rel_tf[:3, :3])
                eef_actions[frame_idx, offset : offset + 7] = np.array(
                    [
                        delta_trans[0],
                        delta_trans[1],
                        delta_trans[2],
                        float(delta_rpy[0]),
                        float(delta_rpy[1]),
                        float(delta_rpy[2]),
                        0.0,
                    ],
                    dtype=np.float32,
                )
                offset += 7

    return eef_states, eef_actions


def align_eef_transforms(
    transforms_camera_frame: Dict[str, np.ndarray], align_matrix: np.ndarray | None = None
) -> Dict[str, np.ndarray]:
    if align_matrix is None:
        return {key: value.copy() for key, value in transforms_camera_frame.items()}

    matrix = np.asarray(align_matrix, dtype=np.float32).reshape(4, 4)
    aligned: Dict[str, np.ndarray] = {}
    for key, value in transforms_camera_frame.items():
        aligned[key] = np.matmul(matrix[np.newaxis, ...], value).astype(np.float32)
    return aligned

def point_3d_to_2d(intrinsics: np.ndarray, point: np.ndarray) -> np.ndarray:
    orig_shape = point.shape
    coords = point.reshape(-1, 3).astype(np.float32)
    depth = coords[:, [2]]
    depth_sign = np.where(depth >= 0.0, 1.0, -1.0).astype(np.float32)
    depth = np.where(np.abs(depth) < PIXEL_EPS, depth_sign * PIXEL_EPS, depth)
    norm = np.concatenate([coords[:, :2] / -depth, np.ones_like(depth)], axis=1)
    pixels = np.einsum("ab,nb->na", intrinsics, norm, dtype=np.float32)
    return pixels.reshape(*orig_shape[:-1], 3)[..., :2]


def _pixel_to_cv(pixel: np.ndarray) -> tuple[int, int]:
    return int(round(float(pixel[1]))), int(round(float(pixel[0])))


def add_pose_axes(image: np.ndarray, pose: np.ndarray, intrinsics: np.ndarray,
                  axis_length: float, thickness: int) -> np.ndarray:
    origin = pose[:3, 3]
    origin_px = point_3d_to_2d(intrinsics, origin)

    for axis_idx in range(3):
        axis_tip = origin + axis_length * pose[:3, axis_idx]
        tip_px = point_3d_to_2d(intrinsics, axis_tip)

        cv2.line(
            image,
            _pixel_to_cv(origin_px),
            _pixel_to_cv(tip_px),
            AXIS_COLORS[axis_idx],
            thickness
        )

        z = pose[2, axis_idx]

        tip_px_cv = _pixel_to_cv(tip_px)

        if z > 0:
            cv2.circle(image, tip_px_cv, radius=5, color=AXIS_COLORS[axis_idx], thickness=-1)
        else:
            cv2.circle(image, tip_px_cv, radius=6, color=AXIS_COLORS[axis_idx], thickness=2)

    return image

def add_point(image: np.ndarray, pixel: np.ndarray, color: tuple[int, int, int], radius: int) -> np.ndarray:
    cv2.circle(image, _pixel_to_cv(pixel), radius, color, -1)
    return image


def add_pose_label(image: np.ndarray, pixel: np.ndarray, label: str, pose_vec: np.ndarray) -> np.ndarray:
    x, y = _pixel_to_cv(pixel)
    height, width = image.shape[:2]
    lines = (
        label,
        f"xyz: {pose_vec[0]:.3f} {pose_vec[1]:.3f} {pose_vec[2]:.3f}",
        f"rpy: {pose_vec[3]:.3f} {pose_vec[4]:.3f} {pose_vec[5]:.3f}",
    )
    base_x = min(max(0, x + 6), width - 1)
    base_y = min(max(TEXT_LINE_STEP, y + TEXT_LINE_STEP), height - 1)
    for idx, text in enumerate(lines):
        anchor_y = base_y + idx * TEXT_LINE_STEP
        if anchor_y >= height:
            anchor_y = height - 1
        cv2.putText(image, text, (base_x, anchor_y), TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
    return image


def _load_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"无法读取视频帧 {frame_idx}，请确认索引在有效范围内。")
        return frame
    finally:
        capture.release()


def extract_instruction(attrs: h5py.AttributeManager) -> str:
    def attr_text(key: str, default: str = "") -> str:
        value = attrs.get(key, default)
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    if attr_text("llm_type") == "reversible":
        direction = attr_text("which_llm_description", "1")
        key = "llm_description" if direction == "1" else "llm_description2"
        return attr_text(key).strip()
    return attr_text("llm_description").strip()


def load_episode(hdf5_path: Path) -> EpisodeData:
    with h5py.File(hdf5_path, "r") as root:
        camera_tf = np.asarray(root["/transforms/camera"], dtype=np.float32)
        instruction = extract_instruction(root.attrs)
        transforms_world = {
            key: np.asarray(root[f"/transforms/{key}"], dtype=np.float32)
            for key in TRANSFORM_KEYS
        }
        confidences = {
            key: np.asarray(root[f"/confidences/{key}"], dtype=np.float32)
            for key in TRANSFORM_KEYS if f"/confidences/{key}" in root
        }
        intrinsics = np.asarray(root["/camera/intrinsic"], dtype=np.float32)

    if intrinsics is None:
        raise ValueError(f"未能在 {hdf5_path} 找到相机内参数据。")

    num_frames = camera_tf.shape[0]
    state_vectors = np.concatenate(
        [transforms_world[key].reshape(num_frames, 16) for key in TRANSFORM_KEYS],
        axis=1,
    )

    cam_inv = np.linalg.inv(camera_tf).astype(np.float32)
    transforms_camera_frame: Dict[str, np.ndarray] = {
        key: np.einsum("tij,tjk->tik", cam_inv, transforms_world[key], dtype=np.float32)
        for key in TRANSFORM_KEYS
    }
    

    eef_states, eef_actions = compute_eef_states_actions(transforms_camera_frame)

    relative_parts = hdf5_path.parts

    task_folder = relative_parts[-2] if len(relative_parts) >= 2 else hdf5_path.parent.name
    task_name = task_folder or "unknown_task"
    data_id = f"{task_name}/{hdf5_path.stem}" if task_folder else hdf5_path.stem

    video_path = hdf5_path.with_suffix(".mp4")

    return EpisodeData(
        source_path=hdf5_path,
        data_id=data_id,
        task_name=task_name,
        instruction=instruction or "unknown EgoDex instruction",
        state_vectors=state_vectors,
        eef_states=eef_states,
        eef_actions=eef_actions,
        video_path=video_path if video_path.exists() else None,
        camera_tf=camera_tf,
        camera_intrinsics=intrinsics,
        transforms_camera_frame=transforms_camera_frame,
        confidences=confidences,
        transforms_world=transforms_world,
    )

def vis(transforms, intrinsics, frame, output_path, gripper_keys=['left_gripper', 'right_gripper'], delta_pose=dict()):
    frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for key in gripper_keys:
        eef_in_cam = transforms[key] @ delta_pose.get(key, np.eye(4))
        print(key, np.linalg.det(eef_in_cam))
        print(eef_in_cam)
        print()
        frame = add_pose_axes(frame, eef_in_cam, intrinsics, AXIS_LENGTH, AXIS_THICKNESS)
        pixel = point_3d_to_2d(intrinsics, eef_in_cam[:3, 3])
        frame = add_point(frame, pixel, POINT_COLORS.get(key, (255, 255, 255)), POINT_RADIUS)
        pose_vec = transform_to_pose7(eef_in_cam)
        frame = add_pose_label(frame, pixel, key, pose_vec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame).save(output_path)
    return output_path


def visualize_eef_pose(
    hdf5_path: Path,
    frame_idx: int,
    output_path: Path,
    video_path: Path | None = None,
) -> Path:
    episode = load_episode(hdf5_path)
    num_frames = episode.num_frames
    index = frame_idx if frame_idx >= 0 else num_frames + frame_idx
    if index < 0 or index >= num_frames:
        raise IndexError(f"帧索引 {frame_idx} 超出范围 [0, {num_frames})。")

    intrinsics = episode.camera_intrinsics.copy()
    intrinsics[0, 2], intrinsics[1, 2] = intrinsics[1, 2], intrinsics[0, 2]

    if video_path is None:
        video_path = episode.video_path or hdf5_path.with_suffix(".mp4")
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("未找到匹配的视频文件，请使用 --video_path 指定。")

    frame = _load_video_frame(video_path, index)

    aligned_transforms = align_eef_transforms(episode.transforms_camera_frame, CAMERA_ALIGN_FIX)
    
    aligned_transforms['left_gripper'] = aligned_transforms['leftHand']
    aligned_transforms['right_gripper'] = aligned_transforms['rightHand']
    aligned_transforms['right_gripper'] = np.einsum('kab,bc->kac', aligned_transforms['rightHand'], np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    return vis({k: v[index] for k, v in aligned_transforms.items()}, intrinsics, frame, output_path, delta_pose=dict(left_gripper=np.eye(4), right_gripper=np.eye(4)))


def build_features(eef_state_dim: int, eef_action_dim: int, target_fps: float, transform_names: List[str]) -> Dict[str, Dict[str, object]]:
    video_feature = {
        "dtype": "video",
        "shape": [int(x) for x in DEFAULT_VIDEO_SHAPE],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(target_fps),
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    def vector_feature(size: int, dtype: str = "float32") -> Dict[str, object]:
        return {"dtype": dtype, "shape": [int(size)]}

    features: Dict[str, Dict[str, object]] = {
        VIDEO_TOP_KEY: video_feature,
        STATE_FEATURE_KEY: vector_feature(1),
        ACTION_FEATURE_KEY: vector_feature(1),
        EEF_STATE_KEY: vector_feature(eef_state_dim),
        EEF_ACTION_KEY: vector_feature(eef_action_dim),
        ANNOTATION_TEXT_KEY: {"dtype": "int64", "shape": [1]},
        TIMESTAMP_KEY: vector_feature(1),
        CAMERA_INTRINSIC_KEY: vector_feature(9),
        CAMERA_EXTRINSIC_KEY: vector_feature(16),
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for name in transform_names:
        if name == "camera":
            continue
        features[f"tf.{name}"] = vector_feature(16)
        features[f"confidences.{name}"] = vector_feature(1)
    return features

def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")


def write_modality_file(dataset_root: Path, eef_state_dim: int, eef_action_dim: int) -> None:
    segments = len(EEF_KEYS)
    if eef_state_dim % segments or eef_action_dim % segments:
        raise ValueError("EEF 维度无法平均分配给左右末端执行器，请检查输入维度。")

    eef_state_stride = int(eef_state_dim / segments)
    eef_action_stride = int(eef_action_dim / segments)

    state_slices: Dict[str, Dict[str, object]] = {}
    action_slices: Dict[str, Dict[str, object]] = {}
    for idx, key in enumerate(EEF_KEYS):
        name = "left_end_effector" if "left" in key.lower() else "right_end_effector"
        state_slices[name] = {
            "start": int(idx * eef_state_stride),
            "end": int((idx + 1) * eef_state_stride),
            "original_key": EEF_STATE_KEY,
        }
        action_slices[name] = {
            "start": int(idx * eef_action_stride),
            "end": int((idx + 1) * eef_action_stride),
            "original_key": EEF_ACTION_KEY,
        }

    modality = {
        "state": state_slices,
        "action": action_slices,
        "video": {"top_head": {"original_key": VIDEO_TOP_KEY}},
        "annotation": {"language.action_text": {"original_key": ANNOTATION_TEXT_KEY}},
    }
    write_json(dataset_root / "meta" / "modality.json", modality)


def update_episode_annotation_files(dataset_root: Path, episode_rows: List[Dict[str, object]]) -> None:
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        return

    for row in episode_rows:
        episode_file = episodes_dir / f"episode_{row['episode_index']:06d}.json"
        if not episode_file.exists():
            continue
        with episode_file.open("r", encoding="utf-8") as fp:
            episode_info = json.load(fp)
        annotation = episode_info.setdefault("annotation", {})
        language_entry = annotation.setdefault("language.action_text", {})
        language_entry["task_index"] = int(row["task_index"])
        language_entry["text"] = row["instruction"]
        language_entry["source"] = "EgoDex attribute"

        episode_info["instruction"] = row["instruction"]
        episode_info.setdefault("source", {})["egodex_data_id"] = row["data_id"]
        if row.get("video_path"):
            episode_info.setdefault("video", {})["top_head"] = row["video_path"]

        with episode_file.open("w", encoding="utf-8") as fp:
            json.dump(episode_info, fp, indent=2, ensure_ascii=False)


def update_info_file(dataset_root: Path, total_episodes: int, unique_tasks: int) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return
    with info_path.open("r", encoding="utf-8") as fp:
        info = json.load(fp)
    info["description"] = "EgoDex → LeRobot 转换数据集，包含真实的手部 SE(3) 与摄像机参数，图像与动作仍为占位符。"
    info["source_dataset"] = "EgoDex"
    info["num_episodes_converted"] = int(total_episodes)
    info["num_unique_tasks"] = int(unique_tasks)
    write_json(info_path, info)


def write_tasks_file(dataset_root: Path, task_rows: List[Dict[str, object]]) -> None:
    write_jsonl(dataset_root / "meta" / "tasks.jsonl", task_rows)
    
def egodex_worker(args):
    (
        hdf5_path,
        stride,
        TRANSFORM_KEYS,
    ) = args

    ep = load_episode(hdf5_path)

    num_frames = ep.num_frames
    indices = np.arange(0, num_frames, stride, dtype=int)
    if len(indices) == 0:
        return None

    # aligned_tfs = align_eef_transforms(ep.transforms_camera_frame, CAMERA_ALIGN_FIX)
    confidences = ep.confidences
    subsampled_tfs = {k: v[indices] for k, v in ep.transforms_world.items()}
    
    subsampled_confidences = {}
    for k, v in confidences.items():
        try:
            subsampled_confidences[k] = v[indices]
        except Exception:
            pass
    subsampled_cam = ep.camera_tf[indices]

    aligned_eef_states, aligned_eef_actions = compute_eef_states_actions(subsampled_tfs)

    frames = []
    for i in range(len(indices)):
        f = {
            "eef_state": aligned_eef_states[i],
            "eef_action": aligned_eef_actions[i],
            "extrinsic": subsampled_cam[i].astype(np.float32).reshape(16),
        }
        for name in TRANSFORM_KEYS:
            if name == "camera" or name not in subsampled_tfs:
                continue
            f[f"tf.{name}"] = subsampled_tfs[name][i].astype(np.float32).reshape(16)
            conf_array = subsampled_confidences.get(name, None)
            if conf_array is None or len(conf_array) <= i:
                conf_val = -1.0
            else:
                conf_val = float(conf_array[i])
            f[f"confidences.{name}"] = [conf_val]
        frames.append(f)

    return dict(
        task_name=ep.task_name,
        instruction=ep.instruction,
        data_id=ep.data_id,
        intrinsic=ep.camera_intrinsics.astype(np.float32).reshape(9),
        frames=frames,
        video_src=str(ep.video_path) if ep.video_path and ep.video_path.exists() else None,
    )


def convert_single_task(task_dir: Path, output_root: Path, target_fps: float = DEFAULT_TARGET_FPS) -> None:
    """将单个 task 目录下的所有 *.hdf5 转成一个 LeRobot 数据集，输出到 output_root/task_name。

    假设结构：
        task_dir/
            000001.hdf5
            000001.mp4
            000002.hdf5
            ...
    """

    hdf5_files = sorted(path for path in task_dir.glob("*.hdf5") if path.is_file())
    if not hdf5_files:
        print(f"[convert_single_task] 在 {task_dir} 未找到任何 .hdf5 文件，跳过。")
        return

    task_name = task_dir.name
    dataset_root = (output_root / task_name).resolve()
    if dataset_root.exists():
        print(f"[convert_single_task] 目标目录已存在，跳过: {dataset_root}")
        return

    first_episode = load_episode(hdf5_files[0])
    eef_state_dim = first_episode.eef_states.shape[1]
    eef_action_dim = first_episode.eef_actions.shape[1]

    if DEFAULT_FPS % target_fps != 0:
        raise ValueError(f"源FPS {DEFAULT_FPS} 不能被目标FPS {target_fps} 整除。")
    stride = int(DEFAULT_FPS / target_fps)

    dataset = LeRobotDataset.create(
        repo_id=task_name,
        root=str(dataset_root),
        fps=float(target_fps),
        robot_type=DEFAULT_ROBOT_TYPE,
        features=build_features(eef_state_dim, eef_action_dim, target_fps, TRANSFORM_KEYS),
    )

    args_list = [
        (
            hdf5_path,
            stride,
            TRANSFORM_KEYS,
        )
        for i, hdf5_path in enumerate(hdf5_files)
    ]

    results = []
    for args in tqdm(args_list, desc=f"Converting {task_name}"):
        results.append(egodex_worker(args))

    zero_state = np.zeros(1, dtype=np.float32)
    zero_action = np.zeros(1, dtype=np.float32)

    for result in results:
        if result is None:
            continue

        for frame_idx, fr in enumerate(result["frames"]):
            tf_vectors = {name: v for name, v in fr.items() if name.startswith("tf.")}
            cf_vectors = {name: v for name, v in fr.items() if name.startswith("confidences.")}
            dataset.add_frame(
                {
                    STATE_FEATURE_KEY: zero_state.copy(),
                    ACTION_FEATURE_KEY: zero_action.copy(),
                    EEF_STATE_KEY: fr["eef_state"],
                    EEF_ACTION_KEY: fr["eef_action"],
                    ANNOTATION_TEXT_KEY: np.array([0], dtype=np.int64),
                    TIMESTAMP_KEY: np.array([frame_idx / target_fps], dtype=np.float32),
                    CAMERA_INTRINSIC_KEY: result["intrinsic"],
                    CAMERA_EXTRINSIC_KEY: fr["extrinsic"],
                    **tf_vectors,
                    **cf_vectors,
                }
            )

        dataset.save_episode(task=result["instruction"], encode_videos=False)

        if result["video_src"]:
            src = Path(result["video_src"])
            episode_index = dataset.meta.total_episodes - 1
            rel = dataset.meta.get_video_file_path(episode_index, VIDEO_TOP_KEY)
            dst = dataset.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    dataset.consolidate(run_compute_stats=False, keep_image_files=False)
    print(f"[convert_single_task] 任务 {task_name} 转换完成，输出到 {dataset_root}")

import traceback
def convert_single_task_wrapper(task_dir: Path, output_root: Path, target_fps: float):
    try:
        return convert_single_task(task_dir, output_root, target_fps)
    except Exception as e:
        print(f"\n[子进程 {os.getpid()}] 处理 task_dir={task_dir} 时出错：", flush=True)
        traceback.print_exc()  # 打印子进程里的完整 traceback
        raise  # 继续把异常抛回主进程


def convert_egodex_to_lerobot_by_task(
    input_root: Path,
    output_root: Path,
    target_fps: float = DEFAULT_TARGET_FPS,
) -> None:
    
    task_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    
    # for task_dir in tqdm(task_dirs, desc="Converting tasks (debug, no mp)"):
    #     convert_single_task(task_dir, output_root, target_fps)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    convert_single_task,
                    task_dirs,
                    itertools.repeat(output_root),
                    itertools.repeat(target_fps),
                ),
                total=len(task_dirs),
                desc="Converting tasks",
            )
        )



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cli_args = list(sys.argv[1:] if argv is None else argv)
    if not cli_args or cli_args[0] not in {"convert", "visualize"}:
        cli_args = ["convert", *cli_args]

    parser = argparse.ArgumentParser(description="EgoDex 数据处理工具：支持数据转换与末端执行器可视化。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="按 task 目录将 EgoDex 转为多个 LeRobot 数据集")
    convert_parser.add_argument("--input_path", type=Path, required=True, help="包含多个 task 子目录的根目录")
    convert_parser.add_argument("--output_path", type=Path, required=True, help="输出根目录，每个 task 一个子目录")
    convert_parser.add_argument("--target_fps", type=float, default=DEFAULT_TARGET_FPS, help="目标下采样FPS (必须整除源FPS 30)")

    visualize_parser = subparsers.add_parser("visualize", help="在指定帧上绘制 EgoDex 末端执行器姿态")
    visualize_parser.add_argument("--hdf5_path", type=Path, required=True)
    visualize_parser.add_argument("--frame_idx", type=int, required=True)
    visualize_parser.add_argument("--output_path", type=Path, required=True)
    visualize_parser.add_argument("--video_path", type=Path, default=None)

    return parser.parse_args(cli_args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "convert":
        args.output_path.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        convert_egodex_to_lerobot_by_task(
            input_root=args.input_path,
            output_root=args.output_path,
            target_fps=args.target_fps,
        )
        end_time = time.time()
        print(f"总耗时：{end_time - start_time:.2f} 秒")
        print(f"已完成按 task 转换，输出根目录：{args.output_path}")
        return

    result_path = visualize_eef_pose(
        hdf5_path=args.hdf5_path,
        frame_idx=args.frame_idx,
        output_path=args.output_path,
        video_path=args.video_path,
    )
    print(f"已保存可视化：{result_path}")


if __name__ == "__main__":
    main()