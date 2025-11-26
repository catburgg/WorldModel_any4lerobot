"""
Usage:
  - run: python final2.py \
        --taco_root /path/to/TACO/dataset \
        --output_root /path/to/save/LeRobot_v2.1_dataset \
        --repo_id your_github_repo_id \
        --fps fps
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pickle
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from manopth.manolayer import ManoLayer
import cv2
from scipy.spatial.transform import Rotation as R
import argparse
import importlib


# -----------------------------
# Config - EDIT THESE PATHS
# -----------------------------
# TACO_ROOT = Path("~/dataset_conversion/dataset").expanduser()    # TACO 根目录
# LEROBOT_SAVE_ROOT = Path("~/dataset_conversion/TACO_lerobot_v2.1_2").expanduser()  # 输出 LeRobot v2.1 根目录
# FPS = 30
CHUNKS_SIZE = 500
CODEBASE_VERSION = "v2.1" # correct
ROBOT_TYPE = "human_hand"

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
# Scan TACO sequences and group by triplet (chunk)
# -----------------------------
def scan_taco_sequences(taco_root: Path) -> List[Dict[str, Any]]:
    seqs = []
    ego_rgb_root = taco_root / "Egocentric_RGB_Videos"
    hand_pose_root = taco_root / "Hand_Poses"
    camera_parameter_root = taco_root / "Egocentric_Camera_Parameters"

    episode_index = 0
    if not ego_rgb_root.exists():
        raise RuntimeError(f"Egocentric_RGB_Videos not found at {ego_rgb_root}")

    # iterate triplet dirs
    for triplet_dir in sorted(ego_rgb_root.iterdir()):
        if not triplet_dir.is_dir():
            continue
        triplet_name = triplet_dir.name  # keep original exact name (with parentheses)
        for seq_dir in sorted(triplet_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            ego_color = seq_dir / "color.mp4"
            if not ego_color.exists():
                print(f"Warning: skipping {seq_dir} (no color.mp4)")
                continue

            relative = Path(triplet_dir.name) / Path(seq_dir.name)
            hp_dir = hand_pose_root / relative
            parameter_dir = camera_parameter_root / relative
            left_hand_p = hp_dir / "left_hand.pkl"
            right_hand_p = hp_dir / "right_hand.pkl"
            if not right_hand_p.exists() or not left_hand_p.exists():
                print(f"Warning: missing hand poses for {relative}, skipping")
                continue

            # deduce frame count from right_hand.pkl keys
            try:
                rh = safe_load_pickle(right_hand_p)
                # keys like '0001','0002' or similar
                n_frames = len(rh)
            except Exception as e:
                print(f"Warning: cannot load right hand pkl {right_hand_p}: {e}, skipping")
                continue

            seq_meta = {
                "episode_index": episode_index,
                "triplet_name": triplet_name,
                "sequence_name": seq_dir.name,
                "relative_path": str(relative),
                "length": n_frames,
                "egocentric_rgb_path": str(ego_color.resolve()),
                "hand_pose_dir": str(hp_dir),
                "camera_parameter_dir": str(parameter_dir)
            }
            seqs.append(seq_meta)
            episode_index += 1

    return seqs

# -----------------------------
# Build parquet for one episode and return episode-level metadata
# -----------------------------
def build_episode_for_chunk(episode_meta: Dict[str,Any], taco_root: Path, out_data_chunk1: Path, videos_chunk_root1: Path, models_root: Path, global_index_start: int) -> Dict[str,Any]:

    ep_idx = int(episode_meta["episode_index"])
    triplet_name = episode_meta["triplet_name"]
    sequence_name = episode_meta["sequence_name"]
    n_frames = int(episode_meta["length"])
    rel = Path(episode_meta["relative_path"])
    chunk_idx = ep_idx // CHUNKS_SIZE
    chunk_name = f"chunk-{chunk_idx:03d}"
    out_data_chunk = out_data_chunk1 / chunk_name
    videos_chunk_root = videos_chunk_root1 / chunk_name

    hand_dir = Path(episode_meta["hand_pose_dir"])
    parameter_dir = Path(episode_meta["camera_parameter_dir"])

    # load hand pose pkls
    right_hand_p = hand_dir / "right_hand.pkl"
    left_hand_p = hand_dir / "left_hand.pkl"
    right_shape_p = hand_dir / "right_hand_shape.pkl"
    left_shape_p = hand_dir / "left_hand_shape.pkl"

    # safe loads
    rh = safe_load_pickle(right_hand_p)
    lh = safe_load_pickle(left_hand_p)
    rs = safe_load_pickle(right_shape_p)
    # rs may be dict {'hand_shape': tensor(...)} or array
    right_hand_shape = rs['hand_shape'].unsqueeze(0)
    ls = safe_load_pickle(left_shape_p)
    left_hand_shape = ls['hand_shape'].unsqueeze(0)
    # camera mapping for this episode
    camera_to_srcpath = {}
    camera_to_srcpath["top_head"] = Path(episode_meta["egocentric_rgb_path"])

    out_dir = videos_chunk_root / f"observation.images.top_head"
    ensure_dir(out_dir)

    # Build per-frame lists
    frames_camera_structs = { cam: [] for cam in camera_to_srcpath.keys() }
    states_list = []
    actions_list = []
    mano_keypoints = []
    camera_intrinsics = []
    camera_extrinsics = []
    timestamps = []
    next_done = []
    frame_indices = []
    episode_indices = []
    global_indices = []

    for fi in range(n_frames):
        ts = float(fi / FPS)
        timestamps.append(ts)
        frame_indices.append(fi)
        episode_indices.append(ep_idx)
        global_indices.append(global_index_start + fi)
        next_done.append(fi == (n_frames - 1))

        # video symlink/copy once per episode per camera
        out_dir = videos_chunk_root / f"observation.images.top_head"
        # filename uses triplet & sequence to avoid collisions
        episode_filename = f"episode_{ep_idx:06d}.mp4"
        target_video_for_episode = out_dir / episode_filename
        frames_camera_structs["top_head"].append({"path": str(target_video_for_episode.resolve()), "timestamp": ts})
        symlink_or_copy(camera_to_srcpath["top_head"], target_video_for_episode)

        # read hand pose & convert it to (2*7, 0)
        rh_pose = rh[f"{fi+1:05d}"]["hand_pose"].unsqueeze(0)
        rh_trans = rh[f"{fi+1:05d}"]["hand_trans"].unsqueeze(0)
        lh_pose = lh[f"{fi+1:05d}"]["hand_pose"].unsqueeze(0)
        lh_trans = lh[f"{fi+1:05d}"]["hand_trans"].unsqueeze(0)
        
        def get_wrist_pose(pose_tensor, trans, mano_layer): # (1,48), (1,3) -> (1,7)
            pose = pose_tensor.numpy().squeeze(0)  # (48,)
            wrist_axisangle = pose[:3]
            R_mat = R.from_rotvec(wrist_axisangle)
            rpy_euler = R_mat.as_euler('xyz', degrees=False)
            r, p, y = rpy_euler
            # 提取手腕的绝对位置
            wrist_pos = rh_joints[0] if mano_layer.side == "right" else lh_joints[0]
            x, y, z = wrist_pos
            return torch.tensor([x, y, z, r, p, y, 0.0])
        
        mano_layer_r = ManoLayer(mano_root=str(models_root / "mano"), 
                               use_pca=False, 
                               ncomps=45, 
                               flat_hand_mean=True, 
                               side="right")
        _, rh_joints = mano_layer_r(rh_pose, right_hand_shape)
        rh_joints = rh_joints + rh_trans # shape is (1,21,3)
        rh_joints = rh_joints.squeeze(0) # (21,3)
        mano_keypoints.append(rh_joints.numpy().tolist())
        rh_wrist_pose = get_wrist_pose(rh_pose, rh_trans, mano_layer_r)

        mano_layer_l = ManoLayer(mano_root=str(models_root / "mano"), 
                               use_pca=False, 
                               ncomps=45, 
                               flat_hand_mean=True, 
                               side="left")
        _, lh_joints = mano_layer_l(lh_pose, left_hand_shape)
        lh_joints = lh_joints + lh_trans
        lh_joints = lh_joints.squeeze(0) # (21,3)
        lh_wrist_pose = get_wrist_pose(lh_pose, lh_trans, mano_layer_l)
        # 拼接左右手手腕姿态，得到(2*7,)的数组
        wrist_poses = torch.cat([lh_wrist_pose, rh_wrist_pose], dim=0)
        states_list.append(wrist_poses.numpy().tolist())

        # action: 占位即可 (2*7, )
        actions_list.append([0.0]*14)

        # camera parameters
        intrinsic_p = parameter_dir / "egocentric_intrinsic.txt"
        extrinsic_p = parameter_dir / "egocentric_frame_extrinsic.npy"
        intrinsic = np.loadtxt(intrinsic_p).reshape((3,3)) 
        extrinsic = np.load(extrinsic_p)[fi].reshape((4,4))
        camera_intrinsics.append(intrinsic.tolist())
        camera_extrinsics.append(extrinsic.tolist())

    # Convert to pyarrow arrays and write parquet
    arrays = {}
    fields = []

    video_paths = [s["path"] for s in frames_camera_structs["top_head"]]
    arrays[f"observation.images.top_head"] = pa.array(video_paths, type=pa.string())
    fields.append(pa.field(f"observation.images.top_head", pa.string()))
    combined_states_list = []
    for ci, ce, mk, eef in zip(camera_intrinsics, camera_extrinsics, mano_keypoints, states_list): # lists of lists
        tmp = []
        tmp.extend([item for row in ci for item in row]) 
        tmp.extend([item for row in ce for item in row])
        tmp.extend([item for row in mk for item in row])
        tmp.extend(eef)
        combined_states_list.append(tmp) # concatenated list
    arrays["observation.state"] = pa.array(combined_states_list, type=pa.list_(pa.float32())); fields.append(pa.field("observation.state", pa.list_(pa.float32())))
    arrays["action"] = pa.array(actions_list, type=pa.list_(pa.float32())); fields.append(pa.field("action", pa.list_(pa.float32())))
    # 补充定义
    arrays["camera.intrinsic"] = pa.array(camera_intrinsics, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("camera.intrinsic", pa.list_(pa.list_(pa.float32()))))
    arrays["camera.extrinsic"] = pa.array(camera_extrinsics, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("camera.extrinsic", pa.list_(pa.list_(pa.float32()))))
    arrays["mano_keypoints"] = pa.array(mano_keypoints, type=pa.list_(pa.list_(pa.float32()))); fields.append(pa.field("mano_keypoints", pa.list_(pa.list_(pa.float32()))))
    arrays["eef.state"] = pa.array(states_list, type=pa.list_(pa.float32())); fields.append(pa.field("eef.state", pa.list_(pa.float32())))
    arrays["eef.action"] = pa.array(actions_list, type=pa.list_(pa.float32())); fields.append(pa.field("eef.action", pa.list_(pa.float32())))
    # 补充定义结束
    arrays["episode_index"] = pa.array(episode_indices, type=pa.int64()); fields.append(pa.field("episode_index", pa.int64()))
    arrays["frame_index"] = pa.array(frame_indices, type=pa.int64()); fields.append(pa.field("frame_index", pa.int64()))
    arrays["timestamp"] = pa.array(timestamps, type=pa.float32()); fields.append(pa.field("timestamp", pa.float32()))
    arrays["next.done"] = pa.array(next_done, type=pa.bool_()); fields.append(pa.field("next.done", pa.bool_()))
    arrays["index"] = pa.array(global_indices, type=pa.int64()); fields.append(pa.field("index", pa.int64()))

    schema = pa.schema(fields)
    table = pa.Table.from_arrays([arrays[k] for k in arrays.keys()], schema=schema)

    ensure_dir(out_data_chunk)
    safe_filename = f"episode_{ep_idx:06d}.parquet"
    out_parquet_path = out_data_chunk / safe_filename
    pq.write_table(table, str(out_parquet_path))

    states_np = [np.array(s, dtype=np.float32) for s in states_list]
    max_state_len = max(s.shape[0] for s in states_np)
    states_padded = np.zeros((len(states_np), max_state_len), dtype=np.float32)
    for i,s in enumerate(states_np):
        states_padded[i,:s.shape[0]] = s
    actions_np = [np.array(a, dtype=np.float32) for a in actions_list]
    max_action_len = max(a.shape[0] for a in actions_np)
    actions_padded = np.zeros((len(actions_np), max_action_len), dtype=np.float32)
    for i,a in enumerate(actions_np):
        actions_padded[i,:a.shape[0]] = a

    ep_stats = {
        "episode_index": ep_idx,
        "stats": {
            "action":{
                "min": actions_padded.min(axis=0).tolist(),
                "max": actions_padded.max(axis=0).tolist(),
                "mean": actions_padded.mean(axis=0).tolist(),
                "std": actions_padded.std(axis=0).tolist(),
                "count": [n_frames]
            },
            "observation.state": {
                "min": states_padded.min(axis=0).tolist(),
                "max": states_padded.max(axis=0).tolist(),
                "mean": states_padded.mean(axis=0).tolist(),
                "std": states_padded.std(axis=0).tolist(),
                "count": [n_frames]
            },
            "timestamp": {
                "min": [min(timestamps)],
                "max": [max(timestamps)],
                "mean": [np.mean(timestamps)],
                "std": [np.std(timestamps)],
                "count": [n_frames]
            },
            "frame_index": {
                "min": [min(frame_indices)],
                "max": [max(frame_indices)],
                "mean": [np.mean(frame_indices)],
                "std": [np.std(frame_indices)],
                "count": [n_frames]
            },
            "episode_index": {
                "min": [min(episode_indices)],
                "max": [max(episode_indices)],
                "mean": [np.mean(episode_indices)],
                "std": [np.std(episode_indices)],
                "count": [n_frames]
            },
            "index": {
                "min": [min(global_indices)],
                "max": [max(global_indices)],
                "mean": [np.mean(global_indices)],
                "std": [np.std(global_indices)],
                "count": [n_frames]
            },
            "task_index": {
                "min": [0],
                "max": [0],
                "mean": [0.0],
                "std": [0.0],
                "count": [n_frames]
            }
        }
    }

    # episode-level metadata to include in meta/episodes.jsonl (NOT in parquet)
    episode_meta_for_metajson = {
        "episode_index": ep_idx,
        "length": n_frames,
        "dataset_from_index": global_index_start,
        "dataset_to_index": global_index_start + n_frames - 1,
        "tasks": [triplet_name],
        "triplet_name": triplet_name,
        "sequence_name": sequence_name,
        "relative_path": episode_meta["relative_path"],
        "parquet_path": str(out_parquet_path.resolve()),
    }

    return {
        "n_frames": n_frames,
        "dataset_from_index": global_index_start,
        "dataset_to_index": global_index_start + n_frames - 1,
        "parquet_path": str(out_parquet_path.resolve()),
        "episode_stats": ep_stats,
        "episode_meta_for_metajson": episode_meta_for_metajson
    }

# -----------------------------
# Main conversion driver
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert TACO dataset to LeRobot dataset")
    parser.add_argument("--taco_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()

def main():
    args = parse_args()
    global TACO_ROOT, LEROBOT_SAVE_ROOT, REPO_ID, FPS
    TACO_ROOT = args.taco_root
    LEROBOT_SAVE_ROOT = args.output_root
    REPO_ID = args.repo_id
    FPS = args.fps

    print("Scanning TACO dataset...")
    sequences = scan_taco_sequences(TACO_ROOT)
    print(f"Found {len(sequences)} sequences / episodes")

    out_root = LEROBOT_SAVE_ROOT / REPO_ID
    meta_root = out_root / "meta"
    ensure_dir(out_root)
    ensure_dir(meta_root)

    # Group sequences by triplet_name (each triplet -> chunk)
    from collections import defaultdict
    triplet_groups = defaultdict(list)
    for seq in sequences:
        triplet_groups[seq["triplet_name"]].append(seq)

    episodes_jsonl_path = meta_root / "episodes.jsonl"
    episodes_stats_jsonl_path = meta_root / "episodes_stats.jsonl"
    tasks_jsonl_path = meta_root / "tasks.jsonl"
    info_json_path = meta_root / "info.json"
    models_root = TACO_ROOT / ".." / "models"

    episodes_records = []
    episodes_stats_records = []
    tasks = []
    dataset_global_index = 0  

    # For each triplet group, create chunk folder and write episodes
    for triplet_name, seqs in triplet_groups.items():
        # chunk folder under data and videos
        data_chunk_root = out_root / "data" 
        videos_chunk_root = out_root / "videos"
        ensure_dir(data_chunk_root)
        ensure_dir(videos_chunk_root)

        # process each sequence/episode in this chunk
        for seq in tqdm(sorted(seqs, key=lambda x: x["episode_index"]), desc=f"Converting chunk"):
            try:
                res = build_episode_for_chunk(seq, TACO_ROOT, data_chunk_root, videos_chunk_root, models_root, dataset_global_index)
            except Exception as e:
                print(f"Error converting episode {seq['episode_index']} ({seq['relative_path']}): {e}")
                continue

            episodes_records.append(res["episode_meta_for_metajson"])
            episodes_stats_records.append(res["episode_stats"])
            dataset_global_index += res["n_frames"]

        tasks.append(triplet_name)

    # write tasks.jsonl (unique triplet tasks)
    unique_tasks = sorted(list(set(tasks)))
    with open(tasks_jsonl_path, 'w') as f:
        for i, t in enumerate(unique_tasks):
            rec = {"task_index": i, "task": t}
            f.write(json.dumps(rec) + "\n")

    # write episodes.jsonl
    with open(episodes_jsonl_path, 'w') as f:
        for r in episodes_records:
            f.write(json.dumps(r) + "\n")

    # write episodes_stats.jsonl
    with open(episodes_stats_jsonl_path, 'w') as f:
        for r in episodes_stats_records:
            f.write(json.dumps(r) + "\n")
    
    # build info.json
    total_episodes = len(episodes_records)
    total_frames = sum(r["length"] for r in episodes_records) if total_episodes > 0 else 0

    # collect all camera keys by scanning data directory
    camera_keys = set()

    videos_root = out_root / "videos"
    for chunk_dir in videos_root.iterdir():
        if not chunk_dir.is_dir():
            continue
        for cam_dir in chunk_dir.iterdir():
            if cam_dir.is_dir() and cam_dir.name.startswith("observation.images."):
                cam_key = cam_dir.name  
                camera_keys.add(cam_key)

    # helper: get video resolution (H,W) and channels (C)
    def get_video_shape(path: Path):
        # return (C,H,W)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        c = 3
        cap.release()
        return (c, h, w)

    # build features for observation.images.*
    features = {}

    # each camera key → read one episode video to detect resolution
    for cam_key in sorted(camera_keys):
        # find any mp4/avi inside this folder
        cam_dir = videos_root / "chunk-000"/ cam_key
        sample_video = None
        for f in cam_dir.iterdir():
            if f.suffix.lower() in [".mp4"]:
                sample_video = f
                break
        if sample_video is None:
            # no video found, skip
            continue

        shape = get_video_shape(sample_video)
        if shape is None:
            continue

        C, H, W = shape
        # observation.images.<cam_key> (cam_key already includes prefix)
        features[cam_key] = {
            "dtype": "video",
            "shape": [H, W, C],
            "info": {
                "video.fps": float(FPS),
                "video.height": H,
                "video.width": W,
                "video.channels": C,
                "video.codec": "mpeg4",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }

    features["observation.state"] = {"dtype": "list[float32]", "shape": [102]} # 102=9+16+21*3+2*7
    features["action"] = {"dtype": "list[float32]", "shape": [14]}
    # 补充定义
    features["camera.intrinsic"] = {"dtype": "list[list[float32]]", "shape": [3,3]}
    features["camera.extrinsic"] = {"dtype": "list[list[float32]]", "shape": [4,4]}
    features["mano_keypoints"] = {"dtype": "list[list[float32]]", "shape": [21,3]} 
    features["eef.state"] = {"dtype": "list[float32]", "shape": [14]}
    features["eef.action"] = {"dtype": "list[float32]", "shape": [14]}
    # 补充定义结束
    features["episode_index"] = {"dtype": "int64", "shape": [1]}
    features["frame_index"] = {"dtype": "int64", "shape": [1]}
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    features["next.done"] = {"dtype": "bool", "shape": [1]}
    features["index"] = {"dtype": "int64", "shape": [1]}

    info = {
        "codebase_version": CODEBASE_VERSION,
        "repo_id": REPO_ID,
        "fps": FPS,
        "features": features,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_chunks": len(triplet_groups),
        "total_tasks": len(triplet_groups),
        "total_videos": total_episodes,
        "chunks_size": CHUNKS_SIZE,
        "robot_type": ROBOT_TYPE,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/observation.images.top_head/episode_{episode_index:06d}.mp4",
        "splits": {
            "train": "0:10000000000", 
            }
    }

    with open(info_json_path, 'w') as f:
        json.dump(info, f, indent=2)

    print("Conversion finished.")
    print(f"Saved LeRobot v2.1 dataset to: {out_root}")
    print(f"meta/episodes.jsonl: {episodes_jsonl_path}")
    print(f"meta/episodes_stats.jsonl: {episodes_stats_jsonl_path}")
    print(f"meta/info.json: {info_json_path}")

    # def get_lerobot_dataset_class():
    #     try:
    #         module = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
    #     except ModuleNotFoundError as exc:
    #         raise ImportError(
    #             "需要安装 'huggingface-lerobot' 才能生成 LeRobot 数据集：pip install huggingface-lerobot"
    #         ) from exc
    #     return getattr(module, "LeRobotDataset")
    
    # LeRobotDataset = get_lerobot_dataset_class()
    # dataset = LeRobotDataset.create(
    #     repo_id = REPO_ID,
    #     root = str(out_root),
    #     fps = FPS,
    #     robot_type = ROBOT_TYPE,
    #     features = features,
    # )
    # api = HfApi()
    # api.upload_folder(
    #     folder_path=LEROBOT_SAVE_ROOT,
    #     repo_id=REPO_ID,
    #     repo_type="dataset"
    # )
    # print(f"Uploaded dataset to Hugging Face Hub: {REPO_ID}")


if __name__ == "__main__":
    main()
