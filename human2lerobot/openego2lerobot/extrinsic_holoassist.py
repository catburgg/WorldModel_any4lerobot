import os
import glob
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import json
import h5py
from utils import load_hdf5_to_dict
from tqdm import tqdm

@dataclass
class HoloAssistCam:
    """
    Data class representing a single episode's camera data.
    """
    episode_name: str
    width: int
    height: int
    intrinsic: np.ndarray        # Shape (3, 3)
    extrinsics: np.ndarray       # Shape (N, 4, 4) - The camera poses
    timestamps: np.ndarray       # Shape (N,) - The recording timestamps
    num_frames: int
    duration: float

@dataclass
class HoloAssistDat:
    path: Path
    task: str
    width: int
    height: int
    intrinsic: np.ndarray
    num_frames: int
    duration: float

def fit(cam: HoloAssistCam, dat: HoloAssistDat):
    if(abs(cam.duration-dat.duration) > 0.2):
        return 0
    if(not np.allclose(cam.intrinsic, dat.intrinsic, atol=1e-3)):
        return 0

    cam_name = cam.episode_name
    task = dat.task

    def check(keywords):
        x1 = any(keyword in cam_name for keyword in keywords)
        x2 = any(keyword in task for keyword in keywords)
        return x1 == x2

    if(check(["coffee", "Coffee", "nespresso", "Nespresso"]) == False):
        return 0
    
    if(check(["DSLR", "dslr"]) == False):
        return 0

    if(check(["switch", "Switch"]) == False):
        return 0
    
    if(check(["circuit", "Circuit"]) == False):
        return 0

    return 1


def parse_intrinsics(file_path: str):
    """
    Parses the single-line Intrinsics.txt file.
    Format: M00..M22 (9 floats) ... other params ... W H (2 ints)
    """
    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None, None, None
            
            parts = line.split()
            values = [float(x) for x in parts]
            
            # 1. Extract Intrinsic Matrix (First 9 values)
            # Row-major order: M00, M01, M02, M10, M11, M12, M20, M21, M22
            K_flat = values[0:9]
            intrinsic_matrix = np.array(K_flat).reshape(3, 3)
            
            # 2. Extract Width and Height (Last 2 values)
            width = int(values[-2])
            height = int(values[-1])
            
            return intrinsic_matrix, width, height
    except Exception as e:
        print(f"Error parsing intrinsics {file_path}: {e}")
        return None, None, None

def parse_poses(file_path: str):
    """
    Parses the Pose_sync.txt file.
    Format per line: Time, OrigTime, M00..M33 (16 floats)
    """
    timestamps = []
    poses = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            # Column 0: Timestamp
            t = float(parts[0])
            timestamps.append(t)
            
            # Columns 2 to 18: 4x4 Matrix elements (16 values)
            # Assuming Row-Major order based on the '0 0 0 1' ending in your example
            matrix_vals = [float(x) for x in parts[2:18]]
            pose_matrix = np.array(matrix_vals).reshape(4, 4)
            poses.append(pose_matrix)
            
        return np.array(timestamps), np.array(poses)
        
    except Exception as e:
        print(f"Error parsing poses {file_path}: {e}")
        return np.array([]), np.array([])

def load_camera(root_dir: Path) -> List[HoloAssistCam]:
    """
    Scans the directory for episode folders and parses cam_info.
    Expected structure: {root_dir}/cam_info/{episode}/Export_py/Video/...
    """
    dataset = []
    
    # Construct search path
    search_path = os.path.join(root_dir, "*", "Export_py", "Video")
    video_dirs = glob.glob(search_path)
    
    print(f"Found {len(video_dirs)} potential cam directories.")

    tmp = 0

    for video_dir in tqdm(video_dirs, desc="Parsing Cams", unit="ep"):
        tmp += 1
        #if(tmp > 300):
        #    break
        # Extract episode name from path
        # Path is .../cam_info/{episode_name}/Export_py/Video
        parts = video_dir.split(os.sep)
        try:
            # Find the index of 'cam_info' and get the next folder
            idx = parts.index("cam_info")
            episode_name = parts[idx + 1]
        except (ValueError, IndexError):
            episode_name = "unknown"

        intrinsics_path = os.path.join(video_dir, "Intrinsics.txt")
        poses_path = os.path.join(video_dir, "Pose_sync.txt")
        
        if not os.path.exists(intrinsics_path) or not os.path.exists(poses_path):
            print(f"Skipping {episode_name}: Missing files.")
            continue
            
        # Parse files
        K, W, H = parse_intrinsics(intrinsics_path)
        times, extrinsics = parse_poses(poses_path)
        
        if K is None or len(times) == 0:
            print(f"Skipping {episode_name}: Parsing failed or empty.")
            continue

        # Create Class Instance
        episode_data = HoloAssistCam(
            episode_name=episode_name,
            width=W,
            height=H,
            intrinsic=K,
            extrinsics=extrinsics,
            timestamps=times,
            num_frames=len(times),
            duration=times[len(times)-1]-times[0]
        )
        
        dataset.append(episode_data)

    return dataset

def load_dataset(root_dir: Path) -> List[HoloAssistDat]:

    dataset = []
    
    search_path = os.path.join(root_dir, "demo_6328")
    input_paths = glob.glob(search_path)

    tmp = 0

    for input_str in tqdm(input_paths, desc="Parsing Episodes", unit="ep"):

        tmp += 1

        input_path = Path(input_str)

        annotation_path = (input_path / "annotation.json").resolve()
        joint_path = (input_path / "joints.hdf5").resolve()
        metadata_path = (input_path / "metadata.hdf5").resolve()

        with open(annotation_path,'r') as f:
            annotation = json.load(f)
        with h5py.File(joint_path,'r') as f:
            joint = load_hdf5_to_dict(f)
        with h5py.File(metadata_path,'r') as f:
            metadata = load_hdf5_to_dict(f)

        intrinsics = joint['intrinsics']
        length = metadata['num_frames']
        video_info = annotation['video_info']

        dataset.append(HoloAssistDat(
            path=input_path,
            width=video_info['width'],
            height=video_info['height'],
            intrinsic=intrinsics,
            num_frames=length,
            task=annotation['task'],
            duration=video_info['duration']
        ))
    return dataset

def save_cam(episode, target_dir: Path):
    
    output_path = target_dir / "camera_info.npz"
    
    np.savez_compressed(
        output_path,
        intrinsic=episode.intrinsic,
        extrinsics=episode.extrinsics,
        metadata=np.array([episode.width, episode.height, episode.num_frames]),
        task=episode.episode_name
    )

if __name__ == "__main__":
    CAM_PATH = Path("/mnt/afs/lvjiangran/zhuwenxuan/holoassist/cam_info")
    DAT_PATH = Path("/mnt/afs/lvjiangran/zhuwenxuan/holoassist/holoassist_data")
    ANN_PATH = Path("/mnt/afs/lvjiangran/zhuwenxuan/holoassist/data-annotation-trainval-v1_1.json")
    
    all_cams = load_camera(CAM_PATH)
    
    print(f"\nSuccessfully loaded {len(all_cams)} episodes of camera data.")
    
    # for ep in all_cams:
    #    print(f"--- Episode: {ep.episode_name} ---")
    #    print(f"Resolution: {ep.width}x{ep.height}")
    #    print(f"Duration: {ep.duration}")
    #    print(f"Intrinsic Matrix:{ep.intrinsic}")
    #    # print(f"First Extrinsic Matrix:\n{ep.extrinsics[0]}")
    


    all_dats = load_dataset(DAT_PATH)

    print(f"\nSuccessfully loaded {len(all_dats)} episodes of openego data.")

    #for ep in all_dats:
    #    print(f"--- Episode: {ep.path.stem} ---")
    #    print(f"Resolution: {ep.width}x{ep.height}")
    #    print(f"Duration: {ep.duration}")
    #    print(f"Intrinsic Matrix:{ep.intrinsic}")

    success = 0
    
    for dat in all_dats:
        fit_cam = []
        cam_path = dat.path / "camera_info.npz"
        cam_path.unlink(missing_ok=True)
        for cam in all_cams:
            if fit(cam, dat):
                fit_cam.append(cam)
        if(len(fit_cam)>1):
            print(f"{len(fit_cam)} Fit cam of ",dat.path.stem)
            for cc in fit_cam:
                print(cc.episode_name)
        if(len(fit_cam)==0):
            print(f"NO Fit cam of ",dat.path.stem)
        if len(fit_cam) == 1:
            save_cam(fit_cam[0], dat.path)
            success += 1
    print(f"{success} episode successfully added extrinsic in total")
            
        
    
    
