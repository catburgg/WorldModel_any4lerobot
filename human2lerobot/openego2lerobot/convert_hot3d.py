import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torchvision
from pathlib import Path
import numpy as np
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'hot3d', 'hot3d')
if lib_path not in sys.path:
    sys.path.append(lib_path)

from dataset_api import Hot3dDataProvider
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions

from utils import AriaCamera, se3_to_6d
from schemas import Hot3DLeRobotFeatures
from lerobot_converter import BaseDatasetConverter, ConvertibleEpisode

class Hot3DConverter(BaseDatasetConverter):
    def __init__(self, output_root, repo_id, assets_file_path, mano_file_path, fps = 10, num_workers=1):
        super().__init__(output_root, repo_id, fps, "dex", num_workers, 1)

        self.assets_file_path = Path(assets_file_path)
        self.width = 512
        self.height = 512
        if not self.assets_file_path.exists():
            raise FileNotFoundError(f"hot3d assets file(assets) not found: {self.mano_file_path}")
        self.mano_file_path = Path(mano_file_path)
        if not self.mano_file_path.exists():
            raise FileNotFoundError(f"MANO file(mano_v1_2/models) not found: {self.mano_file_path}")
    
    def get_dataset_features(self):
        hot3d_lerobot_features = Hot3DLeRobotFeatures(img_width=self.width, img_height=self.height)
        return hot3d_lerobot_features.get_features()

    def process_entry(self, input_path: Path):
        """
        This runs inside a Worker Process.
        Input: path of episode, eg. */P00xx-xxxxxxxx
        """

        sequence_path = input_path
        object_library_path = self.assets_file_path
        mano_hand_model_path = self.mano_file_path

        object_library = load_object_library(object_library_folderpath=object_library_path)
        mano_hand_model = None
        if mano_hand_model_path is not None:
            mano_hand_model = MANOHandModel(mano_hand_model_path)
        hot3d_data_provider = Hot3dDataProvider(
            sequence_folder=sequence_path,
            object_library=object_library,
            mano_hand_model=mano_hand_model,
        )

        # --- 1. Create Video ---
        camera = AriaCamera(hot3d_data_provider)
        length = (len(camera.timestamps) + 2) //3
        timestamps = camera.timestamps
        video = np.zeros([length, self.height, self.width, 3], dtype=np.uint8)

        for i, timestamp_ns in enumerate(camera.timestamps):
            if (i % 3 != 0):
                continue
            raw_image = camera.device_data_provider.get_image(timestamp_ns, camera.stream_id)
            new_image = camera.undistort_image(raw_image)
            video[i//3] = np.rot90(new_image, k=3, axes=(0, 1))
        
        episode_name = input_path.stem

        # print("DDDDD", episode_name)
        temp_vid_path = self.temp_dir / f"{episode_name}.mp4"
        with open(temp_vid_path, 'wb') as f:
            torchvision.io.write_video(temp_vid_path, video, fps=10)
        
        # --- 2. Calculate ex&intrinsics
        T_device_camera, K = camera.get_calibration()
        T_world_camera = np.zeros([length,4,4])
        device_pose_provider = hot3d_data_provider.device_pose_data_provider
        for i, timestamp_ns in enumerate(timestamps):
            if (i % 3 != 0):
                continue
            headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            headset_pose3d = headset_pose3d_with_dt.pose3d
            T_world_device = headset_pose3d.T_world_device
            T_world_camera[i//3] = T_world_device.to_matrix() @ T_device_camera

        # print("BBBBB", episode_name)

        # --- 3. Calculate hand pose ---
        hand_data_provider = hot3d_data_provider.mano_hand_data_provider
        mano_coord = {
            "left": np.zeros([length,21,3]),
            "right": np.zeros([length,21,3])
        }
        wrist_pose = {
            "left": np.zeros([length,6]),
            "right": np.zeros([length,6])
        }
        for i, timestamp_ns in enumerate(timestamps):
            if (i % 3 != 0):
                continue
            hand_poses_with_dt = hand_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            #if(hand_poses_with_dt is not None):
                # print(i)
            hand_pose_collection = hand_poses_with_dt.pose3d_collection
            for hand_pose_data in hand_pose_collection.poses.values():
                handedness_label = hand_pose_data.handedness_label()
                # T_world_wrist = hand_pose_data.wrist_pose
                hand_landmarks = hand_data_provider.get_hand_landmarks(
                    hand_pose_data
                )
                mano_coord[handedness_label][i//3] = hand_landmarks.numpy()
                wrist_pose[handedness_label][i//3] = se3_to_6d(hand_pose_data.wrist_pose)

        # print("OOOOO", episode_name)

        # --- 4. Get language discription ---

        json_file = input_path / "metadata.json"

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        object_list = data["object_names"]
        sentence = f"Manipulate {', '.join(object_list)}."
        
        return ConvertibleEpisode(
            episode_identifier=str(episode_name),
            num_frames=length,
            task_name=Path(input_path).stem,
            task_text=sentence,
            action_text=sentence,
            data_dict={
                "camera.intrinsic": np.tile(K[np.newaxis, :, :], (length, 1, 1)),
                "camera.extrinsic": T_world_camera,
                "eef.left.wrist": wrist_pose["left"],
                "eef.left.hand": mano_coord["left"],
                "eef.right.wrist": wrist_pose["right"],
                "eef.right.hand": mano_coord["right"],
            },
            video_paths={
                "observation.images.top_head": temp_vid_path
            },
            height = self.height,
            width = self.width
        )
    

def main(
    input_path: Path,
    output_path: Path,
    repo_id: str,
    assets_file_path: Path,
    mano_file_path: Path,
    num_workers: int
):
    converter = Hot3DConverter(output_path,repo_id,assets_file_path,mano_file_path,10,num_workers)

    subpaths = [
        (input_path / item.name) for item in input_path.iterdir() 
        if item.is_dir() and item.name.startswith('P')
    ]
    converter.run(subpaths)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes to use")
    parser.add_argument("--assets_file_path", type=Path, required=True)
    parser.add_argument("--mano_file_path", type=Path, required=True)
    args = parser.parse_args()
    main(**vars(args))