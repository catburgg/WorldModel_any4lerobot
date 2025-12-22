import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import numpy as np
import json
import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from moviepy import VideoFileClip
from moviepy.video.fx import Resize

from utils import se3_to_6d, load_hdf5_to_dict, resample_poses, resample_motion, convert_cam_to_world, resize_intrinsics
from schemas import HoloassistLeRobotFeatures
from lerobot_converter import BaseDatasetConverter, ConvertibleEpisode
from episode_visualizer_debugger import visualize_dataset

class HoloassistConverter(BaseDatasetConverter):
    def __init__(self, output_root, repo_id, fps = 10, num_workers=1):
        super().__init__(output_root, repo_id, fps, "dex", num_workers, 1)

        self.width = 896
        self.height = 504
    
    def get_dataset_features(self):
        hot3d_lerobot_features = HoloassistLeRobotFeatures(img_width=self.width, img_height=self.height)
        return hot3d_lerobot_features.get_features()

    def process_entry(self, input_path: Path):
        """
        This runs inside a Worker Process.
        Input: path of episode, eg. /path/to/dataset/demo0721
        """

        video_path = (input_path / "video.mp4").resolve()
        annotation_path = (input_path / "annotation.json").resolve()
        joint_path = (input_path / "joints.hdf5").resolve()
        metadata_path = (input_path / "metadata.hdf5").resolve()
        cam_path = (input_path / "camera_info.npz").resolve()

        with open(annotation_path,'r') as f:
            annotation = json.load(f)
        with h5py.File(joint_path,'r') as f:
            joint = load_hdf5_to_dict(f)
        with h5py.File(metadata_path,'r') as f:
            metadata = load_hdf5_to_dict(f)
        with np.load(cam_path) as cam_data:
            cam_poses = cam_data['extrinsics']
            task = cam_data['task']
            ow = cam_data['metadata'][0]
            oh = cam_data['metadata'][1]

        action = annotation['actions']
        intrinsics = joint['intrinsics']
        joint_names = joint['joint_names']
        num_action = len(action)
        video_info = annotation['video_info']
        duration = video_info['duration']

        episode_name = str(input_path.stem)+"_"+str(task)

        # --- 1. Save video
        clip = VideoFileClip(video_path)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_vid_path = self.temp_dir / f"{episode_name}.mp4"
        clip_resized = clip.with_effects([Resize(new_size=(896, 504))]).with_fps(10)
        clip_resized.write_videofile(temp_vid_path, codec='libx264', audio=False, logger=None)
        length = int(clip_resized.duration * clip_resized.fps)

        # --- 2. Calculate ex&intrinsics
        K = joint['intrinsics']
        cam_poses = resample_poses(cam_poses, length)
 
        # --- 3. Calculate hand pose ---
        mano_coord = {
            "left": np.zeros([length,21,3]),
            "right": np.zeros([length,21,3])
        }
        left_pts = resample_motion(joint['left_hand'], length)
        right_pts = resample_motion(joint['right_hand'], length)
        #print(left_pts[20:50])
        mano_coord['left'] = convert_cam_to_world(left_pts, cam_poses)
        mano_coord['right'] = convert_cam_to_world(right_pts, cam_poses)

        # --- 4. Get language discription ---

        task = annotation['task']
        action_text = []
        for x in action:
            l = x['start_timestamp']
            r = x['end_timestamp']
            sentence = x['label']
            while len(action_text) < l * self.fps:
                action_text.append("")
            while len(action_text) < r * self.fps:
                action_text.append(sentence)
        while len(action_text) < length:
            action_text.append("")
        
        return ConvertibleEpisode(
            episode_identifier=episode_name,
            num_frames=length,
            task_name=episode_name,
            task_text=task,
            action_text=action_text,
            data_dict={
                "camera.intrinsic": np.tile(K[np.newaxis, :, :], (length, 1, 1)),
                "camera.extrinsic": cam_poses,
                #"eef.left.wrist": wrist_pose["left"],
                "eef.left.hand": mano_coord["left"],
                #"eef.right.wrist": wrist_pose["right"],
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
    num_workers: int
):
    converter = HoloassistConverter(output_path,repo_id,10,num_workers)


    subpaths = [
        (input_path / item.name) 
        for item in input_path.iterdir() 
        if item.is_dir() 
        #and item.name.startswith("demo_")
        #and int(item.name.split('_')[-1]) < 1800
    ]
    # converter.run(subpaths)
    x = converter.run(subpaths)
    # print(input_path / "demo_2422")
    # x = converter.process_entry(input_path / "demo_2422")
    #3 print(x.action_text)
    # dd = x.data_dict
    # dd["video_paths"] = x.video_paths
    # visualize_dataset(dd)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes to use")
    args = parser.parse_args()
    main(**vars(args))