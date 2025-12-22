import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from torch import FloatTensor
from pathlib import Path
import numpy as np
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from moviepy import VideoFileClip

from moviepy.video.fx import Resize
from manopth.manolayer import ManoLayer
import pickle

from utils import se3_to_6d, load_hdf5_to_dict, resample_poses, resample_motion, convert_cam_to_world, resize_intrinsics
from schemas import HOI4DLeRobotFeatures
from lerobot_converter import BaseDatasetConverter, ConvertibleEpisode
from episode_visualizer_debugger import visualize_dataset

class HOI4DConverter(BaseDatasetConverter):
    def __init__(self, input_root, output_root, mano_path, repo_id, fps = 10, num_workers=1):
        super().__init__(output_root, repo_id, fps, "dex", num_workers, 0)

        self.width = 960
        self.height = 540
        self.input_root = input_root
        self.mano_path = mano_path
        self.object_map = [
            '', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle',
            'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle',
            'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair'
        ]
    
    def get_dataset_features(self):
        hoi4d_lerobot_features = HOI4DLeRobotFeatures(img_width=self.width, img_height=self.height)
        return hoi4d_lerobot_features.get_features()

    def get_mano(self, pkl_path, side):
        manolayer = ManoLayer(
        mano_root=self.mano_path, use_pca=False, ncomps=45, flat_hand_mean=True, side=side)
        f = open(pkl_path, 'rb')
        hand_info = pickle.load(f, encoding='latin1')
        f.close()

        theta = FloatTensor(hand_info['poseCoeff']).unsqueeze(0)
        beta = FloatTensor(hand_info['beta']).unsqueeze(0)
        trans = FloatTensor(hand_info['trans']).unsqueeze(0)
        hand_verts, hand_joints = manolayer(theta, beta)
        kps3d = hand_joints / 1000.0 + trans.unsqueeze(1)
        # print("KPS3d")
        # print(kps3d)
        return kps3d[0] #, hand_info['kps2D']
    
    def get_mano_episode(self, hand_dir, side):
        ret = np.zeros([300,21,3])
        r2d = np.zeros([300,21,2])
        for i in range(300):
            pkl_path = hand_dir / f"{i}.pickle"
            # print(pkl_path)
            if pkl_path.exists():
                ret[i] = self.get_mano(pkl_path, side)
        return ret #, r2d

    def process_entry(self, input_path: Path):
        """
        This runs inside a Worker Process.
        Input: path of episode, eg. /path/to/dataset/demo0721
        """

        tail = input_path.relative_to(self.input_root)
        camid = tail.parts[0]

        vid_dir = (self.input_root / "HOI4D_release" / tail / "align_rgb").resolve()
        cam_dir = (self.input_root / "camera_params" / camid).resolve()
        ann_dir = (self.input_root / "HOI4D_annotations" / tail).resolve()
        left_dir = (self.input_root / "Hand_pose" / "handpose_left_hand" / tail).resolve()
        right_dir = (self.input_root / "Hand_pose" / "handpose_right_hand" / tail).resolve()
        episode_name = str(tail).replace("/","-")

        if not right_dir.exists(): # There exist some datasets without handpose data.
            return None
        
        print(tail)

        # --- 1. Save video
        video_path = vid_dir / "image.mp4"
        clip = VideoFileClip(video_path)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_vid_path = self.temp_dir / f"{episode_name}.mp4"
        clip_resized = clip.with_effects([Resize(new_size=(self.width, self.height))]).with_fps(self.fps)
        clip_resized.write_videofile(temp_vid_path, codec='libx264', audio=False, logger=None)
        length = int(clip_resized.duration * clip_resized.fps)

        # --- 2. Calculate ex&intrinsics

        cam_path = cam_dir / "intrin.npy"
        K = np.load(cam_path)
        K = resize_intrinsics(K, 1920, 1080, self.width, self.height)

        pose_path = ann_dir / "3Dseg" / "output.log"
        with open(pose_path, 'r') as f:
            raw_lines = [line.strip().split() for line in f if line.strip()]
        matrix_rows = [line for line in raw_lines if len(line) == 4]
        cam_poses = np.array(matrix_rows, dtype=float).reshape(-1, 4, 4)
        cam_poses = resample_poses(cam_poses, length)

        # --- 3. Calculate hand pose ---
        mano_coord = {
            "left": np.zeros([length,21,3]),
            "right": np.zeros([length,21,3])
        }

        right_mano = self.get_mano_episode(right_dir, "right")
        left_mano = self.get_mano_episode(left_dir, "left")
        left_pts = resample_motion(left_mano, length)
        right_pts = resample_motion(right_mano, length)
        mano_coord['left'] = convert_cam_to_world(left_pts, cam_poses)
        mano_coord['right'] = convert_cam_to_world(right_pts, cam_poses)

        # --- 4. Get language discription ---
        object_id = int((str(tail.parts[2]))[1:])
        object_name = self.object_map[object_id]
        
        text_dir = ann_dir / "action" / "color.json"
        with open(text_dir, 'r') as f:
            action_annotation = json.load(f)

        action_text = []
        action_list = []
        for events in action_annotation['events']:
            l = events['startTime']
            r = events['endTime']
            verb = events['event']
            if verb != "Stop" and verb != "rest":
                sentence = f"{verb} {object_name}"
            else:
                sentence = verb
            while len(action_text) < l * self.fps:
                action_text.append("")
            while len(action_text) < r * self.fps:
                action_text.append(sentence)
            action_list.append(sentence)
        while len(action_text) < length:
            action_text.append("")
        

        return ConvertibleEpisode(
            episode_identifier=episode_name,
            num_frames=length,
            task_name=episode_name,
            task_text=action_list,
            action_text=action_text,
            data_dict={
                "camera.intrinsic": np.tile(K[np.newaxis, :, :], (length, 1, 1)),
                "camera.extrinsic": cam_poses,
                # "eef.left.wrist": wrist_pose["left"],
                "eef.left.hand": mano_coord['left'],
                # "eef.right.wrist": wrist_pose["right"],
                "eef.right.hand": mano_coord['right'],
                # "eef.right.2d": right_2d,
                # "eef.left.2d": left_2d
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
    mano_path: Path,
    repo_id: str,
    num_workers: int
):
    converter = HOI4DConverter(input_path, output_path, mano_path, repo_id, 15, num_workers)

    with open(input_path / "release.txt", "r") as f:
        paths = [input_path / line.strip() for line in f if line.strip()]

    converter.run(paths)
    #x = converter.run([])
    # print(paths[-1])
    # x = converter.process_entry(paths[6])
    # print(x.action_text)
    # print(x.data_dict['eef.right.hand'][100])
    # print(x.data_dict['camera.extrinsic'][100])
    # print(x.data_dict['camera.intrinsic'][100])

    # print(x.action_text)
    # dd = x.data_dict
    # dd["video_paths"] = x.video_paths
    # visualize_dataset(dd)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--mano_path", type=Path, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes to use")
    args = parser.parse_args()
    main(**vars(args))