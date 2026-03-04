import argparse
import os
import sys
import shutil
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from torch import FloatTensor
from pathlib import Path
import numpy as np
from typing import List
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import tensorflow_datasets as tfds
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import cv2
import json

from utils import resample_6d, resample_motion, process_frames
from schemas import LetFeatures
from lerobot_converter import BaseDatasetConverter, ConvertibleEpisode
from kinematics import KinematicsSolver
from offset import apply_transform

def extract_image(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

FRAME_INTERVAL_NS = int(1e9 / 10)

class LetConverter(BaseDatasetConverter):
    def __init__(self, output_root, repo_id, fps, num_workers):
        super().__init__(output_root, repo_id, fps, "kuavo", num_workers, 0)

        self.width = 848
        self.height = 480
    
    def get_dataset_features(self):
        features = LetFeatures(img_width=self.width, img_height=self.height)
        return features.get_features()

    def process_entry(self, bag_path):

        print(f"Processing {bag_path.name}...")

        try:

            next_trigger_time = None
            start_time = None
            latest_data = {
                "image": None,
                "joint_q": None,
                "imu_q": None,
                "claw": None,
            }
            action_instr = []
            vid = []
            lw = []
            rw = []
            lg = []
            rg = []
            
            with AnyReader([bag_path]) as reader:
                connections = [x for x in reader.connections if x.topic in [
                    '/cam_h/color/image_raw/compressed',
                    '/sensors_data_raw',
                    '/leju_claw_command',
                ]]
                

                KL = KinematicsSolver("left")
                KR = KinematicsSolver("right")
                
                for connection, timestamp, rawdata in reader.messages(connections=connections):
                    if start_time is None:
                        start_time = timestamp
                        next_trigger_time = start_time

                    msg = reader.deserialize(rawdata, connection.msgtype)

                    if connection.topic == '/cam_h/color/image_raw/compressed':
                        latest_data["image"] = extract_image(msg.data)

                    elif connection.topic == '/sensors_data_raw':
                        latest_data["joint_q"] = np.array(msg.joint_data.joint_q, dtype=np.float64)
                        latest_data["imu_q"] = np.array([
                            msg.imu_data.quat.x, msg.imu_data.quat.y, 
                            msg.imu_data.quat.z, msg.imu_data.quat.w
                        ], dtype=np.float64)
                    
                    elif connection.topic == '/leju_claw_command':
                        #print(dir(msg.data))
                        latest_data["claw"] = msg.data.position

                    if timestamp >= next_trigger_time:
                        if (latest_data["image"] is not None and 
                            latest_data["joint_q"] is not None and 
                            latest_data["imu_q"] is not None and 
                            latest_data["claw"] is not None):
                            
                            #imu_q = np.array([0.0, 0.0, 0.0, 1.0])
                            imu_q = latest_data["imu_q"]
                            motors = latest_data["joint_q"]
                            base_pos = np.array([0.0, 0.0, 0.0])
                            base_vec = np.concatenate([base_pos, imu_q])
                            q_pinocchio = np.concatenate([
                                base_vec,
                                motors[0:19],
                                np.zeros(4),
                                motors[19:26],
                                np.zeros(4),
                                motors[26:28]
                            ])

                            l_pose = KL.compute_eef_pose(q_pinocchio)
                            r_pose = KR.compute_eef_pose(q_pinocchio)
                            l_pose = apply_transform(l_pose,"let")
                            r_pose = apply_transform(r_pose,"let")

                            vid.append(latest_data["image"])
                            lw.append(l_pose)
                            rw.append(r_pose)
                            lg.append(np.array([1-latest_data["claw"][0]/100]).astype(np.float32))
                            rg.append(np.array([1-latest_data["claw"][1]/100]).astype(np.float32))

                        next_trigger_time += FRAME_INTERVAL_NS
                
                clip = ImageSequenceClip(vid, fps=10)
                temp_path = self.temp_dir / f"{bag_path.name}.mp4"
                clip.write_videofile(temp_path, logger=None)
            
            json_path = Path(str(bag_path)[:-4] + ".json")
            print(json_path)
            with open(json_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                # task_instr = data["englishInitSceneText"] + data["taskPrompt"]
                marks = data["marks"]
                for dic in marks:
                    duration = dic["duration"]
                    action = dic["enSkillDetail"]
                    for _ in range(int(duration * 10)):
                        action_instr.append(action)
                if len(action_instr) < len(lw):
                    action_instr += ["none"] * (len(lw) - len(action_instr))
                elif len(action_instr) > len(lw):
                    action_instr = action_instr[:len(lw)]
            if len(action_instr) == 0:
                action_instr = "none" * len(lw)
            
            length = len(lw)

            return ConvertibleEpisode(
                episode_identifier=f"{bag_path.name}",
                num_frames=length,
                task_name=f"{bag_path.name}",
                task_text=action_instr,
                action_text=action_instr,
                data_dict={
                    "eef.left.wrist": lw,
                    "eef.left.hand": lg,
                    "eef.right.wrist": rw,
                    "eef.right.hand": rg,
                },
                video_paths={
                    "observation.images.top_head": temp_path,
                },
                height = self.height,
                width = self.width
            )
        except Exception as e:
            print(f"Failed to process {bag_path.name}: {e}")
            return None

def main(
    input_path: Path,
    #name: str,
    output_path: Path,
    #repo_id: str,
    num_workers: int
):
    #BAG_DIR = Path("/mnt/project/public/world_model/dataset/LejuRobotics/real/Unlabelled/Parts_offline-P4-dex_hand")
    #BAG_DIR = Path("/mnt/project/public/world_model/dataset/LejuRobotics/real/Unlabelled/SMT_tray_rack_blanking-P4-claw")
    #BAG_DIR = Path("/mnt/project/public/world_model/dataset/let_dataset/datasets/rosbag/real/Labelled/quick_sort-P4-claw")
    BAG_DIR = input_path
    OUTPUT_DIR = output_path
    repo_id = "tmp/tmp"
    bag_files = sorted(list(BAG_DIR.glob("*.bag")))
    #bag_files = bag_files[:1]
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    converter = LetConverter(OUTPUT_DIR, repo_id, 10, num_workers)
    
    converter.run(bag_files)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    #parser.add_argument("--name", type=str, required=True)
    #parser.add_argument("--repo_id", type=str, default="tmp/tmp")
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))