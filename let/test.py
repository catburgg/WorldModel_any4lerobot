import numpy as np
import cv2
import torch
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from kinematics import KinematicsSolver
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from utils import process_frames

# --- Configuration ---
# Path to your downloaded .bag files
BAG_DIR = Path("/mnt/project/public/world_model/dataset/LejuRobotics/real/Unlabelled/Parts_offline-P4-dex_hand")
# Where to save the converted LeRobot dataset
OUTPUT_DIR = Path("/mnt/home/lvjiangran/zhuwenxuan/let/results")
REPO_ID = "tmp/tmp"

# Define the feature spec for LeRobot
# Adjust image sizes (e.g., 640x480) based on the actual data
FEATURES = {
    "observation.images.head": {"dtype": "video", "shape": [3, 480, 640], "names": ["c", "h", "w"]},
    "observation.images.left_wrist": {"dtype": "video", "shape": [3, 480, 640], "names": ["c", "h", "w"]},
    "observation.images.right_wrist": {"dtype": "video", "shape": [3, 480, 640], "names": ["c", "h", "w"]},
    "observation.state": {"dtype": "float32", "shape": [14], "names": ["motors"]},  # Example: 14 arm joints
    "action": {"dtype": "float32", "shape": [14], "names": ["motors"]},            # Example: Next state as action
}

def extract_image(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def main():

    # 2. Iterate over Bag Files
    bag_files = sorted(list(BAG_DIR.glob("*.bag")))

    bag_files = bag_files[:1]

    temp_dir = Path("/mnt/home/lvjiangran/zhuwenxuan/let")
    
    for bag_path in bag_files:
        print(f"Processing {bag_path.name}...")
        
        with AnyReader([bag_path]) as reader:
            connections = [x for x in reader.connections if x.topic in [
                '/cam_h/color/image_raw/compressed',
                '/sensors_data_raw',
            ]]

            KL = KinematicsSolver("left")
            KR = KinematicsSolver("right")
            vid = []
            
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                # Deserialize (rosbags requires converting ROS1 payload to CDR first for unified handling)
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                if connection.topic == '/cam_h/color/image_raw/compressed':
                    fm = msg.format
                    data = msg.data
                    vid.append(extract_image(data))

                elif connection.topic == '/sensors_data_raw':
                    joint_q = np.array(msg.joint_data.joint_q, dtype=np.float64)
                    imu_q = np.array([
                        msg.imu_data.quat.x,
                        msg.imu_data.quat.y,
                        msg.imu_data.quat.z,
                        msg.imu_data.quat.w
                    ], dtype=np.float64)
                    base_pos = np.array([0.0, 0.0, 0.0])
                    q_pinocchio = np.concatenate([base_pos, imu_q, joint_q])
                    l_pose = KL.compute_eef_pose(q_pinocchio)
                    r_pose = KR.compute_eef_pose(q_pinocchio)
            
            #vid = process_frames(vid)
            clip = ImageSequenceClip(vid, fps=30).with_fps(10)
            temp_path = temp_dir / f"{bag_path.name}.mp4"
            clip.write_videofile(temp_path, logger=None)
                
                        

    print("Conversion complete!")

if __name__ == "__main__":
    main()