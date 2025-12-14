import cv2
import numpy as np
import tqdm
import shutil
import json
import torch
import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

# 用于生成 annotations_map.json ，将任务描述语句转为编号
class SentenceRegistry:
    def __init__(self):
        self.sentence_to_id = {}
        self.id_to_sentence = {}
        self.next_id = 0

    def get_id(self, sentence):
        if sentence not in self.sentence_to_id:
            self.sentence_to_id[sentence] = self.next_id
            self.id_to_sentence[self.next_id] = sentence
            self.next_id += 1
        return self.sentence_to_id[sentence]

    def to_dict(self):
        return self.id_to_sentence

class EgoMe2LeRobot:
    def __init__(self, input_root: Path, output_root: Path, fps: int, repo_id: str):
        self.input_root = input_root
        self.output_root = output_root
        self.fps = fps
        self.repo_id = repo_id
        self.features = {
            "eef.left.wrist": {
                "dtype": "float32",
                "shape": (6,),
            },
            "eef.right.wrist": {
                "dtype": "float32",
                "shape": (6,),
            },
            "eef.left.hand": {
                "dtype": "float32",
                "shape": (21, 3),
            },
            "eef.right.hand": {
                "dtype": "float32",
                "shape": (21, 3),
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
                "dtype": "int64",
                "shape": (1,),
            },
            "observation.images.top_head": {
                "dtype": "video",
                "shape": (3, 960, 1280),
                "names": ["channel", "height", "width"],
            },
        }

    def create_lerobot_dataset(self) -> LeRobotDataset:
        dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type="dex",
            features=self.features,
            root=self.output_root,
            use_videos=True,
            image_writer_processes=16,
            image_writer_threads=8
        )
        return dataset
    
    def load_annotations(self):
        ann_path = self.input_root / "Annotation"
        
        p = ann_path / "total.json"
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "annotations" in data:
                    return data["annotations"]
                return data
        
        return {}

    def generate_annotaions_map(self, ego_videos):
        print("Pre-populating sentence registry...")

        task_text_registry = SentenceRegistry()
        action_text_registry = SentenceRegistry()

        for video_id, meta in ego_videos.items():
            coarse_text = meta.get("Coarse-level", "")
            task_text_registry.get_id(coarse_text)
            
            fine_steps = meta.get("Fine-level", [])
            if fine_steps:
                for step in fine_steps:
                    desc = step.get("Step discription", "")
                    action_text_registry.get_id(desc)

        meta_path = self.output_root / "meta"
        meta_path.mkdir(parents=True, exist_ok=True)
        
        maps = {
            "annotation.language.action_text": action_text_registry.to_dict(),
            "annotation.language.task_text": task_text_registry.to_dict()
        }
        
        print("Saving annotation maps...")
        with open(meta_path / "annotation_maps.json", "w", encoding="utf-8") as f:
            json.dump(maps, f, indent=4, ensure_ascii=False)
        print("Saved annotation maps.")

        return task_text_registry, action_text_registry
    
    def get_video_frames(self, video_file, target_fps):
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Failed to open {video_file}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        duration = frame_count / fps
        num_frames = int(duration * target_fps)
        
        # 用 lerobot 提供的工具，按照时间戳提取帧。返回形状 (T, C, H, W) ，值被归一化到 float32 [0, 1]
        timestamps = [t / target_fps for t in range(0, num_frames)]
        frames = decode_video_frames(video_file, timestamps, tolerance_s=1.0 / target_fps)
            
        # 转回 uint8 [0, 255]
        frames = (frames * 255).to(torch.uint8)
        for j, frame in enumerate(frames):
            yield frame, timestamps[j]

    def process_single_episode(self, dataset: LeRobotDataset, video_file: Path, meta, task_text_registry, action_text_registry):
        print(f"Processing video: {video_file.name}")
        frames_gen = self.get_video_frames(video_file, target_fps=self.fps)

        coarse_text = meta.get("Coarse-level", "")
        coarse_id = task_text_registry.get_id(coarse_text)
        fine_steps = meta.get("Fine-level", [])

        for frame_img, timestamp in frames_gen:
            fine_text = ""
            for step in fine_steps:
                desc = step.get("Step discription", "")
                ts = step.get("Step timestamp", [0, 0])
                if ts and len(ts) >= 2:
                    start, end = ts[0], ts[1]
                    if start <= timestamp <= end:
                        fine_text = desc
                        break
            
            fine_id = action_text_registry.get_id(fine_text)

            frame = {
                "eef.left.wrist": np.zeros((6,), dtype=np.float32),
                "eef.right.wrist": np.zeros((6,), dtype=np.float32),
                "eef.left.hand": np.zeros((21, 3), dtype=np.float32),
                "eef.right.hand": np.zeros((21, 3), dtype=np.float32),
                "camera.intrinsic": np.zeros((3, 3), dtype=np.float32),
                "camera.extrinsic": np.zeros((4, 4), dtype=np.float32),
                "annotation.language.action_text": np.array([fine_id], dtype=np.int64),
                "annotation.language.task_text": np.array([coarse_id], dtype=np.int64),
                "observation.images.top_head": frame_img,
            }
            dataset.add_frame(frame, task=meta.get("Action", ""))

        dataset.save_episode()
        
    def convert(self):
        dataset = self.create_lerobot_dataset()

        # 加载注释
        annotations = self.load_annotations()
        # 筛选 Ego 视角视频
        ego_videos = {k: v for k, v in annotations.items() if isinstance(v, dict) and v.get("View") == "Ego"}
        print(f"Found {len(ego_videos)} ego videos.")
        # 生成 annotations_map.json
        (task_text_registry, action_text_registry) = self.generate_annotaions_map(ego_videos)

        for video_id, meta in tqdm.tqdm(ego_videos.items()):
            video_file = self.input_root / "Video" / video_id
            if not video_file.exists():
                continue
            self.process_single_episode(dataset, video_file, meta, task_text_registry, action_text_registry)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to the EgoMe dataset root directory."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to save the converted LeRobot dataset."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frame rate for the LeRobot dataset."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="user/EgoMe2LeRobot_dataset",
        help="Huggingface repo ID for the LeRobot dataset."
    )
    args = parser.parse_args()
    return args.input_root, args.output_root, args.fps, args.repo_id

def main():
    input_root, output_root, fps, repo_id = parse_args()
    input_root = Path(input_root)
    output_root = Path(output_root)

    # 删除已有的数据集，应当只在测试时保留
    # if output_root.exists():
    #     shutil.rmtree(output_root)
    #     print(f"Removed existing dataset at {output_root}")
        
    converter = EgoMe2LeRobot(
        input_root=input_root,
        output_root=output_root,
        fps=fps,
        repo_id=repo_id
    )
    converter.convert()
    
if __name__ == "__main__":
    main()