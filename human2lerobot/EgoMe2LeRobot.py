import cv2
import numpy as np
import tqdm
import shutil
import json
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

features = {
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

repo_id = "user/EgoMe2LeRobot_dataset"
root_path = Path("./EgoMe2LeRobot_dataset")
original_dataset_path = Path("./EgoMe")
video_path = original_dataset_path / "Video"

# 用于生成 annotations_map.json
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

def load_annotations(root):
    ann_path = root / "Annotation"
    
    p = ann_path / "total.json"
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "annotations" in data:
                return data["annotations"]
            return data
    
    return {}

def get_video_frames(video_file, target_fps=10):
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Failed to open {video_file}")
        return
    
    # 获取帧率与总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration = frame_count / fps
    num_frames = int(duration * target_fps)
    
    # 按块处理，防止内存溢出
    chunk_size = 100
    for i in range(0, num_frames, chunk_size):
        chunk_indices = range(i, min(i + chunk_size, num_frames))
        timestamps = [t / target_fps for t in chunk_indices]
        
        # 用 lerobot 提供的工具，按照时间戳提取帧。返回形状 (T, C, H, W) ，值被归一化到 float32 [0, 1]
        frames = decode_video_frames(video_file, timestamps, tolerance_s=1.0/target_fps)
        
        # 转回 uint8 [0, 255]
        frames = (frames * 255).to(torch.uint8)
        
        for j, frame in enumerate(frames):
            yield frame, timestamps[j]

def main():
    # 删除已有的数据集，应当只在测试时保留
    # if root_path.exists():
    #     shutil.rmtree(root_path)

    # 创建 LeRobot 数据集，指定 fps = 10
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="dex",
        features=features,
        root=root_path,
        use_videos=True
    )

    # 从 EgoMe 数据集加载标注
    annotations = load_annotations(original_dataset_path)
    
    # 筛选所有 Ego 视角视频
    ego_videos = {k: v for k, v in annotations.items() if isinstance(v, dict) and v.get("View") == "Ego"}
    
    print(f"Found {len(ego_videos)} ego videos.")

    # 生成 annotations_map.json
    task_text_registry = SentenceRegistry()
    action_text_registry = SentenceRegistry()

    print("Pre-populating sentence registry...")

    for video_id, meta in ego_videos.items():
        coarse_text = meta.get("Coarse-level", "")
        task_text_registry.get_id(coarse_text)
        
        fine_steps = meta.get("Fine-level", [])
        if fine_steps:
            for step in fine_steps:
                desc = step.get("Step discription", "")
                action_text_registry.get_id(desc)

    meta_path = root_path / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)
    
    maps = {
        "annotation.language.action_text": action_text_registry.to_dict(),
        "annotation.language.task_text": task_text_registry.to_dict()
    }
    
    with open(meta_path / "annotation_maps.json", "w", encoding="utf-8") as f:
        json.dump(maps, f, indent=4, ensure_ascii=False)
    print(f"Saved annotation maps.")

    for video_id, meta in tqdm.tqdm(ego_videos.items()):
        video_file = video_path / video_id
        if not video_file.exists():
            continue
        
        # 获取视频帧，抽帧到 10 fps
        frames_gen = get_video_frames(video_file, target_fps=10)
        
        coarse_text = meta.get("Coarse-level", "")
        coarse_id = task_text_registry.get_id(coarse_text)

        fine_steps = meta.get("Fine-level", [])

        # 查找这一帧对应的细粒度描述
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
    
if __name__ == "__main__":
    main()