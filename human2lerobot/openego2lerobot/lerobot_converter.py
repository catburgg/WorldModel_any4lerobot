import abc
import shutil
import json
import dataclasses
import multiprocessing
import copy 
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

@dataclasses.dataclass
class ConvertibleEpisode:
    
    episode_identifier: str
    num_frames: int
    
    data_dict: Dict[str, torch.Tensor]
    
    video_paths: Dict[str, str]
    height: int
    width: int
    
    task_name: str
    task_text: Union[str, List[str]]
    action_text: Union[str, List[str]]
    
    def cleanup(self):
        for path in self.video_paths.values():
            p = Path(path)
            if p.exists():
                p.unlink()

class BaseDatasetConverter(abc.ABC):
    def __init__(self, output_root: str, repo_id: str, fps: int, robot_type: str, num_workers: int, has_main_task: int):
        self.output_root = Path(output_root)
        self.repo_id = repo_id
        self.num_workers = num_workers
        self.dataset = None
        self.fps = fps
        self.robot_type = robot_type
        self.temp_dir = self.output_root / "tmp"
        self.task_list = []

    @abc.abstractmethod
    def get_dataset_features(self) -> Dict:
        pass

    @abc.abstractmethod
    def process_entry(self, entry: Any) -> Optional[ConvertibleEpisode]:
        """
        Input: A file path or ID.
        Output: ConvertibleEpisode with RAW strings in .task_text / .action_text
        """
        pass

    def run(self, raw_data_list: List[Any]):
        """Main execution entry point."""
        features = self.get_dataset_features()
        # features.pop("observation.images.top_head")
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.output_root,
            robot_type=self.robot_type,
            fps=10,
            features=features,
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        with multiprocessing.Pool(self.num_workers) as pool:
            results = pool.imap(self._worker_wrapper, raw_data_list)
            
            for episode in tqdm(results, total=len(raw_data_list), desc="Processing Episodes"):
                if episode:
                    self._register_episode(episode)
        
        self._modify_task()
        

    def _worker_wrapper(self, entry):
        try:
            return self.process_entry(entry)
        except Exception as e:
            print(f"Worker Error on {entry}: {e}")
            return None

    def _register_episode(self, episode: ConvertibleEpisode):
        """
        Injects data into LeRobot. 
        Converts text -> int here (Main Process) to avoid Race Conditions.
        """
        
        task_name = episode.task_name

        self.task_list.append(episode.task_text)

        for i in range(episode.num_frames):
            frame_data = {
                key: tensor[i] for key, tensor in episode.data_dict.items()
            }
            # print(episode.action_text)
            if isinstance(episode.action_text, str):
                self.dataset.add_frame(frame_data, task=episode.action_text)
            else:
                self.dataset.add_frame(frame_data, task=episode.action_text[i])

        self.dataset.save_episode()

        ep_idx = self.dataset.meta.total_episodes - 1
        chunk_idx = ep_idx // 1000

        for video_key, temp_path in episode.video_paths.items():
            rel_path = f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{ep_idx:06d}.mp4"
            dest_path = self.dataset.root / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_path, dest_path)

    def _modify_task(self):
        path = self.output_root / "meta" / "episodes.jsonl"
        with open(path, 'r') as f:
            records = [json.loads(line) for line in f]
        length = len(self.task_list)
        for i in range(length):
            records[i]['tasks'] = self.task_list[i]
        with open(path, 'w') as f:
            for entry in records:
                f.write(json.dumps(entry) + '\n') 