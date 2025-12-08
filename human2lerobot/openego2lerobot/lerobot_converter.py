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
    def __init__(self, output_root: str, repo_id: str, fps: int, robot_type: str, num_workers: int):
        self.output_root = Path(output_root)
        self.repo_id = repo_id
        self.num_workers = num_workers
        self.dataset = None
        self.fps = fps
        self.robot_type = robot_type
        self.temp_dir = self.output_root / "tmp"
        
        self.vocab_db = {
            "task_text": {},   # eg: {"Pick up phones": 0, "Open door": 1}
            "action_text": {}
        }

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
        
        self._save_vocabularies()

    def _worker_wrapper(self, entry):
        try:
            return self.process_entry(entry)
        except Exception as e:
            print(f"Worker Error on {entry}: {e}")
            return None

    def _encode_text_feature(self, raw_data: Union[str, List[str]], vocab_key: str, num_frames: int) -> torch.Tensor:
        """
        Converts raw strings to indices using the self.vocab_db.
        """
        mapping = self.vocab_db[vocab_key]
        
        if isinstance(raw_data, str):
            data_list = [raw_data] * num_frames
        else:
            if len(raw_data) != num_frames:
                raise ValueError(f"Length of {vocab_key} list ({len(raw_data)}) != num_frames ({num_frames})")
            data_list = raw_data

        indices = []
        for text in data_list:
            if text not in mapping:
                mapping[text] = len(mapping)
            indices.append(mapping[text])
            
        return torch.tensor(indices, dtype=torch.int64).unsqueeze(1)

    def _register_episode(self, episode: ConvertibleEpisode):
        """
        Injects data into LeRobot. 
        Converts text -> int here (Main Process) to avoid Race Conditions.
        """
        
        task_indices = self._encode_text_feature(
            episode.task_text, "task_text", episode.num_frames
        )
        episode.data_dict["annotation.language.task_text"] = task_indices

        action_indices = self._encode_text_feature(
            episode.action_text, "action_text", episode.num_frames
        )
        episode.data_dict["annotation.language.action_text"] = action_indices

        task_name = episode.task_name

        for i in range(episode.num_frames):
            frame_data = {
                key: tensor[i] for key, tensor in episode.data_dict.items()
            }
            self.dataset.add_frame(frame_data, task=task_name)

        self.dataset.save_episode()

        ep_idx = self.dataset.meta.total_episodes - 1
        chunk_idx = ep_idx // 1000

        for video_key, temp_path in episode.video_paths.items():
            rel_path = f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{ep_idx:06d}.mp4"
            dest_path = self.dataset.root / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(temp_path, dest_path)

    def _save_vocabularies(self):
        """Saves the String <-> Int mappings to the dataset root."""
        vocab_path = self.output_root / "vocabularies.json"
        inverted_db = {}
        for key, mapping in self.vocab_db.items():
            inverted_db[key] = {
                "s2i": mapping,
                "i2s": {v: k for k, v in mapping.items()}
            }
            
        with open(vocab_path, 'w') as f:
            json.dump(inverted_db, f, indent=2)
        print(f"Vocabularies saved to {vocab_path}")