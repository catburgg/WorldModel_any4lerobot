import pandas as pd
import sys
from pathlib import Path
from moviepy import VideoFileClip
from moviepy.video.fx import Resize
import shutil

root = Path("/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob")
all_sheets_dict = pd.read_excel(root / 'captions.xlsx', sheet_name=None)
df = pd.concat(all_sheets_dict.values(), ignore_index=True)

from lerobot_converter import BaseDatasetConverter, ConvertibleEpisode
from schemas import TasteRobFeatures
from moviepy.video.fx import Resize
class TasteRobConverter(BaseDatasetConverter):
    def __init__(self, input_root, output_root, repo_id, fps = 10, num_workers=1):
        super().__init__(output_root, repo_id, fps, "dex", num_workers, 0)

        self.width = 960
        self.height = 540
        self.input_root = input_root
        self.tmp_dir = self.output_root / "tmp"
        self.mapping = df.set_index(['id'])['caption'].to_dict()
    
    def get_dataset_features(self):
        tasterob_features = TasteRobFeatures(img_width=self.width, img_height=self.height)
        return tasterob_features.get_features()
    
    def process_entry(self, entry: Path) -> ConvertibleEpisode:
        vid = Path(entry).name
        caption = self.mapping.get(int(vid.split(".")[-2]), "")
        print(int(vid.split(".")[-2]))

        with VideoFileClip(str(entry)) as clip:
            temp_vid_path = self.temp_dir / f"{vid}"
            clip_resized = clip.with_effects([Resize(new_size=(self.width, self.height))]).with_fps(self.fps)
            clip_resized.write_videofile(temp_vid_path, codec='libx264', audio=False, logger=None)
            length = int(clip_resized.duration * clip_resized.fps)
            clip_resized.close()
        
        return ConvertibleEpisode(
            episode_identifier=vid,
            num_frames=length,
            task_name=entry,
            task_text=caption,
            action_text=caption,
            data_dict={},
            video_paths={
                "observation.images.top_head": temp_vid_path
            },
            height = self.height,
            width = self.width
        )



dataset_list = [
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/DoubleHand/Bathroom",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/DoubleHand/Dinning",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/DoubleHand/Office_55670_to_57854",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Bathroom",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Bedroom",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Dressingtable",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Kitchen_15553_to_19760",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Kitchen_19761_to_26789",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Kitchen_26790_to_30789",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Kitchen_30790_to_38183",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Kitchen_38184_to_43232",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Office_1_to_3043",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Office_3044_to_6052",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/Office_6053_to_9429",
    #"/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/dinning_49808_to_55669",
    "/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/tmp/dinning_77015_to_101800",
    "/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/dinning_101801_to_107847",
    "/mnt/project/public/world_model/dataset/tasterob/1234_lgswdn/TASTE-Rob/SingleHand/supplement",
]
output_root = "/mnt/project/world_model/data/HumanData/TASTE-Rob"
num_workers = 12

if __name__ == "__main__":
    for dataset in dataset_list:
        if not Path(dataset).exists():
            print(f"Dataset path {dataset} does not exist.", file=sys.stderr)
            continue

        scene = dataset.split("/")[-1]
        handy = dataset.split("/")[-2]
        name = f"TasteRob_{handy}_{scene}"
        paths = [
            str(p) for p in Path(dataset).iterdir()
        ]

        converter = TasteRobConverter(
            dataset,
            output_root = Path(output_root) / name,
            repo_id = "tmp/tmp",
            num_workers = num_workers
        )
        shutil.rmtree(Path(output_root) / name, ignore_errors=True)
        converter.run(paths)