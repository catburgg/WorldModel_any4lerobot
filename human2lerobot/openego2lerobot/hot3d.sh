rm -rf /mnt/project/world_model/data/HumanData/hot3d

python /mnt/home/lvjiangran/zhuwenxuan/any4lerobot/human2lerobot/openego2lerobot/convert_hot3d.py\
    --input_path /mnt/project/public/world_model/dataset/hot3d/hot3d/hot3d_dataset\
    --output_path /mnt/project/world_model/data/HumanData/hot3d\
    --repo_id "/mnt/afs/lvjiangran/zhuwenxuan/tmp/tmp"\
    --num_workers 24\
    --assets_file_path /mnt/project/public/world_model/dataset/hot3d/hot3d/hot3d_dataset/assets\
    --mano_file_path /mnt/afs/lvjiangran/zhuwenxuan/mano_v1_2/models

