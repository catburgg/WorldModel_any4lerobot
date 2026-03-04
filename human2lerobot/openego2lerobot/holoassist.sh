
rm -rf /mnt/project/world_model/data/HumanData/HoloAssist

python /mnt/home/lvjiangran/zhuwenxuan/any4lerobot/human2lerobot/openego2lerobot/convert_holoassist.py\
    --input_path /mnt/project/public/world_model/dataset/holoassist/holoassist_data\
    --output_path /mnt/project/world_model/data/HumanData/HoloAssist\
    --repo_id "/mnt/afs/lvjiangran/zhuwenxuan/tmp/tmp"\
    --num_workers 24

# ls /mnt/project/world_model/data/HumanData/HoloAssist1