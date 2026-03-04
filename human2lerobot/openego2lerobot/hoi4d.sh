CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate lerobot

echo "Active environment: $CONDA_DEFAULT_ENV"

rm -rf /mnt/project/world_model/data/HumanData/HOI4D
python /mnt/home/lvjiangran/zhuwenxuan/any4lerobot/human2lerobot/openego2lerobot/convert_hoi4d.py\
    --input_path /mnt/afs/lvjiangran/zhuwenxuan/HOI4D\
    --output_path /mnt/project/world_model/data/HumanData/HOI4D\
    --mano_path /mnt/afs/lvjiangran/zhuwenxuan/mano_v1_2/models\
    --repo_id /mnt/afs/lvjiangran/zhuwenxuan/tmp/tmp\
    --num_workers 8