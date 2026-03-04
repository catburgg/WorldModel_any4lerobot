source /mnt/home/lvjiangran/miniconda3/bin/activate
conda activate /mnt/home/lvjiangran/miniconda3/envs/zwx_oxe

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate environment 'zwx_oxe'."
    exit 1
fi

echo "Environment 'zwx_oxe' activated."

python convert_let.py\
    --input_path /mnt/project/public/world_model/dataset/let_dataset/datasets/rosbag/real/Labelled/rubbish_sorting-P4-claw\
    --output_path /mnt/project/world_model/data/RobotData/lejurobot/rubbish_sorting-P4-claw\
    --num_workers 12

python convert_let.py\
    --input_path /mnt/project/public/world_model/dataset/let_dataset/datasets/rosbag/real/Labelled/shop_oversale-P4-claw\
    --output_path /mnt/project/world_model/data/RobotData/lejurobot/shop_oversale-P4-claw\
    --num_workers 12