#!/bin/bash

# if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/anaconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/miniconda3/etc/profile.d/conda.sh"
# else
#     source ~/.bashrc
# fi
source /mnt/home/lvjiangran/miniconda3/bin/activate
conda activate /mnt/home/lvjiangran/miniconda3/envs/lerobot

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate environment 'lerobot'."
    exit 1
fi

echo "Environment 'lerobot' activated."

echo "Running conversion script..."

python -u /mnt/home/lvjiangran/zhuwenxuan/any4lerobot/human2lerobot/tasterob/convert.py

echo "Job finished."