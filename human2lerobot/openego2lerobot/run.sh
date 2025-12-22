CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate lerobot

echo "Active environment: $CONDA_DEFAULT_ENV"

./hoi4d.sh
./holoassist.sh
./hot3d.sh