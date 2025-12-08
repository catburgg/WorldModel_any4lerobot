# EgoVLA to LeRobot Converter

This script converts EgoVLA finetune data (HuggingFace Dataset format) to LeRobot format.

## Features

- Converts images to video format (10Hz by default)
- Extracts state and action data
- Preserves meta information (language instructions, etc.)
- Groups data by episode (seq_name)
- Handles end-effector (hand) state and action

## Requirements

```bash
pip install lerobot datasets opencv-python pillow numpy tqdm
```

## Usage

```bash
python egovla2lerobot.py \
    --data-path /path/to/HF_hand_FIXED_SET_MIX_train \
    --image-path /path/to/HF_images \
    --image-mapping-path /path/to/hf_images_mapping.pkl \
    --output-path /path/to/output/lerobot_dataset \
    --target-fps 10 \
    --source-fps 30 \
    --max-episodes 100  # Optional: limit number of episodes
```

## Arguments

- `--data-path`: Path to HuggingFace Dataset with hand/state data
- `--image-path`: Path to HuggingFace Dataset with images
- `--image-mapping-path`: Path to image mapping pickle file (maps seq_name and frame_count to image indices)
- `--output-path`: Path to save LeRobot dataset
- `--target-fps`: Target FPS for output videos (default: 10)
- `--source-fps`: Source FPS of input data (default: 30)
- `--max-episodes`: Maximum number of episodes to convert (optional, default: all)

## Output Format

The script creates a LeRobot dataset with the following structure:

- `observation.images.main`: Video file (MP4 format, 10Hz)
- `observation.state`: State vector (qpos + ee poses)
- `action`: Action vector (delta from current to next state)
- `eef.state`: End-effector state (left + right hand, 12 dims: 6 per hand)
- `eef.action`: End-effector action (delta, 12 dims)
- `annotation.language.action_text`: Language instruction

## Data Mapping

The script extracts data from EgoVLA format as follows:

### State
- `curent_qpos` → part of `observation.state`
- `current_left_ee_pose` → part of `observation.state`
- `current_right_ee_pose` → part of `observation.state`

### Action
- Computed as delta from current to next frame

### End-Effector State
- `current_left_mano_trans` + `current_left_mano_rot` → `eef.state` (first 6 dims)
- `current_right_mano_trans` + `current_right_mano_rot` → `eef.state` (last 6 dims)

### End-Effector Action
- Computed as delta from current to next frame for both hands

### Images
- `rgb_obs` from image dataset → converted to video file

## Notes

- The script automatically downsamples from source FPS (30Hz) to target FPS (10Hz)
- Episodes are grouped by `seq_name`
- Frames within each episode are sorted by `frame_count`
- Missing images are replaced with black images
- Missing state/action data is replaced with zeros

