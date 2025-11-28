import os
os.environ["OMP_NUM_THREADS"] = "1"
import subprocess
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

PYTHON_EXE = "/mnt/afs/zhangxuheng/miniconda3/envs/lerobot2.1/bin/python"
CONVERT_SCRIPT = "/mnt/home/zhangxuheng/code/world_model/any4lerobot/utils/version_convert/convert_dataset_v20_to_v21.py"

MAX_WORKERS = 64

def process_single_task(task_dir: Path):
    task_name = task_dir.name
    task_root = str(task_dir.resolve())
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{task_name}.log"
    
    print(f"[Start] {task_name}")
    
    cmd = [
        PYTHON_EXE,
        CONVERT_SCRIPT,
        "--repo-id", task_name,
        "--root", task_root
    ]
    
    try:
        with open(log_file, "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        return True, task_name, None
    except subprocess.CalledProcessError as e:
        return False, task_name, e.stderr
    except Exception as e:
        return False, task_name, str(e)

def main():
    parser = argparse.ArgumentParser(description="Batch convert datasets")
    parser.add_argument("--root_dir", type=Path, help="Parent directory containing task folders")
    args = parser.parse_args()
    
    parent_path = args.root_dir

    subdirs = sorted([p for p in parent_path.iterdir() if p.is_dir()])
    total = len(subdirs)
    
    print(f"Found {total} tasks. Starting parallel execution with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_task, d): d.name for d in subdirs}
        completed_count = 0
        
        for future in as_completed(future_to_task):
            completed_count += 1
            success, task_name, output = future.result()
            
            if success:
                print(f"✅ [{completed_count}/{total}] Task {task_name} completed.")
            else:
                print(f"[{completed_count}/{total}] Task {task_name} FAILED.")
                print(f"Error log: {output}\n")

    print("All tasks processing finished.")

if __name__ == "__main__":
    main()