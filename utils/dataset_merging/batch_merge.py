import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MERGE_SCRIPT = os.path.join(THIS_DIR, "merge_lerobot_dataset.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root 目录，里面的所有子目录作为 --sources")
    parser.add_argument("--output", required=True, help="merge_lerobot_dataset 的 --output")
    parser.add_argument("--fps", required=True, help="merge_lerobot_dataset 的 --fps")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    subdirs = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]

    cmd = [
        sys.executable,
        MERGE_SCRIPT,
        "--sources",
        *subdirs,
        "--output",
        os.path.abspath(args.output),
        "--fps",
        str(args.fps),
    ]

    log_path = os.path.join(THIS_DIR, "merge.log")
    print(f"[run_merge_wrapper] logs -> {log_path}")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=log_f,
        )
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()