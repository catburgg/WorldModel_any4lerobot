rm -rf ./result
python convert_hot3d.py\
                --input_path /mnt/afs/zhuwenxuan/hot3d/hot3d/dataset\
                --output_path /mnt/afs/zhuwenxuan/hot3d2lero/result\
                --repo_id "tmp/tmp"\
                --num_workers 16\
                --assets_file_path /mnt/afs/zhuwenxuan/hot3d/hot3d/dataset/assets\
                --mano_file_path /mnt/afs/zhuwenxuan/mano_v1_2/models

