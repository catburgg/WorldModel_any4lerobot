rm -rf /mnt/afs/lvjiangran/zhuwenxuan/hot3d/result
python convert_hot3d.py\
                --input_path /mnt/afs/lvjiangran/zhuwenxuan/hot3d/hot3d_dataset\
                --output_path /mnt/afs/lvjiangran/zhuwenxuan/hot3d/result\
                --repo_id "tmp/tmp"\
                --num_workers 16\
                --assets_file_path /mnt/afs/lvjiangran/zhuwenxuan/hot3d/hot3d_dataset/assets\
                --mano_file_path /mnt/afs/lvjiangran/zhuwenxuan/mano_v1_2/models

