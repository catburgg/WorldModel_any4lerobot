rm -rf /mnt/afs/lvjiangran/zhuwenxuan/holoassist/result

python convert_holoassist.py\
                --input_path /mnt/afs/lvjiangran/zhuwenxuan/holoassist/holoassist_data\
                --output_path /mnt/afs/lvjiangran/zhuwenxuan/holoassist/result\
                --repo_id "tmp/tmp"\
                --num_workers 16