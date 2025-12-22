rm -rf /mnt/afs/lvjiangran/zhuwenxuan/HOI4D_convert
python /mnt/afs/lvjiangran/zhuwenxuan/any4lerobot/human2lerobot/openego2lerobot/convert_hoi4d.py\
    --input_path /mnt/afs/lvjiangran/zhuwenxuan/HOI4D\
    --output_path /mnt/afs/lvjiangran/zhuwenxuan/HOI4D_convert\
    --mano_path /mnt/afs/lvjiangran/zhuwenxuan/mano_v1_2/models\
    --repo_id /mnt/afs/lvjiangran/zhuwenxuan/tmp/tmp\
    --num_workers 32