from modelscope.hub.snapshot_download import snapshot_download

# Retry logic wrapper
max_retries = 5
for i in range(max_retries):
    try:
        print(f"Attempt {i+1} of {max_retries}...")
        snapshot_download(
            'lejurobot/let_dataset',
            cache_dir='/mnt/project/public/world_model/dataset/LET',
            local_dir='/mnt/project/public/world_model/dataset/LET',
            # This ignores the 'pkg_resources' warning
            ignore_file_pattern=['.git', '.gitattributes'] 
        )
        print("Download Success!")
        break
    except Exception as e:
        print(f"Error: {e}")
        import time
        time.sleep(5) # Wait 5 seconds before retrying