import os
import subprocess

# 设置视频文件夹路径
video_dir = 'training_data'
video_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

print(f"🔍 Found {len(video_list)} video files.")

for video_file in video_list:
    input_path = os.path.join(video_dir, video_file)
    output_csv = os.path.splitext(input_path)[0] + '.csv'

    print(f"📽️ Processing: {input_path}")
    subprocess.run([
        'python', 'golf_pose_export_full.py',
        '--video', input_path,
        '--output', output_csv
    ])
