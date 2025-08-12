import cv2
import mediapipe as mp
import pandas as pd
import os

# === 配置路径 ===
input_dir = "training_data"         # 视频输入目录
output_dir = "keypoints_csv"        # CSV 输出目录
os.makedirs(output_dir, exist_ok=True)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# 遍历所有视频文件
for filename in os.listdir(input_dir):
    if not filename.endswith(".mp4"):
        continue

    video_path = os.path.join(input_dir, filename)
    output_csv = os.path.join(output_dir, filename.replace(".mp4", ".csv"))

    print(f"🎥 正在处理: {filename} ...")
    cap = cv2.VideoCapture(video_path)
    frame_data = []
    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            row = {'frame': frame_index}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f'x_{i}'] = lm.x
                row[f'y_{i}'] = lm.y
                row[f'z_{i}'] = lm.z
                row[f'v_{i}'] = lm.visibility
            frame_data.append(row)

        frame_index += 1

    cap.release()

    # 保存 CSV
    df = pd.DataFrame(frame_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ 导出完成，共 {len(df)} 帧，保存为：{output_csv}")

pose.close()
print("✅ 所有视频处理完成！")
