import cv2
import mediapipe as mp
import pandas as pd
import os

# 配置路径
video_path = "golf_swing.mp4"  # 你的视频文件名
output_csv = "my_golf_keypoints.csv"

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
cap = cv2.VideoCapture(video_path)

# 初始化数据列表
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
pose.close()

# 保存 CSV
df = pd.DataFrame(frame_data)
df.to_csv(output_csv, index=False)
print(f"✅ 导出完成，共 {len(df)} 帧，保存为：{output_csv}")
