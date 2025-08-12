# golf_pose_export_full.py
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import argparse
import matplotlib.pyplot as plt

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='Input video path')
parser.add_argument('--output', type=str, required=True, help='Output CSV path')
args = parser.parse_args()

video_path = args.video
output_csv = args.output

# 初始化 mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Cannot open {video_path}")
    exit()

frame_data = []
frame_idx = 0
elbow_angles = []

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        row = [frame_idx]
        for i in range(33):
            row.extend([lm[i].x, lm[i].y])  # 可加 z, visibility

        # 计算左手肘角度
        def get(p): return np.array([lm[p].x * w, lm[p].y * h])
        shoulder = get(mp_pose.PoseLandmark.LEFT_SHOULDER)
        elbow = get(mp_pose.PoseLandmark.LEFT_ELBOW)
        wrist = get(mp_pose.PoseLandmark.LEFT_WRIST)

        def angle(a, b, c):
            ba = a - b
            bc = c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

        elbow_angle = angle(shoulder, elbow, wrist)
        row.append(elbow_angle)
        elbow_angles.append(elbow_angle)

        frame_data.append(row)

    frame_idx += 1

cap.release()
pose.close()

# 写入 CSV
header = ['Frame']
for i in range(33):
    header.extend([f'x{i}', f'y{i}'])
header.append('ElbowAngle')

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(frame_data)

print(f"✅ Saved: {output_csv}")

# 绘制肘部角度随时间变化图
plt.plot(elbow_angles)
plt.title('Left Elbow Angle Over Time')
plt.xlabel('Frame')
plt.ylabel('Angle (deg)')
plot_name = 'elbow_angle_plot.png'
plt.savefig(plot_name)
print(f"✅ Saved: {plot_name}")
