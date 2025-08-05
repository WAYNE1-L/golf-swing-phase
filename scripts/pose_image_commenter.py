import cv2
import mediapipe as mp
import numpy as np
import os

# === 配置 ===
IMG_PATH = "D:/GOLF/input_image.png"
SAVE_PATH = "D:/GOLF/output_annotated.png"
TEXT_PATH = "D:/GOLF/comment.txt"

# === 检查路径 ===
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"图像路径不存在: {IMG_PATH}")

# === 读取图像 ===
image = cv2.imread(IMG_PATH)
if image is None:
    raise ValueError("图像读取失败，检查文件路径或格式。")

# === 初始化 MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# === 姿态识别 ===
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
if not results.pose_landmarks:
    raise ValueError("未检测到人体姿势关键点。")

landmarks = results.pose_landmarks.landmark

# === 角度计算函数 ===
def calc_angle(p1, p2, p3):
    a = np.array([p1.x - p2.x, p1.y - p2.y])
    b = np.array([p3.x - p2.x, p3.y - p2.y])
    cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

# === 提取关键角度 ===
right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

elbow_angle = calc_angle(right_shoulder, right_elbow, right_wrist)

# === 自动点评逻辑 ===
if elbow_angle < 150:
    quick_fix = "Your trail arm bends too much during takeaway."
    pro_tip = "Keep your trail arm straighter like the Pro for a wider arc."
else:
    quick_fix = "Good arm extension!"
    pro_tip = "Maintain that wide takeaway arc."

# === 绘图输出 ===
annotated = image.copy()
mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# 标注角度
cx, cy = int(right_elbow.x * image.shape[1]), int(right_elbow.y * image.shape[0])
cv2.putText(annotated, f"{int(elbow_angle)} deg", (cx - 20, cy - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 保存图像
cv2.imwrite(SAVE_PATH, annotated)

# 保存点评
with open(TEXT_PATH, "w") as f:
    f.write(f"Quick Fix: {quick_fix}\nPro Tip: {pro_tip}")

print("✅ 分析完成，已生成：")
print(f"- 图像：{SAVE_PATH}")
print(f"- 点评：{TEXT_PATH}")
