import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# 初始化 mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 更完整的骨柄连接
CUSTOM_CONNECTIONS = frozenset([
    (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER),
])

# 计算角度

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

# 判断阶段

def detect_phase(lw_y, ls_y, arm_angle, velocity):
    if lw_y > ls_y + 50 and arm_angle < 70:
        return "Preparation"
    elif velocity > 3 and arm_angle > 110:
        return "Downswing"
    elif lw_y < ls_y - 50 and arm_angle > 90:
        return "Backswing"
    else:
        return "Impact / Follow-through"

# 错误提示

def detect_error(left_elbow_angle, right_elbow_y, right_shoulder_y):
    tips = []
    if left_elbow_angle > 160:
        tips.append("Left arm too straight")
    if right_elbow_y < right_shoulder_y - 40:
        tips.append("Right elbow too high")
    return "; ".join(tips)

# 视频路径
video_path = 'my_golf.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("\u274c Cannot open video.")
    exit()

# 检测是否是竖屏，如果是，转动
ret, frame = cap.read()
if not ret:
    print("\u274c Cannot read frame.")
    exit()

if frame.shape[0] > frame.shape[1]:
    print("⟳ Rotating video: detected portrait orientation")
    rotate_video = True
else:
    rotate_video = False

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置读视频指针

# CSV 输出
output_csv = 'golf_pose_export.csv'
csv_fields = ['Frame', 'Phase', 'ElbowAngle', 'Errors']
csv_rows = []

prev_lw_y = None
frame_idx = 0

with mp_pose.Pose(model_complexity=1, enable_segmentation=False,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 自动旋转
        if rotate_video:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        h, w, _ = frame.shape

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            def get(p): return [lm[p].x * w, lm[p].y * h]

            lw = get(mp_pose.PoseLandmark.LEFT_WRIST)
            le = get(mp_pose.PoseLandmark.LEFT_ELBOW)
            ls = get(mp_pose.PoseLandmark.LEFT_SHOULDER)
            re = get(mp_pose.PoseLandmark.RIGHT_ELBOW)
            rs = get(mp_pose.PoseLandmark.RIGHT_SHOULDER)

            angle = calc_angle(ls, le, lw)
            lw_y = lw[1]
            ls_y = ls[1]
            re_y = re[1]
            rs_y = rs[1]

            velocity = abs(lw_y - prev_lw_y) if prev_lw_y else 0
            prev_lw_y = lw_y

            phase = detect_phase(lw_y, ls_y, angle, velocity)
            error = detect_error(angle, re_y, rs_y)

            csv_rows.append({
                'Frame': frame_idx,
                'Phase': phase,
                'ElbowAngle': round(angle, 2),
                'Errors': error
            })

            for connection in CUSTOM_CONNECTIONS:
                p1 = get(connection[0])
                p2 = get(connection[1])
                cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), (200, 200, 200), 2)

            for point in CUSTOM_CONNECTIONS:
                for idx in point:
                    p = get(idx)
                    cv2.circle(frame, tuple(map(int, p)), 5, (0, 255, 0), -1)

        frame_idx += 1
        cv2.imshow("Golf Pose Export", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

with open(output_csv, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"✅ Export complete: {output_csv}")
