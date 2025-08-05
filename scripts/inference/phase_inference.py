import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# === 参数设置 ===
WINDOW_SIZE = 30
MODEL_PATH = "model/swing_phase_lstm_66.h5"
CSV_PATH = "golf_swing_labeled.csv"

# === 加载数据 ===
df = pd.read_csv(CSV_PATH)
df = df[df['phase'] != 'Impact']  # 删除 Impact 样本

# === 构造 66维特征（x_0 ~ x_32, y_0 ~ y_32）===
columns_to_keep = [f'x_{i}' for i in range(33)] + [f'y_{i}' for i in range(33)]
features = df[columns_to_keep]
labels = df['phase']

# === 标签编码 ===
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# === 创建滑动窗口数据 ===
def create_sequences(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i+window_size].values)
        y.append(labels_encoded[i + window_size - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels, WINDOW_SIZE)
print(f"✅ 输入特征维度 shape = {X.shape}")

# === 加载模型并预测 ===
model = load_model(MODEL_PATH)
predictions = model.predict(X)
pred_classes = np.argmax(predictions, axis=1)
pred_phases = le.inverse_transform(pred_classes)

# === 加载点评模板 ===
template_dict = {
    "Setup": "站位良好，注意双脚与肩同宽。",
    "Takeaway": "启动挥杆顺畅，保持手臂与身体同步。",
    "Backswing": "后摆充分，注意肩部旋转与重心转移。",
    "Downswing": "下杆有力，保持重心前移并确保左臂伸直。",
    "Follow-through": "完成动作流畅，保持身体平衡与视线朝向目标。"
}

# === 生成点评 ===
comments = []
for i, phase in enumerate(pred_phases):
    frame_index = i + WINDOW_SIZE
    comment = template_dict.get(phase, "无对应点评")
    comments.append((frame_index, phase, comment))

# === 保存为文本文件 ===
with open("phase_comments.txt", "w", encoding="utf-8") as f:
    for frame_index, phase, comment in comments:
        f.write(f"第 {frame_index} 帧 - 阶段识别为：{phase}\n")
        f.write(f"点评：{comment}\n")
        f.write("-" * 30 + "\n")

print("✅ 点评结果已保存为 phase_comments.txt")
