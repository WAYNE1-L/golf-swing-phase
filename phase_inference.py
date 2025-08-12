import pandas as pd
import numpy as np
import argparse
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os

# ==== 模型路径（固定为 run48） ====
MODEL_PATH = "model/best_of_50/swing_model_run48.h5"
WINDOW_SIZE = 30
PHASES = ["Setup", "Takeaway", "Backswing", "Downswing", "Follow-through"]

# ==== 参数解析 ====
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="输入CSV路径（新视频关键点）")
parser.add_argument("--output", type=str, required=True, help="输出CSV路径（预测结果）")
args = parser.parse_args()

# ==== 加载模型 ====
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 模型不存在：{MODEL_PATH}")
print(f"✅ 使用模型: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# ==== 读取输入关键点 CSV ====
df = pd.read_csv(args.input)
keypoints = list(range(33))
columns_to_use = [f'x_{i}' for i in keypoints] + [f'y_{i}' for i in keypoints]
features = df[columns_to_use]

# ==== 构建滑动窗口输入 ====
X = []
for i in range(len(features) - WINDOW_SIZE):
    window = features.iloc[i:i+WINDOW_SIZE].values
    X.append(window)
X = np.array(X)

# ==== 模型预测 ====
predictions = model.predict(X, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# ==== 标签反编码 ====
label_encoder = LabelEncoder()
label_encoder.fit(PHASES)
decoded_phases = label_encoder.inverse_transform(predicted_classes)

# ==== 合并回 DataFrame ====
pad = ["Unknown"] * (len(df) - len(decoded_phases))
df["predicted_phase"] = pad + list(decoded_phases)

# ==== 保存结果 ====
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df.to_csv(args.output, index=False)
print(f"✅ 预测完成，已保存至：{args.output}")

# ==== 点评反馈生成 ====
def generate_comment(phases):
    count = Counter(phases)
    total = len(phases)
    comments = []
    for phase in PHASES:
        pct = count[phase] / total * 100
        comments.append(f"{phase}: {pct:.1f}%")
    if count["Downswing"] / total < 0.08:
        comments.append("⚠️ Downswing 时间过短，建议加快提前启动")
    if count["Setup"] / total > 0.25:
        comments.append("⚠️ Setup 阶段过长，可能未及时进入动作")
    return comments

valid_phases = df["predicted_phase"].tolist()[len(df) - len(decoded_phases):]
feedback = generate_comment(valid_phases)
print("\n📋 点评反馈：")
for line in feedback:
    print("•", line)