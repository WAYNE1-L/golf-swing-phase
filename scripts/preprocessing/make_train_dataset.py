import os
import pandas as pd
import numpy as np

# 参数设置
csv_folder = "training_data"  # 存放 CSV 文件的文件夹
output_npz = "train/train_data.npz"  # 输出路径
sequence_length = 10  # 每个样本使用连续10帧

# 动作阶段标签映射
label_map = {
    "Preparation": 0,
    "Backswing": 1,
    "Downswing": 2,
    "Impact": 3,
    "Follow-through": 4
}

X_data = []
y_data = []

# 遍历所有 csv 文件
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_folder, filename))

        if "Phase" not in df.columns:
            print(f"⛔ Missing 'Phase' column in: {filename}")
            continue

        coords = df[[col for col in df.columns if 'x' in col or 'y' in col]].values
        labels = df["Phase"].map(label_map).values

        for i in range(len(coords) - sequence_length):
            seq_x = coords[i:i + sequence_length]
            seq_y = labels[i + sequence_length - 1]
            X_data.append(seq_x)
            y_data.append(seq_y)

# 转为 numpy 数组
X_data = np.array(X_data)  # [样本数, 10帧, 66维]
y_data = np.array(y_data)

# 保存为 .npz 文件
os.makedirs("train", exist_ok=True)
np.savez(output_npz, X=X_data, y=y_data)
print(f"✅ Saved dataset: {output_npz} ({X_data.shape[0]} samples)")
