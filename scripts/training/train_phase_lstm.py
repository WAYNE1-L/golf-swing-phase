# train_phase_model.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# === 参数设置 ===
WINDOW_SIZE = 30
CSV_PATH = "golf_swing_labeled.csv"
MODEL_SAVE_PATH = "model/swing_phase_lstm_34.h5"

# === 加载数据 ===
df = pd.read_csv(CSV_PATH)

# 删除不需要的阶段（如 Impact）
df = df[df['phase'] != 'Impact']

# === 提取 17 个关键点的 x/y，共 34 维 ===
keypoints_to_use = [
    0,   # Nose
    11, 12,  # Shoulders
    13, 14,  # Elbows
    15, 16,  # Wrists
    23, 24,  # Hips
    25, 26,  # Knees
    27, 28,  # Ankles
    31, 32   # Heels
]

columns_to_keep = []
for i in keypoints_to_use:
    columns_to_keep.append(f'x_{i}')
    columns_to_keep.append(f'y_{i}')

features = df[columns_to_keep]
labels = df['phase']

# === 编码标签 ===
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

# === 构造序列数据 ===
def create_sequences(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i+window_size].values)
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels_encoded, WINDOW_SIZE)

# One-hot 编码标签
y_cat = to_categorical(y, num_classes=num_classes)

print(f"✅ 训练样本数: {X.shape[0]}, 输入维度: {X.shape[1:]}")

# === 构建模型 ===
model = Sequential()
model.add(LSTM(64, input_shape=(WINDOW_SIZE, 34), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 确保保存目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# === 训练模型并保存 ===
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
model.fit(X, y_cat, batch_size=32, epochs=30, validation_split=0.1, callbacks=[checkpoint])

print("✅ 模型已保存为", MODEL_SAVE_PATH)
