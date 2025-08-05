import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

# ==== 参数设置 ====
CSV_PATH = "golf_swing_labeled.csv"
MODEL_PATH = "model/swing_phase_lstm_66.h5"
WINDOW_SIZE = 30

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)
df = df[df['phase'] != 'Impact']  # 删除 Impact 样本

# ==== 构造 66维特征 ====
keypoints = list(range(33))  # x_0 到 x_32 和 y_0 到 y_32
columns_to_use = [f'x_{i}' for i in keypoints] + [f'y_{i}' for i in keypoints]

features = df[columns_to_use]
labels = df['phase']

# ==== 标签编码 ====
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

# ==== 创建滑动窗口 ====
def create_sequences(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i+window_size].values)
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels_encoded, WINDOW_SIZE)
print(f"✅ 训练样本数: {len(X)}, 输入维度: {X.shape[1:]}")

# ==== One-hot 编码 ====
y_cat = to_categorical(y, num_classes=num_classes)

# ==== 构建模型 ====
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, X.shape[2])))
model.add(LSTM(64))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==== 模型保存路径 ====
os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)

# ==== 训练模型 ====
model.fit(X, y_cat, batch_size=32, epochs=30, validation_split=0.1, callbacks=[checkpoint])

print(f"✅ 模型已保存至：{MODEL_PATH}")
