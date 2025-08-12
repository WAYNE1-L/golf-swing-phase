import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# ==== 参数设置 ====
CSV_PATH = "golf_swing_labeled.csv"
MODEL_PATH = "model/swing_phase_lstm_66_v2.h5"
WINDOW_SIZE = 30

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)
df = df[df['phase'] != 'Impact']  # 删除 Impact 样本

# ==== 特征列 ====
keypoints = list(range(33))
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
print(f"✅ 样本数: {len(X)}, 输入维度: {X.shape[1:]}")

y_cat = to_categorical(y, num_classes=num_classes)

# ==== 构建模型 ====
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(WINDOW_SIZE, X.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ==== 训练 & 保存 ====
os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model.fit(
    X, y_cat,
    batch_size=16,
    epochs=50,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop]
)

print(f"✅ 模型已保存至：{MODEL_PATH}")
