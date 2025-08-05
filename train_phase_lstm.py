# train_phase_lstm.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
CSV_PATH = "golf_swing_labeled.csv"
MODEL_SAVE_PATH = "phase_lstm_model.h5"
WINDOW_SIZE = 10

# === 读取数据 ===
df = pd.read_csv(CSV_PATH)
df = df[df['phase'] != 'Impact'].reset_index(drop=True)

# === 选择关键点索引（只保留重要部位）===
selected_indices = [11,12,13,14,15,16,23,24,25,26,27,28]

X_all = []
for _, row in df.iterrows():
    features = []
    for idx in selected_indices:
        features.extend([row[f'x_{idx}'], row[f'y_{idx}'], row[f'z_{idx}']])
    X_all.append(features)

X_all = np.array(X_all)
y_all = df['phase'].values

# === 标签编码 ===
le = LabelEncoder()
y_encoded = le.fit_transform(y_all)
y_categorical = to_categorical(y_encoded)

# === 构造序列样本 ===
X_seq, y_seq = [], []
for i in range(WINDOW_SIZE, len(X_all)):
    X_seq.append(X_all[i - WINDOW_SIZE:i])
    y_seq.append(y_categorical[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# === 划分训练集 ===
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# === LSTM 模型 ===
model = Sequential([
    Masking(mask_value=0.0, input_shape=(WINDOW_SIZE, X_all.shape[1])),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# === 保存模型 ===
model.save(MODEL_SAVE_PATH)
print(f"✅ 模型已保存至 {MODEL_SAVE_PATH}")

# === 输出评估 ===
y_pred = model.predict(X_test)
y_pred_class = y_pred.argmax(axis=1)
y_true_class = y_test.argmax(axis=1)

print(classification_report(y_true_class, y_pred_class, target_names=le.classes_))

# 混淆矩阵
cm = confusion_matrix(y_true_class, y_pred_class)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues', fmt='d')
plt.title("混淆矩阵")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
