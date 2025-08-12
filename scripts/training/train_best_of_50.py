import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# ==== 配置 ====
CSV_PATH = "golf_swing_labeled.csv"
WINDOW_SIZE = 30
OUTPUT_DIR = "model/best_of_50"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 16
N_RUNS = 50

# ==== 数据准备 ====
df = pd.read_csv(CSV_PATH)
df = df[df["phase"] != "Impact"]

keypoints = list(range(33))
columns_to_use = [f"x_{i}" for i in keypoints] + [f"y_{i}" for i in keypoints]
features = df[columns_to_use]
labels = df["phase"]

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

def create_sequences(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i+window_size].values)
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels_encoded, WINDOW_SIZE)
y_cat = to_categorical(y, num_classes=num_classes)

results = []

# ==== 多轮训练 ====
for run in range(1, N_RUNS + 1):
    print(f"\n🚀 第 {run}/{N_RUNS} 次训练...")

    # 随机种子设置
    seed = 42 + run
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 构建模型
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(WINDOW_SIZE, X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 输出模型路径
    model_path = os.path.join(OUTPUT_DIR, f"swing_model_run{run}.h5")
    checkpoint = ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    # 训练
    history = model.fit(
        X, y_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )

    best_acc = max(history.history["val_accuracy"])
    best_loss = min(history.history["val_loss"])

    print(f"✅ Run {run} 完成 - 验证准确率: {best_acc:.4f}, 最佳模型保存在: {model_path}")

    results.append({
        "Run": run,
        "Val Accuracy": best_acc,
        "Val Loss": best_loss,
        "Model Path": model_path
    })

# ==== 保存统计结果 ====
df_results = pd.DataFrame(results)
df_results.sort_values(by="Val Accuracy", ascending=False, inplace=True)
df_results.to_csv("results/best_of_50_summary.csv", index=False)

print("\n📊 所有训练完成，Top5 模型如下：")
print(df_results.head())
print("✅ 完整记录保存在：results/best_of_50_summary.csv")
