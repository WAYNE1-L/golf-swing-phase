import pandas as pd
import numpy as np
import os
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder

# ==== 参数 ====
CSV_PATH = "golf_swing_labeled.csv"
OUTPUT_MODEL_DIR = "model/multi_runs"
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
N_RUNS = 10
EPOCHS = 30
WINDOW_SIZE = 30

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)
df = df[df['phase'] != 'Impact']  # 可选：排除 impact 阶段

# ==== 构造特征和标签 ====
keypoints = list(range(33))
columns_to_use = [f'x_{i}' for i in keypoints] + [f'y_{i}' for i in keypoints]
features = df[columns_to_use]
labels = df['phase']

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

# ==== 重复训练 N 次 ====
for run in range(1, N_RUNS + 1):
    print(f"\n🚀 正在进行第 {run} 次训练...")

    # 设置随机种子
    seed = 42 + run
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(WINDOW_SIZE, X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_path = os.path.join(OUTPUT_MODEL_DIR, f"swing_model_run{run}.h5")
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    history = model.fit(
        X, y_cat,
        batch_size=16,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[checkpoint, early_stop],
        verbose=0  # 可设为 1 查看每轮日志
    )

    best_val_acc = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    print(f"✅ Run {run} 完成 - 最佳验证准确率: {best_val_acc:.4f}, 最低验证损失: {best_val_loss:.4f}")

    results.append({
        "Run": run,
        "Best Val Accuracy": best_val_acc,
        "Best Val Loss": best_val_loss,
        "Model Path": model_path
    })

# ==== 保存结果表格 ====
df_results = pd.DataFrame(results)
df_results.to_csv("results/train_summary.csv", index=False)

print("\n📊 所有训练完成！结果已保存至 results/train_summary.csv")
print(df_results)
