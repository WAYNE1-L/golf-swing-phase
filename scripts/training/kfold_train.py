import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflow as tf

# ==== 参数 ====
CSV_PATH = "golf_swing_labeled.csv"
WINDOW_SIZE = 30
NUM_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 16

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)
df = df[df['phase'] != 'Impact']

keypoints = list(range(33))
columns_to_use = [f'x_{i}' for i in keypoints] + [f'y_{i}' for i in keypoints]
features = df[columns_to_use]
labels = df['phase']

# ==== 标签编码 ====
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

# ==== 滑动窗口处理 ====
def create_sequences(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features.iloc[i:i+window_size].values)
        y.append(labels[i + window_size - 1])
    return np.array(X), np.array(y)

X, y = create_sequences(features, labels_encoded, WINDOW_SIZE)
y_cat = to_categorical(y, num_classes=num_classes)

print(f"✅ 输入序列数: {len(X)}, 特征维度: {X.shape[1:]}")

# ==== 设置 KFold ====
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
val_accuracies = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"\n📦 Fold {fold + 1}/{NUM_FOLDS}")

    # 划分数据
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_cat[train_index], y_cat[val_index]

    # 设置随机种子保证一致性
    tf.keras.utils.set_random_seed(42 + fold)

    # 构建模型
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(WINDOW_SIZE, X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )

    best_val_acc = max(history.history["val_accuracy"])
    val_accuracies.append(best_val_acc)
    print(f"✅ Fold {fold + 1} 验证准确率: {best_val_acc:.4f}")

# ==== 输出总结果 ====
mean_acc = np.mean(val_accuracies)
std_acc = np.std(val_accuracies)
print("\n🎯 K折交叉验证完成")
print(f"📊 平均验证准确率: {mean_acc:.4f}")
print(f"📉 标准差: {std_acc:.4f}")
