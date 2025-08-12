import json
import matplotlib.pyplot as plt

# 读取训练记录
with open("model/train_history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

# 绘图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="训练准确率")
plt.plot(history["val_accuracy"], label="验证准确率")
plt.title("准确率")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="训练损失")
plt.plot(history["val_loss"], label="验证损失")
plt.title("损失")
plt.legend()

plt.tight_layout()
plt.savefig("results/train_curves.png")
print("✅ 已生成训练过程图像：results/train_curves.png")
