import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 读取 CSV：应包含 real_phase, pred_phase 两列
df = pd.read_csv("results/predicted_phases.csv")

y_true = df["real_phase"]
y_pred = df["pred_phase"]

labels = sorted(y_true.unique())

cm = confusion_matrix(y_true, y_pred, labels=labels)

# 可视化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
print("✅ 已生成混淆矩阵图像：results/confusion_matrix.png")
