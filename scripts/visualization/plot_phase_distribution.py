import pandas as pd
import matplotlib.pyplot as plt

# 读取已标注 CSV
df = pd.read_csv("data/labeled_csv/golf_swing_labeled.csv")

# 统计阶段分布
counts = df["phase"].value_counts()

# 绘图
plt.figure(figsize=(8, 5))
counts.plot(kind="bar", color="skyblue")
plt.title("每个阶段的样本数量")
plt.xlabel("阶段")
plt.ylabel("数量")
plt.tight_layout()
plt.savefig("results/phase_distribution.png")
print("✅ 已生成阶段分布柱状图：results/phase_distribution.png")
