import pandas as pd

df = pd.read_csv("results/predicted_phases.csv")

# 仅保留真实为 Takeaway 的样本
takeaway_df = df[df["real_phase"] == "Takeaway"]

# 判断是否被误判
wrong = takeaway_df[takeaway_df["real_phase"] != takeaway_df["pred_phase"]]

# 输出诊断结果
print(f"✅ Takeaway 样本总数：{len(takeaway_df)}")
print(f"❌ 被误判数：{len(wrong)}")
print("🔍 被误判的前几项：")
print(wrong.head())

# 保存为 CSV
wrong.to_csv("results/takeaway_errors.csv", index=False)
print("📄 已保存至 results/takeaway_errors.csv")
