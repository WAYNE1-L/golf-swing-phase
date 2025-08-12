import pandas as pd
import os

input_dir = "labeled_csv"
output_file = "golf_swing_labeled.csv"

all_dfs = []

for file in os.listdir(input_dir):
    if file.endswith("_labeled.csv"):
        video_id = file.replace("_labeled.csv", "")
        df = pd.read_csv(os.path.join(input_dir, file))
        df["video_id"] = video_id
        all_dfs.append(df)

merged_df = pd.concat(all_dfs, ignore_index=True)
merged_df.to_csv(output_file, index=False)

print(f"✅ 合并完成，共 {len(merged_df)} 行，保存为：{output_file}")
print(f"📌 包含视频：{[f.replace('_labeled.csv','') for f in os.listdir(input_dir) if f.endswith('_labeled.csv')]}")
