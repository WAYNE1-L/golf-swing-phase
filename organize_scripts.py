import os
import shutil

data_dir = "data"

# 分类规则：文件夹 → 文件特征
folders = {
    "images": [".jpg", ".jpeg", ".png"],
    "raw_videos": [".mp4", ".mov", ".avi"],
    "keypoints_csv": ["keypoints.csv"],
    "labeled_csv": ["labeled.csv", "golf_swing_labeled.csv"],
    "npz": [".npz"]
}

# 创建目标文件夹（如不存在）
for folder in folders:
    os.makedirs(os.path.join(data_dir, folder), exist_ok=True)

# 遍历 data 目录下的文件
for fname in os.listdir(data_dir):
    fpath = os.path.join(data_dir, fname)
    if os.path.isdir(fpath):
        continue  # 已归类的目录跳过

    moved_flag = False
    for folder, rules in folders.items():
        for rule in rules:
            # 匹配文件扩展名
            if rule.startswith(".") and fname.lower().endswith(rule):
                shutil.move(fpath, os.path.join(data_dir, folder, fname))
                print(f"✅ Moved {fname} → {folder}/")
                moved_flag = True
                break
            # 匹配关键词（如包含 "labeled.csv"）
            elif rule in fname:
                shutil.move(fpath, os.path.join(data_dir, folder, fname))
                print(f"✅ Moved {fname} → {folder}/")
                moved_flag = True
                break
        if moved_flag:
            break

# 全部完成后提示
print("🎉 文件整理完成！")
