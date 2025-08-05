import csv
import numpy as np
import matplotlib.pyplot as plt

# 修改成你的视频导出的文件名
your_file = 'my_golf.csv'         # ← 你的数据
pro_file = 'full_keypoints.csv'   # ← 麦克罗伊数据（职业选手）

# 读取关键点数据（去掉 Frame 列）
def load_keypoints(file):
    data = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            keypoints = list(map(float, row[1:]))  # 跳过 Frame
            data.append(keypoints)
    return np.array(data)

# 加载数据
your_data = load_keypoints(your_file)
pro_data = load_keypoints(pro_file)

# 对齐帧数（取最短）
min_len = min(len(your_data), len(pro_data))
your_data = your_data[:min_len]
pro_data = pro_data[:min_len]

# 每帧的欧几里得平均距离（越小越相似）
frame_diffs = np.linalg.norm(your_data - pro_data, axis=1)
avg_diff = np.mean(frame_diffs)

# 相似度评分（越高越好，简单线性映射）
max_possible = 300  # 你可以调整这个值
similarity_score = max(0, 100 - (avg_diff / max_possible * 100))

# 写评分文件
with open('similarity_score.txt', 'w') as f:
    f.write(f"Average Difference per Frame: {avg_diff:.2f} px\n")
    f.write(f"Similarity Score (0-100): {similarity_score:.2f}\n")

# 绘图
plt.figure(figsize=(10, 4))
plt.plot(frame_diffs, label='Frame-wise Difference', color='orange')
plt.xlabel('Frame')
plt.ylabel('Distance')
plt.title('Per-frame Keypoint Distance (Euclidean)')
plt.grid(True)
plt.legend()
plt.savefig('frame_diff_plot.png')
plt.close()

print("✅ Similarity analysis complete.")
print("📊 Saved: similarity_score.txt")
print("🖼️ Saved: frame_diff_plot.png")
