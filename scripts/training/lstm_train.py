import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 加载数据
data = np.load('train/train_data.npz')['data']
labels = np.load('train/train_labels.npy')

# 标签映射
phase2id = {'Preparation': 0, 'Backswing': 1, 'Downswing': 2, 'Impact': 3, 'FollowThrough': 4}
num_classes = len(phase2id)

# 转为张量
X = torch.tensor(data, dtype=torch.float32).to(device)
y = torch.tensor([phase2id[label] for label in labels], dtype=torch.long).to(device)

# 超参数
input_size = 66      # 33 关键点 * 2
hidden_size = 128
seq_len = X.shape[1] # 默认是10帧
batch_size = 32
epochs = 20
lr = 0.001

# 模型定义
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # 取最后一层的隐藏状态
        out = self.fc(hn[-1])
        return out

model = LSTMClassifier(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练过程
losses = []
for epoch in range(epochs):
    permutation = torch.randperm(X.size(0))
    total_loss = 0
    for i in range(0, X.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = X[indices]
        batch_y = y[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (X.size(0) // batch_size)
    losses.append(avg_loss)
    print(f"📘 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# 可视化 loss 曲线
plt.plot(losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()
plt.savefig("train_loss_curve.png")
plt.show()

# 保存模型
torch.save(model.state_dict(), "trained_lstm_model.pth")
print("✅ 模型已保存为 trained_lstm_model.pth")
