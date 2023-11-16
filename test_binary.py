import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# 定义训练数据和标签
train_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
train_labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 定义模型、损失函数和优化器
input_size = 2
output_size = 1
model = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 进行模型训练
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 进行模型预测
test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
with torch.no_grad():
    predicted_labels = model(test_data)
    predicted_labels = (predicted_labels > 0.5).float()

print("Predicted Labels:")
print(predicted_labels)