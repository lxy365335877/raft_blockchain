import os
import torch
import random
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import time



# 下载并加载MNIST数据集
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root="./mnt/e/datasets", train=True, transform=transform, download=True)

# 数据预处理
num_nodes = 10
node_datasets = []

# 计算每个节点应分配的样本数量
samples_per_node = [5000,3000,3500,4000,1000,1200,1500,1800,2000,2500]
total_samples = len(train_dataset)

# samples_per_node = [random.randint(1000, 5000) for _ in range(num_nodes)]
total_assigned_samples = sum(samples_per_node)


# 划分数据并保存到列表中
all_indices = list(range(total_samples))
start_idx = 0
for i, num_samples in enumerate(samples_per_node):
    end_idx = start_idx + num_samples
    node_indices = all_indices[start_idx:end_idx]

    node_dataset = torch.utils.data.Subset(train_dataset, node_indices)
    node_datasets.append(node_dataset)

    start_idx = end_idx

# 创建数据加载器
batch_size = 64
node_loaders = [DataLoader(node_dataset, batch_size=batch_size, shuffle=True,drop_last=True) for node_dataset in node_datasets]

# 打印每个节点的样本数量
for i, node_loader in enumerate(node_loaders):
    print(f"Node {i}: {len(node_loader.dataset)} samples")


# 定义简单的CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 定义CNN网络结构
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型和优化器
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002)

# 训练模型
num_epochs = 50 # 假设进行5轮训练
for node_idx, node_loader in enumerate(node_loaders):
    model = CNNModel()  # 每个节点重新创建一个新的模型，确保模型的参数在每个节点上是独立的

    optimizer = optim.SGD(model.parameters(), lr=0.002)  # 每个节点都创建一个新的优化器，确保优化器状态在每个节点上是独立的
    criterion = nn.CrossEntropyLoss()  # 每个节点都重新创建一个新的损失函数，确保损失函数在每个节点上是独立的

    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        for batch_idx, (inputs, targets) in enumerate(node_loader):
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播，得到模型预测结果
            loss = criterion(outputs, targets)  # 计算损失函数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数，优化模型

            if batch_idx % 10 == 0:
                print(f"Node {node_idx}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    time.sleep(1)  # 等待1秒，模拟节点间的切换

# 保存训练好的模型参数
torch.save(model.state_dict(), "trained_model.pth")