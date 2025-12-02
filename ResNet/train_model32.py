import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import model as res
import utils

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('./ResNet')

# 超参数
#原文使用
#Caltech101数据集数据集更小，因此批量大小也改小了
num_epochs = 50
batch_size = 32
momentum = 0.9
weight_decay = 0.0005
#原文对所有层使用相同的学习率，并在整个训练过程中手动调整。原文遵循的启发式方法是，当验证错误率不再随着当前学习率的提高而提高时，将学习率除以10。学习率初始化为0.01，并在终止前降低三倍。
learning_rate = 0.005

#TensorBoard
writer = SummaryWriter("runs/resnet18")


# Caltech101数据集没有train参数，需要手动分割训练集和测试集
print(os.getcwd())
dataset = torchvision.datasets.Caltech101(root='./../Datasets', download=False, transform=utils.transform_train)

# Per-class train-test split
train_indices = []
test_indices = []
train_ratio = 0.9

# Group indices by class
class_indices = {}
for idx in range(len(dataset)):
    _, label = dataset[idx]
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)

# Split each class separately
for label, indices in class_indices.items():
    np.random.shuffle(indices)
    split_point = int(len(indices) * train_ratio)
    train_indices.extend(indices[:split_point])
    test_indices.extend(indices[split_point:])

# Create subset datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)
# Caltech101数据集的Subset会保留classes属性，可以直接获取
classes = dataset.categories

model = res.ResNet18(101)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, weight_decay=weight_decay, lr=learning_rate)

running_loss = 0.0
running_acc = 0

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (torch.max(outputs.data,1)[1] == labels).sum().item()

        # 打印结果
        if (i+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/20, epoch*n_total_steps+i)
            writer.add_scalar('training accuracy', 100*running_acc/(20*batch_size), epoch*n_total_steps+i)
            running_loss = 0.0
            running_acc =0.0

print('Finished Training')
torch.save(model.state_dict(), 'ResNet18.pth')