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
writer = SummaryWriter("runs/resnet34")


# Caltech101数据集没有train参数，需要手动分割训练集和测试集
print(os.getcwd())
dataset = torchvision.datasets.Caltech101(root='./../Datasets', download=False, transform=utils.transform_train)


# Create subset datasets
train_dataset = torch.utils.data.Subset(dataset, utils.train_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

# Caltech101数据集的Subset会保留classes属性，可以直接获取
classes = dataset.categories

model = res.ResNet34(101)
model = model.to(device)

#选择一个模型保存点
model_path = None
# model_path = './checkpoints/resnet34/ResNet34_epoch_24.pth'

if model_path:
    model.load_state_dict(torch.load(model_path))
    start_epoch = int(model_path.split('_')[-1].split('.')[0])+1
else:
    start_epoch = 0


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, weight_decay=weight_decay, lr=learning_rate)

running_loss = 0.0
running_acc = 0

n_total_steps = len(train_loader)
for epoch in range(start_epoch,num_epochs):
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
            print(f'Epoch{epoch} [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/20, epoch*n_total_steps+i)
            writer.add_scalar('training accuracy', 100*running_acc/(20*batch_size), epoch*n_total_steps+i)
            running_loss = 0.0
            running_acc =0.0
    # 保存点
    torch.save(model.state_dict(), f'./checkpoints/resnet34/ResNet34_epoch_{epoch}.pth')
    if (epoch+1) % 5 == 0:
        for i in range(4):
            if os.path.exists(f'./checkpoints/resnet34/ResNet34_epoch_{epoch-1-i}.pth'):
                os.remove(f'./checkpoints/resnet34/ResNet34_epoch_{epoch-1-i}.pth')
    
print('Finished Training')
