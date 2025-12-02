import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # 第二层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            # 第三层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            # 第四层
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            # 第五层
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            # 到全连接层的最大值池化
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            # 使用Dropout技术
            # nn的dropout训练时会将启用的输出自动翻倍，因此不用像原论文一样，最后预测时将输出乘以0.5
            nn.Dropout(p=dropout),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    # 在测试时，网络通过提取五个224×224的补丁（四个角补丁和中心补丁）及其水平反射（总共十个补丁）进行预测，并对网络softmax层在十个补丁上的预测进行平均。
    # 这个操作可以通过transforms.TenCrop直接实现
    def test_pred(self, x: torch.Tensor) -> torch.Tensor:
        # 定义测试时的数据增强变换
        test_transform = transforms.Compose([
            transforms.TenCrop(224)
        ])

        # 应用变换
        crops = test_transform(x)

        # 对每个补丁进行预测
        predictions = []
        with torch.no_grad():
            for i in range(len(crops)):
                crop = crops[i]
                pred = self.forward(crop)
                pred = torch.softmax(pred, dim=1)
                predictions.append(pred)

        # 对所有预测结果取平均
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('./AlexNet')

# 超参数
#原文使用随机梯度下降训练模型，批量大小为128个样本，动量为0.9，权重衰减为0.0005。
#我使用Caltech101数据集，数据集更小，因此批量大小也改小了
num_epochs = 50
batch_size = 30
momentum = 0.9
weight_decay = 0.0005
#原文对所有层使用相同的学习率，并在整个训练过程中手动调整。原文遵循的启发式方法是，当验证错误率不再随着当前学习率的提高而提高时，将学习率除以10。学习率初始化为0.01，并在终止前降低三倍。
learning_rate = 0.005

#TensorBoard
writer = SummaryWriter("runs/alexnet1")


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

# 创建模型
model = AlexNet(num_classes=len(classes)).to(device)
if not os.path.exists('AlexNet.pth'):
    # 损失函数和优化器
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
            # optimizer.zero_grad()会清零参数的梯度，但不会影响优化器内部维护的动量变量
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
    torch.save(model.state_dict(), 'AlexNet.pth')
else:
    model.load_state_dict(torch.load('AlexNet.pth'))


num_classes = len(classes)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    
    # Top-5 metrics
    top1_correct = 0
    top5_correct = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model.test_pred(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # Calculate top-1 and top-5 accuracy
        _, top5_pred = torch.topk(outputs, 5, dim=1)
        top1_correct += (predicted == labels).sum().item()
        top5_correct += torch.sum(top5_pred == labels.unsqueeze(1)).item()
        
        for i in range(len(images)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    # Overall accuracies
    accuracy = 100.0 * n_correct / n_samples
    top1_accuracy = 100.0 * top1_correct / n_samples
    top5_accuracy = 100.0 * top5_correct / n_samples
    
    # Error rates
    top1_error_rate = 100.0 - top1_accuracy
    top5_error_rate = 100.0 - top5_accuracy
    
    print(f'Accuracy of the network: {accuracy} %')
    print(f'Top-1 Accuracy: {top1_accuracy:.2f}%, Top-1 Error Rate: {top1_error_rate:.2f}%')
    print(f'Top-5 Accuracy: {top5_accuracy:.2f}%, Top-5 Error Rate: {top5_error_rate:.2f}%')
    
    for i in range(num_classes):
        accuracy = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {accuracy} %')