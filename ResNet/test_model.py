import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import model as res
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir('./ResNet')

batch_size = 32

dataset = torchvision.datasets.Caltech101(root='./../Datasets', download=False, transform=utils.transform_train)
test_dataset = torch.utils.data.Subset(dataset, utils.test_indices)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)
classes = dataset.categories

model = res.ResNet18(101)
model = model.to(device)
model.load_state_dict(torch.load('./checkpoints/resnet18/ResNet18_epoch_49.pth'))

model_34 = res.ResNet34(101)
model_34 = model_34.to(device)
model_34.load_state_dict(torch.load('./checkpoints/resnet34/ResNet34_epoch_49.pth'))

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
        outputs = model_34(images)
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