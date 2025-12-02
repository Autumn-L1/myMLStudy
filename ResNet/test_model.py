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

dataset = torchvision.datasets.Caltech101(root='./../Datasets', download=False, transform=utils.transform_train)
test_dataset = torch.utils.data.Subset(dataset, utils.test_indices)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

model = res.ResNet18(101)
model = model.to(device)
model.load_state_dict(torch.load('./'))