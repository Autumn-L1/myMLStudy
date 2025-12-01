import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import utils
import train_model as m

os.chdir('./AlexNet')

model = m.AlexNet()
model.load_state_dict(torch.load('AlexNet.pth'))

