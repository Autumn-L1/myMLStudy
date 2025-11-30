import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import utils
import train_model as m

os.chdir('D:/User/Documents/ImportantData/MyStudy/myMLStudy/AlexNet')

model = m.AlexNet()
model.load_state_dict(torch.load('AlexNet.pth'))

