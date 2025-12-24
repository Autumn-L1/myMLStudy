import torch
import torch.nn as nn
import numpy as np

import loss_functions as lf
class GANconfig:
    '''
    GAN配置
    noise_size: 噪声的维度
    image_size: 生成的图片的尺寸
    loss_fn: 损失函数
    device: 设备

    lr: 学习率
    lr_D: D的优化器的学习率, 默认为lr
    lr_G: G的优化器的学习率, 默认为lr

    noist_to_image: 图片输入是否添加噪声（未实现）

    label_smooth: 标签平滑（未实现）

    '''
    def __init__(self,
                 noise_size = 100,
                 image_size = 28,
                 loss_fn = None,
                 device = None,
                 lr = 0.0002,
                 lr_D = None,
                 lr_G = None,


                 noise_to_image = False,
                 noise_type = "gaussian",
                 noise_mean = 0, 
                 noise_sigma = 1,
                
                 label_smooth = False,
                 ):
        self.noise_size = noise_size
        self.image_size = image_size
        self.loss_fn = loss_fn(device)
        self.device = device

        if lr_D is None:
            self.lr_D = lr
        if lr_G is None:
            self.lr_G = lr
        
        self.noise_to_image = noise_to_image
        self.noise_type = noise_type
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma

        self.label_smooth = label_smooth

        if self.label_smooth:
            self.loss_fn = lf.Label_smooth_GANLoss(device)
        else:
            if loss_fn == None:
                self.loss_fn = lf.Original_GANLoss(device)
            else: 
                self.loss_fn = loss_fn

class Discriminater(nn.Module):
    def __init__(self, input_size):
        super(Discriminater, self).__init__()
        if isinstance(input_size, int):
            self.input_size = (input_size,input_size)
        elif isinstance(input_size, tuple):
            self.input_size = input_size
        else:
            raise ValueError("input_size must be int or tuple[int,int]")
        linear_in_size = 1
        for i in range(len(self.input_size)):
            linear_in_size *= self.input_size[i]
        self.fc1 = nn.Linear(linear_in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            raise ValueError("output_size must be int or tuple[int,int]")
        linear_out_size = 1
        for i in range(len(self.output_size)):
            linear_out_size *= self.output_size[i]
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, linear_out_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        x = x.view(-1, *self.output_size)
        return x
    

class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()

        self.config = config

        self.noise_size = config.noise_size
        self.generator = Generator(config.noise_size, config.image_size)
        self.discriminator = Discriminater(config.image_size)
        self.loss_fn = config.loss_fn
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr_D)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=config.lr_G)
        
        self.info = {
            "last_D_loss": 0,
            "last_G_loss": 0,
            "acc_D": 0,
            "acc_real": 0,
            "acc_fake": 0,
        }
        self.device = config.device

    def to(self, device):
        super().to(device)
        self.device = device

    def generate(self, input):
        return self.generator(input)
    
    def discriminate(self, input):
        input = self.image_process(input)
        return self.discriminator(input)
    
    def stepD(self, real):
        # 1. Update D network
        self.optimizerD.zero_grad()

        noise = torch.randn(real.size(0), self.noise_size).to(real.device)
        fake = self.generate(noise)
        
        predictions_real = self.discriminate(real)
        predictions_fake = self.discriminate(fake)

        loss_D = self.loss_fn.discriminator_loss(predictions_real, predictions_fake)
        loss_D.backward()
        self.optimizerD.step()
        self.info["last_D_loss"] = loss_D.item()
        
        acc_D = (sum(predictions_real > 0.5).item() + sum(predictions_fake < 0.5).item()) / (2 * real.size(0))
        acc_real = sum(predictions_real > 0.5).item() / real.size(0)
        acc_fake = sum(predictions_fake < 0.5).item() / real.size(0)
        self.info["acc_D"] = acc_D
        self.info["acc_real"] = acc_real
        self.info["acc_fake"] = acc_fake

    def stepG(self, batch_size):
        # 2. Update G network
        self.optimizerG.zero_grad()

        noise = torch.randn(batch_size, self.noise_size).to(self.device)
        fake = self.generate(noise)

        predictions = self.discriminate(fake)

        loss_G = self.loss_fn.generator_loss(predictions)
        loss_G.backward()
        self.optimizerG.step()
        self.info["last_G_loss"] = loss_G.item()


    def image_process(self, image):
        if self.config.noise_to_image: 
            image = self.add_noise(image)
        return image
    def add_noise(self, image):
        if self.config.noise_type == "gaussian":
            image = image + torch.randn_like(image) * self.config.noise_sigma + self.config.noise_mean
        else:
            raise ValueError("unknown noise_type")
        return image
    

