import torch
import torch.nn as nn
import numpy as np

import loss_functions as lf


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
    def __init__(self, 
                 noise_size, 
                 output_size,
                 loss_fn,
                 lr = 0.0002,
                 d_training_round = 100,
                 device = None
                 ):
        super(GAN, self).__init__()

        self.input_size = noise_size
        self.generator = Generator(noise_size, output_size)
        self.discriminator = Discriminater(output_size)
        self.loss_fn = loss_fn
        self.optimizerD = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizerG = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.info = {
            "last_D_loss" : 0,
            "last_G_loss" : 0,
            "acc_D": 0,
            "acc_real": 0,
            "acc_fake": 0,
        }
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def generate(self, input):
        return self.generator(input)
    def discriminate(self, input):
        return self.discriminator(input)
    
    def stepD(self, real):
        # 1. Update D network
        self.optimizerD.zero_grad()

        noise = torch.randn(real.size(0), self.input_size).to(real.device)
        fake = self.generate(noise)
        # real标签1， fake标签0
        #lables = torch.cat([torch.ones(real.size(0)), torch.zeros(real.size(0))], dim=0)
        predictions_real = self.discriminate(real)
        predictions_fake = self.discriminate(fake)

        loss_D = self.loss_fn.discriminator_loss(predictions_real, predictions_fake)
        loss_D.backward()
        self.optimizerD.step()
        self.info["last_D_loss"] = loss_D.item()
        acc_D = (sum(predictions_real > 0.5).item()+ sum(predictions_fake < 0.5).item())/ (2*real.size(0))
        acc_real = sum(predictions_real > 0.5).item() / real.size(0)
        acc_fake = sum(predictions_fake < 0.5).item() / real.size(0)
        self.info["acc_D"] = acc_D
        self.info["acc_real"] = acc_real
        self.info["acc_fake"] = acc_fake


    def stepG(self, batch_size):
        # 2. Update G network
        self.optimizerG.zero_grad()

        noise = torch.randn(batch_size, self.input_size).to(self.device)
        fake = self.generate(noise)

        predictions = self.discriminate(fake)

        loss_G = self.loss_fn.generator_loss(predictions)
        loss_G.backward()
        self.optimizerG.step()
        self.info["last_G_loss"] = loss_G.item()

    def info(self):
        return self.info
        # 3. Return other info

