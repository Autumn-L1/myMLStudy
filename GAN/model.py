import torch
import torch.nn as nn
import numpy as np

import loss_functions as lf


class Discriminater(nn.Module):
    def __init__(self, input_size):
        super(Discriminater, self).__init__()
        if isinstance(input_size, int):
            self.input_size = (input_size,input_size)
        elif isinstance(input_size, tuple[int,int]):
            self.input_size = input_size
        else:
            raise ValueError("input_size must be int or tuple[int,int]")
        self.fc1 = nn.Linear(self.input_size[0]*self.input_size[1], 512)
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
        elif isinstance(output_size, tuple[int,int]):
            self.output_size = output_size
        else:
            raise ValueError("output_size must be int or tuple[int,int]")
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, self.output_size[0]*self.output_size[1])
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
                 input_size, 
                 output_size,
                 lr = 0.0002,
                 d_training_round = 100,
                 loss_fn = lf.OriginalGANLoss()):
        super(GAN, self).__init__()
        self.input_size = input_size
        self.generator = Generator(input_size, output_size)
        self.discriminator = Discriminater(output_size)
        self.loss_fn = loss_fn
        self.optimizerD = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizerG = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self.info = {
            "last_D_loss" : 0,
            "last_G_loss" : 0,
        }

    def generate(self, input):
        return self.generator(input)
    def discriminate(self, input):
        return self.discriminator(input)
    
    def stepD(self, real):
        # 1. Update D network
        self.optimizerD.zero_grad()

        noise = torch.randn(real.size(0), self.input_size)
        fake = self.generate(noise)
        # real标签1， fake标签0
        lables = torch.cat([torch.ones(real.size(0)), torch.zeros(real.size(0))], dim=0)
        predictions = self.discriminate(torch.cat([real, fake], dim=0))

        loss_D = self.loss_fn.discriminator_loss(predictions, lables)
        loss_D.backward()
        self.optimizerD.step()


    def stepG(self, real):
        # 2. Update G network
        self.optimizerG.zero_grad()

        noise = torch.randn(real.size(0), self.input_size)
        fake = self.generate(noise)

        # real标签1， fake标签0
        lables = torch.cat([torch.ones(real.size(0)), torch.zeros(real.size(0))], dim=0)
        predictions = self.discriminate(torch.cat([real, fake], dim=0))

        loss_G = self.loss_fn.generator_loss(predictions,lables)
        loss_G.backward()
        self.optimizerG.step()

    def info(self):
        return self.info
        # 3. Return other info

