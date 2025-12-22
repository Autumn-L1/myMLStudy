import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os

import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

import model
import loss_functions as lf

# 配置
os.chdir("./GAN")
target_class = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writerD = SummaryWriter("runs/mnist_class_%d_D" % target_class)
writerG = SummaryWriter("runs/mnist_class_%d_G" % target_class)
lr = 0.0002
batch_size = 64
num_iteration = 500

num_D_training_round = 100
# 数据集
dataset = torchvision.datasets.MNIST(root='./../Datasets', train=True, transform=transforms.ToTensor(), download=False)
indices = [i for i in range(len(dataset)) if dataset.targets[i] == target_class]
subset = torch.utils.data.Subset(dataset, indices)

train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

# 模型
model = model.GAN(noise_size = 100, 
                  output_size = (1, 28, 28), 
                  loss_fn=lf.OriginalGANLoss(device = device),
                  lr=lr)
model.to(device)

#保存图片的函数
if not os.path.exists("./generated_images"):
    os.mkdir("./generated_images")
if not os.path.exists("./generated_images/mnist_class_%d" % target_class):
    os.mkdir("./generated_images/mnist_class_%d" % target_class)
def save_generated_images(gan_model, iteration, num_images=10):
    gan_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_images, gan_model.input_size).to(gan_model.device)
        # Generate images
        fake_images = gan_model.generate(noise)
        
        # Create a grid of images
        grid = vutils.make_grid(fake_images, nrow=5, padding=2, normalize=True)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Display the grid
        plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f'Generated Images for Digit {target_class} at Iteration {iteration}', fontsize=14)
        
        # Save the figure
        plt.savefig(f'./generated_images/mnist_class_{target_class}/generated_images_iter_{iteration}_digit_{target_class}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    gan_model.train()  # Set model back to training mode

# 训练
# for i in range(200):
#     real,_ = next(iter(train_loader))
#     model.stepD(real.to(device))
for i in range(num_iteration):
    for k in range(num_D_training_round):
        real,_ = next(iter(train_loader))
        model.stepD(real.to(device))
        if k % 10 == 0:
            print("iteration: ", i, "; D_round_",k ,"; D_loss: %.4f" % model.info["last_D_loss"])
            writerD.add_scalar("D_loss", model.info["last_D_loss"], i*(num_D_training_round+1)+k)
            writerD.add_scalar("D_acc", model.info["acc_D"], i*(num_D_training_round+1)+k)
            writerD.add_scalar("D_acc_real", model.info["acc_real"], i*(num_D_training_round+1)+k)
            writerD.add_scalar("D_acc_fake", model.info["acc_fake"], i*(num_D_training_round+1)+k)
    real,_ = next(iter(train_loader))
    model.stepG(batch_size)
    print("iteration: ", i, "; G_loss: %.4f" % model.info["last_G_loss"])
    writerG.add_scalar("G_loss", model.info["last_G_loss"], i*(num_D_training_round+1)+k)
    if i % 10 == 0:
        save_generated_images(model, i, 10)
