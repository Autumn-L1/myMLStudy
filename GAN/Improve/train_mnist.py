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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "..", "..", "Datasets")
GENERATED_IMAGES_DIR = os.path.join(BASE_DIR, "generated_images")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

os.chdir(BASE_DIR)
target_class = 2
generated_class_dir = os.path.join(GENERATED_IMAGES_DIR, f"mnist_class_{target_class}_2-1_noise+TTUR")
writer_dir = os.path.join(RUNS_DIR, f"mnist_class_{target_class}_2-1_noise+TTUR")

batch_size = 64
num_iteration = 50000
config = model.GANconfig(
    noise_size = 100,
    image_size = (1, 28, 28),
    #loss_fn = lf.Original_GANLoss,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr = 0.0002,
    lr_D = 0.0004,
    lr_G = 0.0001,

    noise_to_image= True,
    label_smooth= False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writerD = SummaryWriter("runs/mnist_class_%d_D" % target_class)
# writerG = SummaryWriter("runs/mnist_class_%d_G" % target_class)
writer = SummaryWriter(writer_dir)


num_D_training_round = 2
num_G_training_round = 1
# 数据集
dataset = torchvision.datasets.MNIST(root=DATASETS_DIR, train=True, transform=transforms.ToTensor(), download=False)
indices = [i for i in range(len(dataset)) if dataset.targets[i] == target_class]
subset = torch.utils.data.Subset(dataset, indices)

train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

# 模型
model = model.GAN(config)
model.to(device)

#保存图片的函数

if not os.path.exists(GENERATED_IMAGES_DIR):
    os.mkdir(GENERATED_IMAGES_DIR)
if not os.path.exists(generated_class_dir):
    os.mkdir(generated_class_dir)

def save_generated_images(gan_model, iteration, num_images=10):
    gan_model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_images, gan_model.config.noise_size).to(gan_model.config.device)
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
        file_path = os.path.join(generated_class_dir, f"generated_images_iter_{iteration}_digit_{target_class}.png")
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    gan_model.train()  # Set model back to training mode

# 训练
# for i in range(200):
#     real,_ = next(iter(train_loader))
#     model.stepD(real.to(device))
step_count = 0
for i in range(num_iteration):
    for k in range(num_D_training_round):
        real,_ = next(iter(train_loader))
        model.stepD(real.to(device))
        if k % 10 == 0:
            print("iteration: ", i, "; D_round_",k ,"; D_loss: %.4f" % model.info["last_D_loss"])
            writer.add_scalar("D_loss", model.info["last_D_loss"], step_count)
            writer.add_scalar("D_acc", model.info["acc_D"], step_count)
            writer.add_scalar("D_acc_real", model.info["acc_real"], step_count)
            writer.add_scalar("D_acc_fake", model.info["acc_fake"], step_count)
        step_count += 1
    for k in range(num_G_training_round):
        model.stepG(batch_size)
        print("iteration: ", i, "; G_round_",k ,"; G_loss: %.4f" % model.info["last_G_loss"])
        writer.add_scalar("G_loss", model.info["last_G_loss"], step_count)
    if i % 500 == 0:
        save_generated_images(model, i, 10)
    step_count += 1
    # delta = -0.1*((model.info["acc_D"]-0.75)/0.25)*num_D_training_round
    # if num_D_training_round > 10 and delta <= 0:
    #     num_D_training_round = round(num_D_training_round + delta)
    # elif num_D_training_round < 600 and delta >= 0:
    #     num_D_training_round = round(num_D_training_round + delta)
    # writerD.add_scalar("num_D_training_round", num_D_training_round, step_count)