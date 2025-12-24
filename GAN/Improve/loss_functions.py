#参考：https://zhuanlan.zhihu.com/p/1889981710814390031
import torch
import torch.nn as nn

# 原始GAN损失函数实现
class OriginalGANLoss:
    def __init__(self, device):
        self.device = device
        self.criterion = nn.BCELoss()
    
    def discriminator_loss(self, real_output, fake_output):
        # 真实样本的目标标签为1.0
        real_labels = torch.ones_like(real_output, device=self.device)
        # 生成样本的目标标签为0.0
        fake_labels = torch.zeros_like(fake_output, device=self.device)

        # 计算判别器对真实样本的损失
        real_loss = self.criterion(real_output, real_labels)
        # 计算判别器对生成样本的损失
        fake_loss = self.criterion(fake_output, fake_labels)

        # 总损失为两部分之和
        d_loss = real_loss + fake_loss
        return d_loss
    
    def generator_loss(self, fake_output):
    # 生成器希望判别器将生成样本判断为真实样本
        target_labels = torch.ones_like(fake_output, device=self.device)
        g_loss = self.criterion(fake_output, target_labels)
        return g_loss