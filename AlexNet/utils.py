import torch
import torchvision.transforms as transforms

# ImageNet由可变分辨率的图像组成，AlexNet需要恒定的输入维度,因此需要将图像下采样到256×256的固定分辨率。
# 下采样函数
def imagenet_downsampling(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    return transform(image)

# AlexNet原论文使用了两种数据增强策略：
# 在256×256图像中随机提取的224×224补丁及其水平翻转上训练网络
# 在测试时，网络通过提取五个224×224的补丁（四个角补丁和中心补丁）及其水平反射（总共十个补丁）进行预测，并对网络softmax层在十个补丁上的预测进行平均
def extract_patch(image):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224)
    ])
    return transform(image)
# 对训练图像中RGB通道的强度进行调整
# 在ImageNet训练集的所有RGB像素值上执行主成分分析（PCA）。对于每张训练图像，我们添加由所找到的主成分乘以相应特征值和一个均值为零、标准差为0.1的高斯分布随机变量所得的倍数。
# 需要运行PCA以获得参数,这里暂且用一套网上的参数
# 添加由所找到的主成分乘以相应特征值和一个均值为零、标准差为0.1的高斯分布随机变量所得的倍数。添加由所找到的主成分乘以相应特征值和一个均值为零、标准差为0.1的高斯分布随机变量所得的倍数。
def adjust_RGB(image):
    eigvals = torch.Tensor([0.2175, 0.0188, 0.0045])
    eigvecs = torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203]
    ])
    alphastd = 0.1
    alpha = image.new_empty(3).normal_(0, alphastd)
    # 论文公式: 噪声=主成分*相应特征值*高斯分布随机变量,
    rgb = (eigvecs.to(image.device) * (alpha * eigvals.to(image.device))).sum(dim=1)
    # 噪声添加到每个像素
    image = image + rgb.view(3, 1, 1)
    # 将图像张量的像素值限制在0到1的范围内
    image = torch.clamp(image, 0, 1)
    return image

def preprocess(image):
    # 先转换为张量
    tensor = transforms.ToTensor()(image)
    # 下采样
    downsampled = imagenet_downsampling(tensor)
    # 提取补丁
    patch = extract_patch(downsampled)
    # RGB调整
    adjusted = adjust_RGB(patch)
    return adjusted


