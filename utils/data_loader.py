"""
数据加载与预处理模块
====================
功能：
  1. 下载并加载 CIFAR-10 数据集
  2. 对训练集做数据增强（data augmentation）——提升模型泛化能力
  3. 对测试集仅做归一化
  4. 创建 DataLoader 实现批量加载

教学要点：
  - 数据增强（翻转、裁剪、色彩抖动）能有效防止过拟合
  - 归一化（减均值、除标准差）让数据分布稳定，加速训练收敛
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 RGB 三通道统计值（在50,000张训练图上计算）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ImageNet RGB 三通道统计值（在120万张ImageNet训练图上计算）
# 迁移学习时必须用这个！预训练权重是在这个分布下学到的
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# 10个类别名称
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_transforms(train: bool = True) -> transforms.Compose:
    """获取数据变换流水线。训练时含数据增强，测试时仅归一化。"""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),       # 随机裁剪（平移不变性）
            transforms.RandomHorizontalFlip(),           # 随机水平翻转
            transforms.ColorJitter(brightness=0.1,       # 亮度
                                   contrast=0.1,         # 对比度
                                   saturation=0.1,       # 饱和度
                                   hue=0.05),            # 色相
            transforms.ToTensor(),                       # PIL→Tensor [0,1]
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    """创建训练集和测试集的 DataLoader。"""
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=get_transforms(train=True),
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[数据] CIFAR-10 加载完成！")
    print(f"  训练集: {len(train_set)} 张 | 测试集: {len(test_set)} 张")
    print(f"  类别: {CLASS_NAMES}")
    return train_loader, test_loader


def get_dataloaders_transfer(batch_size: int = 64, num_workers: int = 2):
    """
    为迁移学习模型创建 DataLoader。

    与普通版区别：
      1. 使用 ImageNet 归一化参数（预训练模型的"母语"）
      2. 将图像放大到 224×224（预训练 ResNet 的标准输入尺寸）
      3. 更小的 batch_size（显存限制，224×224 比 32×32 大49倍）

    为什么迁移学习需要 ImageNet 归一化？
      → ImageNet 预训练权重是在 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        的数据分布下学到的。如果换成 CIFAR-10 的归一化（mean≈[0.49,0.48,0.45]），
        像素值的分布就变了，预训练的特征提取器会"看不懂"输入。
      → 类比：一个在英语环境中学说话的人，突然让他听中文——他能听出音节，
        但理解不了语义。
    """
    train_transforms = transforms.Compose([
        transforms.Resize(256),                          # 先放大到256
        transforms.RandomCrop(224),                      # 再随机裁到224（增强）
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ★ ImageNet归一化
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(224),                          # 直接放大到224
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=train_transforms,
    )
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transforms,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[数据] CIFAR-10 (迁移学习模式, ImageNet归一化) 加载完成！")
    print(f"  训练集: {len(train_set)} 张 | 测试集: {len(test_set)} 张")
    print(f"  图像尺寸: 224×224 | Batch Size: {batch_size}")
    return train_loader, test_loader
