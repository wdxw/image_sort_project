"""
模型2 & 3：ResNet-18 —— 残差网络
================================

ResNet（Residual Network）是深度学习史上的里程碑——它在 2015 年赢得了
ImageNet 冠军，首次让超深网络（152层）的训练成为可能。

核心思想：残差连接（Residual Connection / Skip Connection）
==========================================================

标准网络:  input → [Weight Layer] → [ReLU] → [Weight Layer] → output
                  ↑_____________________________|
              梯度需要穿过两层才能传递，层越多越容易"消失"

ResNet:    input → [Weight Layer] → [ReLU] → [Weight Layer] → (+) → output
              |_______________________________________________↑
              残差连接（恒等映射）：梯度可以直接"抄近路"回传

为什么有效？
  - 梯度有"高速公路"直达浅层 → 解决了梯度消失
  - 网络学习的是"残差" F(x) = H(x) - x，而不是直接学 H(x)
  - 如果恒等映射就是最优解，网络只需让所有权重趋向于0即可

架构详情（ResNet-18）：
  - 初始卷积: 3×3 conv (stride=1, 适配32×32小图)
  - Layer1: 2个 BasicBlock，通道=64， 空间=32×32
  - Layer2: 2个 BasicBlock，通道=128，空间=16×16
  - Layer3: 2个 BasicBlock，通道=256，空间=8×8
  - Layer4: 2个 BasicBlock，通道=512，空间=4×4
  - 平均池化 + 全连接层 → 10类

两种使用方式：
  1. pretrained=False（模型2）：从头训练，权重随机初始化
  2. pretrained=True （模型3）：加载 ImageNet 预训练权重 → 迁移学习微调
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights


# ============================================================
# BasicBlock —— ResNet-18/34 的基本构建块
# ============================================================
class BasicBlock(nn.Module):
    """
    残差块的基本单元。

    结构:
        input → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+) → ReLU → output
                |______________________________________↑
                       残差连接（恒等映射）

    如果输入/输出通道数不一致（下采样时），残差连接用 1×1 卷积调整维度。
    """

    # 扩展系数：BasicBlock 输入输出通道一致，所以是1
    # （Bottleneck 是4，ResNet-50/101/152 用）
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        """
        参数:
            in_channels:  输入通道数
            out_channels: 输出通道数
            stride:       卷积步长。stride=2 时空间尺寸减半
            downsample:   下采样模块（当 in_channels != out_channels 时使用）
        """
        super(BasicBlock, self).__init__()

        # 第一个 3×3 卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个 3×3 卷积（stride=1，不改变空间尺寸）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样模块：当输入/输出维度不匹配时，用1×1卷积对齐
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # 保存输入，用作残差连接

        # 主路径：Conv → BN → ReLU → Conv → BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：如果维度变化，先对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        # 相加 + ReLU —— 这是 ResNet 的精髓！
        out += identity
        out = self.relu(out)

        return out


# ============================================================
# ResNet —— 完整网络
# ============================================================
class ResNet(nn.Module):
    """
    ResNet 网络主体。

    参数:
        block:      构建块类型（BasicBlock → ResNet-18/34）
        layers:     每个 layer 的块数量列表，如 [2,2,2,2] → ResNet-18
        num_classes:分类数
        cifar10:    是否适配 CIFAR-10 的 32×32 输入
    """

    def __init__(self, block: type, layers: list, num_classes: int = 10,
                 cifar10: bool = True):
        super(ResNet, self).__init__()

        self.in_channels = 64  # 初始通道数，随网络加深逐步翻倍
        self.cifar10 = cifar10

        # ---- 初始卷积层 ----
        if cifar10:
            # CIFAR-10 图像小（32×32），用 3×3 conv stride=1，不缩小尺寸
            # 这样保留了更多空间信息给浅层特征
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.maxpool = nn.Identity()  # 不用 maxpool（空间已经很小了）
        else:
            # ImageNet 标准配置：7×7 conv stride=2 + maxpool
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ---- 四个残差层 ----
        # 每个 layer 的第一个 block 负责下采样（空间减半，通道翻倍）
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)   # 64通道,  32×32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # 128通道, 16×16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   # 256通道, 8×8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 512通道, 4×4

        # ---- 分类头 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到 1×1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重（仅对从零训练的模型）
        self._init_weights()

    def _make_layer(self, block: type, out_channels: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        """
        构建一个残差层（包含 num_blocks 个 BasicBlock）。

        第一个 block 的 stride 可能 >1（下采样），后续 block 的 stride=1。
        """
        downsample = None

        # 如果输入/输出通道不匹配（通常是第一个block），用1×1卷积对齐
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个 block（可能带下采样）
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        # 更新 in_channels 为当前输出通道数
        self.in_channels = out_channels * block.expansion

        # 后续 block（stride=1，无下采样）
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming 初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_feature_extractor(self) -> nn.Module:
        """
        返回去掉分类头的特征提取器（迁移学习阶段1会用到）。
        用处：先冻结特征提取器，只训练新的分类头。
        """
        return nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
            self.avgpool, nn.Flatten(1)
        )


# ============================================================
# 工厂函数
# ============================================================

def create_resnet18_scratch() -> ResNet:
    """
    创建从零训练的 ResNet-18（适配 CIFAR-10）。

    这是"模型2"：和 SimpleCNN 一样从头开始学，但架构强得多。
    残差连接让它即使有18层也能有效训练。
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, cifar10=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ResNet-18 Scratch] 总参数: {total_params:,}")
    return model


def create_resnet18_transfer(num_classes: int = 10, freeze_backbone: bool = True):
    """
    创建基于迁移学习的 ResNet-18（加载 ImageNet 预训练权重）。

    这是"模型3"——实用中最强的方法。

    迁移学习的两阶段训练策略：
    ┌─────────────────────────────────────────────────────────────┐
    │ 阶段1（freeze_backbone=True）：                              │
    │   冻结 backbone（所有卷积层不更新）                            │
    │   → 只训练新换上的 FC 层（分类头）                            │
    │   → 用较大的学习率（0.01），训练 5-10 epochs                  │
    │   → 目的：让分类头快速适应新任务，不破坏预训练特征              │
    ├─────────────────────────────────────────────────────────────┤
    │ 阶段2（freeze_backbone=False）：                             │
    │   解冻整个网络，端到端微调                                     │
    │   → 用较小的学习率（0.001），训练更多 epochs                   │
    │   → 目的：让预训练特征向 CIFAR-10 数据分布微调                  │
    └─────────────────────────────────────────────────────────────┘

    为什么 ImageNet 预训练权重对 CIFAR-10 也有帮助？
      ImageNet 有1000类、120万张图片，模型已经学会了识别边缘、纹理、
      形状等通用视觉特征。CIFAR-10 虽然图像小，但这些底层特征是一样的。
      迁移学习相当于"站在巨人的肩膀上"。

    参数:
        num_classes: 分类数（CIFAR-10 为 10）
        freeze_backbone: True → 冻结 backbone（阶段1）；False → 解冻（阶段2）
    """
    # 加载预训练模型
    model = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 替换最后的全连接层：ImageNet 有1000类，CIFAR-10 只有10类
    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        # 冻结所有参数（除了刚换上的 fc 层）
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print("[ResNet-18 Transfer] Backbone 已冻结，仅训练 FC 层（阶段1）")
    else:
        print("[ResNet-18 Transfer] 全部参数可训练（阶段2：全局微调）")

    # 统计可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total:,} | 可训练: {trainable:,} "
          f"({100 * trainable / total:.1f}%)")

    return model
