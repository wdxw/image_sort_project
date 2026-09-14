"""
模型1：SimpleCNN —— 从零搭建的卷积神经网络
==========================================

架构设计（由浅入深，逐层理解）：

输入: 3×32×32 彩色图像
  ↓
[卷积块1] Conv2d(3→32, kernel=3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool(2)
  输出: 32×16×16    ← 理解: 32个卷积核各自扫过图像，提取边缘/纹理特征
  ↓
[卷积块2] Conv2d(32→64, kernel=3, pad=1) → BatchNorm2d(64) → ReLU → MaxPool(2)
  输出: 64×8×8      ← 理解: 通道翻倍，空间尺寸减半——逐步抽象
  ↓
[卷积块3] Conv2d(64→128, kernel=3, pad=1) → BatchNorm2d(128) → ReLU → MaxPool(2)
  输出: 128×4×4     ← 理解: 深层通道学习更高级的语义特征
  ↓
Flatten: 128×4×4 = 2048  →  摊平成1维向量
  ↓
[全连接层1] Linear(2048→256) → ReLU → Dropout(0.4)
  ↓  理解: Dropout 随机"关掉"40%神经元，迫使每个神经元都学到有用特征
[全连接层2] Linear(256→10)
  ↓
输出: 10个类别的 logits（未归一化的概率）

教学总结：
  - Conv → BN → ReLU → Pool 是 CNN 的标准"积木"
  - BatchNorm 让每层输入分布稳定 → 可以用更大的学习率 → 训练更快
  - MaxPool 减小空间尺寸 → 减少计算量 + 提供局部平移不变性
  - Dropout 是最常用的正则化手段 → 防止过拟合
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    轻量级 CNN，适合作为 baseline 模型。
    参数量约 60 万，在 RTX 4060 上训练一个 epoch 只需几秒。
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        """
        参数:
            num_classes: 分类类别数（CIFAR-10 为 10）
            dropout: Dropout 比例（0.4 表示丢弃 40% 的神经元）
        """
        super(SimpleCNN, self).__init__()

        # ---- 特征提取部分（卷积层）----
        # 这里使用了 Sequential 来把多个层"打包"在一起，代码更简洁

        self.conv_block1 = nn.Sequential(
            # 第1卷积层：输入3通道(RGB) → 输出32通道
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),    # 批归一化：稳定训练
            nn.ReLU(inplace=True), # 激活函数：引入非线性
            nn.MaxPool2d(2),       # 2×2 池化：32×32 → 16×16
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 16×16 → 8×8
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),       # 8×8 → 4×4
        )

        # ---- 分类部分（全连接层）----
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 128×4×4 = 2048 → 1维
            nn.Linear(128 * 4 * 4, 256),      # 2048 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),               # 随机丢弃，防过拟合
            nn.Linear(256, num_classes),       # 256 → 10（10个类别的分数）
        )

        # 初始化权重（有助于训练更稳定）
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化 —— 针对 ReLU 激活函数优化的初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。PyTorch 会自动构建计算图，用于反向传播求梯度。

        参数:
            x: 输入图像张量，形状为 (batch_size, 3, 32, 32)

        返回:
            形状为 (batch_size, 10) 的 logits
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


def create_simple_cnn(dropout: float = 0.4) -> SimpleCNN:
    """工厂函数：创建一个配置好的 SimpleCNN"""
    model = SimpleCNN(num_classes=10, dropout=dropout)
    # 计算参数量（了解你的模型有多大）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SimpleCNN] 总参数: {total_params:,} | 可训练参数: {trainable_params:,}")
    return model
