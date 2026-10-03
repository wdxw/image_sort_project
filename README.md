# image_sort_project

基于深度学习的 CIFAR-10 图像分类 —— 项目说明书（README）

## 项目名称

基于深度学习的图像分类算法设计与对比研究

## 项目简介

本项目在 CIFAR-10 标准数据集上，从零搭建并对比了 3 种不同策略的深度学习
图像分类模型。通过"由浅入深、层层递进"的方式，系统地展示了卷积神经网络（CNN）、
残差网络（ResNet）以及迁移学习（Transfer Learning）的核心原理与实践方法。

3 种模型策略：

| 模型 | 策略 | 测试准确率 | 说明 |
| --- | --- | --- | --- |
| 1. SimpleCNN | 从零训练 | 84.19% | 自建轻量级 CNN 基线 |
| 2. ResNet-18 (从头) | 从零训练 | 94.31% | 残差网络 + 标准训练 |
| 3. ResNet-18 (微调) | 迁移学习 + 微调 | 95.08% ★ | ImageNet 预训练 + 微调 |

实验结论：迁移学习是中小规模数据集上性价比最高的方案——不需要从零设计架构，
不需要海量训练数据，不需要超长训练时间，却能取得最优的分类效果。

## 项目目录结构

```text
image_sort_project/
├── README.md                ← 本项目说明文档
├── main.py                  ← 主程序入口，依次训练/加载 3 个模型并对比评估
├── train.py                 ← 通用训练模块（训练循环、验证、早停、模型保存）
├── evaluate.py              ← 模型评估与对比模块（分类指标、混淆矩阵、对比图表）
├── download_data.py         ← CIFAR-10 数据集下载脚本（支持代理加速）
├── requirements.txt         ← Python 依赖包清单
│
├── models/                  ← 模型定义
│   ├── simple_cnn.py        → SimpleCNN：自建 3 层卷积网络（~62 万参数）
│   └── resnet_cifar.py      → ResNet-18：支持从零训练 + ImageNet 迁移学习
│
├── utils/                   ← 工具模块
│   ├── data_loader.py       → 数据加载、预处理、数据增强
│   └── plot.py              → 可视化（训练曲线、混淆矩阵、模型对比柱状图）
│
├── data/                    ← CIFAR-10 数据集目录（自动下载）
│   └── cifar-10-batches-py/
│
├── checkpoints/             ← 训练好的模型权重（.pth 文件）
│   ├── SimpleCNN_best.pth
│   ├── ResNet18_Scratch_best.pth
│   └── ResNet18_Transfer_best.pth
│
└── results/                 ← 所有输出结果
    ├── SimpleCNN_training_curves.png          → 训练曲线图
    ├── SimpleCNN_confusion_matrix.png         → 混淆矩阵
    ├── SimpleCNN_report.txt                  → 分类报告
    ├── ResNet18_Scratch_training_curves.png
    ├── ResNet18_Scratch_confusion_matrix.png
    ├── ResNet18_Scratch_report.txt
    ├── ResNet18_Transfer_training_curves.png
    ├── ResNet18_Transfer_confusion_matrix.png
    ├── ResNet18_Transfer_report.txt
    ├── model_comparison.png                   → 三模型指标对比柱状图
    └── model_comparison.csv                   → 三模型指标对比表格
```

## 环境要求

### 硬件：

- GPU 推荐（NVIDIA RTX 4060 8GB，全部训练约 15-20 分钟）
- CPU 也可运行（ResNet 从零训练会较慢）
- 磁盘空间约 500 MB（含数据集）

### 软件：

- Python >= 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CUDA（可选，有 GPU 时使用）

### 依赖安装：

```sh
pip install -r requirements.txt
```

## 快速开始

### 1. 安装依赖：

```sh
cd image_sort_project
pip install -r requirements.txt
```

### 2. 运行主程序：

```sh
python main.py
```

### 3. 程序会自动：

- 检测 GPU/CPU 环境
- 下载 CIFAR-10 数据集（首次运行，约 170MB）
- 训练/加载 3 个模型
- 评估并生成所有图表和报告
- 输出对比结论

### 4. 查看结果：

- 图表：results/*.png
- 分类报告：results/*_report.txt
- 对比表格：results/model_comparison.csv

## 使用方法详解

### 1. 运行控制（在 main.py 开头修改）：

```python
SKIP_TRAIN_SIMPLE = True       # 跳过 SimpleCNN 训练，直接加载已有权重
SKIP_TRAIN_RESNET_SCRATCH = True  # 跳过 ResNet 从零训练
SKIP_TRAIN_RESNET_TRANSFER = False # 是否重新训练迁移学习模型
```

### 2. 单独训练某个模型：

```python
from train import train_model
from models.simple_cnn import create_simple_cnn
history = train_model(model, train_loader, test_loader, "MyModel", num_epochs=30)
```

### 3. 单独评估已训练的模型：

```python
from evaluate import evaluate_model
result = evaluate_model(model, test_loader, device, "ModelName")
```

### 4. 下载数据集（如果网络慢，可提前用代理下载）：

```sh
python download_data.py
```

## 模型详情

### 模型 1：SimpleCNN（基线模型）

- 架构：3 个卷积块（Conv→BN→ReLU→MaxPool）+ 2 个全连接层
- 参数量：620,810（约 62 万）
- 优化器：Adam（lr=0.001）
- 学习率调度：Cosine Annealing
- 正则化：Dropout(0.4) + Weight Decay(1e-4)
- 测试准确率：84.19%

### 模型 2：ResNet-18（从零训练）

- 架构：18 层残差网络，含 8 个 BasicBlock
- 参数量：11,173,962（约 1117 万）
- 优化器：SGD + Momentum(0.9) + Nesterov（lr=0.1）
- 学习率调度：Cosine Annealing
- 正则化：Weight Decay(5e-4)
- CIFAR-10 适配：3×3 conv stride=1（替换标准的 7×7 conv），去掉初始 MaxPool
- 测试准确率：94.31%

### 模型 3：ResNet-18（迁移学习微调）★ 最佳模型

- 基础：ImageNet 预训练 ResNet-18（120 万张图片预训练）
- 参数量：约 1118 万（仅替换最后的 FC 层）
- 输入：224×224（上采样自 32×32），ImageNet 归一化
- 阶段 1：冻结 Backbone，Adam(lr=0.01) 训练分类头，10 epochs
- 阶段 2：解冻全部，判别性学习率 SGD 微调，20 epochs
  - 浅层（Layer1-2）：lr = 0.0001（通用特征，轻微调整）
  - 中层（Layer3）：lr = 0.0005
  - 深层（Layer4+FC）：lr = 0.001（任务特定，较大调整）

- 测试准确率：95.08%

## 核心技术点

### 1. 数据增强（Data Augmentation）

- RandomCrop(32, padding=4)：随机裁剪实现平移增强
- RandomHorizontalFlip：随机水平翻转
- ColorJitter：亮度、对比度、饱和度、色相随机抖动

→ 在不增加标注成本的前提下扩充数据多样性，防止过拟合

### 2. 残差连接（Skip Connection）

- 输入直接加到输出上，为梯度提供"高速公路"
- 解决了深层网络的梯度消失问题
- 18 层网络也能有效训练

### 3. 迁移学习（Transfer Learning）

- 利用 ImageNet 预训练权重的通用视觉特征
- 两阶段微调策略：先训练分类头 → 再全局微调
- 判别性学习率：不同深度使用不同学习率

### 4. 学习率调度（Cosine Annealing）

- 按余弦曲线从初始值衰减到 0
- 训练初期大步快走，后期小步精调

### 5. 早停机制（Early Stopping）

- 验证准确率连续 10 个 epoch 不提升即停止
- 避免过拟合，节省训练时间

### 6. 归一化的重要性

- 从零训练：使用 CIFAR-10 统计值归一化
- 迁移学习：必须使用 ImageNet 统计值归一化

→ 错误使用归一化参数会导致准确率从 95% 暴跌至 77%

## 关键实验结果

最终测试集指标对比：

| 模型 | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| SimpleCNN | 84.19% | 84.12% | 84.19% | 84.13% |
| ResNet-18 (从零) | 94.31% | 94.32% | 94.31% | 94.31% |
| ResNet-18 (迁移学习) | 95.08% | 95.08% | 95.08% | 95.06% |

### 分析要点：

- SimpleCNN 对相似类别（猫/狗）混淆较多（~15% 互相误判）
- ResNet-18 从零训练将猫类 F1 从 68.98% 大幅提升至 87.25%
- 迁移学习进一步将猫类 F1 提升至 90.15%，精确率达 93.08%
- 迁移学习在困难类别上的提升尤为显著

## 文件说明

### `main.py`

主程序入口。依次训练或加载 3 个模型，进行完整评估和对比，输出结论。
通过 SKIP_TRAIN_* 变量灵活控制训练流程。

### `train.py`

通用训练模块。提供 train_epoch（单轮训练）、validate（单轮验证）、
train_model（完整训练流程，支持早停和模型保存）、train_transfer_learning
（两阶段迁移学习训练）。

### `evaluate.py`

模型评估模块。计算 Accuracy、Precision、Recall、F1-Score（Macro + Weighted），
生成分类报告和模型对比。支持多模型批量评估。

### `download_data.py`

CIFAR-10 下载脚本。使用 requests + 代理加速下载，支持断点续传验证。

### `utils/data_loader.py`

数据加载与预处理。提供两种 DataLoader：

- 标准模式（CIFAR-10 归一化，32×32）
- 迁移学习模式（ImageNet 归一化，224×224）

### `utils/plot.py`

可视化模块。生成训练曲线（Loss + Accuracy）、归一化混淆矩阵、
多模型指标对比柱状图。

### `models/simple_cnn.py`

SimpleCNN 模型定义。3 个卷积块 + 2 个全连接层，含 Kaiming 权重初始化。

### `models/resnet_cifar.py`

ResNet-18 模型定义。包含 BasicBlock 和 ResNet 主体类，支持：

- 从零训练（CIFAR-10 适配版）
- 迁移学习（ImageNet 预训练 + 分类头替换 + Backbone 冻结/解冻）

## 常见问题

### Q: 运行时提示显存不足？

A: 在 main.py 中将 BATCH_SIZE 从 128 改为 64 或 32。

### Q: CIFAR-10 下载太慢或失败？

A: 运行 python download_data.py 使用代理下载，或手动下载后放到 data/ 目录。

### Q: 训练太慢？

A: 确保安装了 CUDA 版本的 PyTorch。CPU 训练 ResNet 从零会比较慢，
可以设置 SKIP_TRAIN_* = True 跳过训练，直接用已有权重评估。

### Q: 迁移学习模型准确率只有 ~77%？

A: 检查是否使用了正确的归一化参数。迁移学习必须使用 ImageNet 归一化
（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]）并将图像
放大到 224×224。

## 引用参考

- CIFAR-10 数据集：<https://www.cs.toronto.edu/~kriz/cifar.html>
- ResNet 论文：He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- PyTorch 文档：<https://pytorch.org/docs/stable/>
- torchvision 模型库：<https://pytorch.org/vision/stable/models.html>
