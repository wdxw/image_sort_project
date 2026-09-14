"""
通用训练模块
============
提供与模型无关的训练循环，支持：
  - 标准训练/验证流程
  - 学习率调度（CosineAnnealing、ReduceLROnPlateau）
  - 模型保存（仅保存验证集表现最好的权重）
  - 训练历史记录（用于后续画图）

教学要点：
  - 训练循环的每一步在做什么？
  - 为什么需要验证集？（监控过拟合）
  - 学习率调度器如何帮助训练？
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import os
import copy
from typing import Callable


# ============================================================
# 单 epoch 训练
# ============================================================
def train_epoch(model: nn.Module, loader, criterion, optimizer,
                device: torch.device, epoch: int, total_epochs: int,
                scheduler=None) -> tuple:
    """
    训练一个 epoch。

    流程（每一步的作用）：
      1. model.train() → 启用 Dropout、BatchNorm 的训练行为
      2. 从 DataLoader 取一批数据
      3. 数据送入 GPU（如果有）
      4. optimizer.zero_grad() → 清零梯度（PyTorch 默认累加梯度）
      5. model(data) → 前向传播，得到预测
      6. criterion(output, target) → 计算损失
      7. loss.backward() → 反向传播：自动求每个参数的梯度
      8. optimizer.step() → 根据梯度更新参数

    返回:
        (avg_loss, accuracy) 该 epoch 的平均损失和准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # tqdm 进度条——让你直观看到训练进度
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [Train]',
                leave=False, ncols=100, ascii=True)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        # 数据移到设备（CPU 或 GPU）
        inputs, targets = inputs.to(device), targets.to(device)

        # === 核心训练5步 ===
        optimizer.zero_grad()                   # 1. 清零梯度
        outputs = model(inputs)                 # 2. 前向传播
        loss = criterion(outputs, targets)      # 3. 计算损失
        loss.backward()                         # 4. 反向传播（求梯度）
        optimizer.step()                        # 5. 更新参数

        # === 统计 ===
        running_loss += loss.item()
        _, predicted = outputs.max(1)           # 取分数最高的类别
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条显示
        pbar.set_postfix({
            'Loss': f'{loss.item():.3f}',
            'Acc': f'{100. * correct / total:.1f}%'
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================================
# 单 epoch 验证
# ============================================================
@torch.no_grad()  # 装饰器：禁用梯度计算 → 省显存、速度快
def validate(model: nn.Module, loader, criterion, device: torch.device) -> tuple:
    """
    在验证集/测试集上评估模型。

    与训练的区别：
      - model.eval() → Dropout 关闭，BatchNorm 用移动平均统计量
      - torch.no_grad() → 不构建计算图，不存梯度 → 省显存
      - 不调用 optimizer.step() → 参数不更新

    返回:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='[Valid]', leave=False, ncols=100, ascii=True)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            'Loss': f'{loss.item():.3f}',
            'Acc': f'{100. * correct / total:.1f}%'
        })

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================================
# 完整训练流程
# ============================================================
def train_model(model: nn.Module,
                train_loader,
                val_loader,
                model_name: str,
                num_epochs: int = 30,
                lr: float = 0.001,
                weight_decay: float = 0.0,
                optimizer_type: str = 'adam',
                scheduler_type: str = 'cosine',
                device: torch.device | None = None,
                save_dir: str = './checkpoints',
                patience: int = 10) -> dict:
    """
    完整的模型训练流程。

    参数:
        model:          PyTorch 模型
        train_loader:   训练集 DataLoader
        val_loader:     验证集 DataLoader
        model_name:     模型名称（用于保存文件名和日志）
        num_epochs:     训练总轮数
        lr:             初始学习率
        weight_decay:   权重衰减系数（L2 正则化，防止过拟合）
        optimizer_type: 'adam' 或 'sgd'
        scheduler_type: 'cosine' 或 'plateau' 或 'none'
        device:         训练设备（GPU/CPU）
        save_dir:       模型保存目录
        patience:       早停耐心值（多少 epoch 没提升就停止）

    返回:
        history: 训练历史字典
    """
    # ---- 设备选择 ----
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"设备: {device}")
    print(f"Epochs: {num_epochs} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"{'='*60}")

    model = model.to(device)

    # ---- 损失函数 ----
    # CrossEntropyLoss：内部包含 LogSoftmax + NLLLoss
    # 输入是 logits（未归一化分数），不需要先做 softmax
    criterion = nn.CrossEntropyLoss()

    # ---- 优化器 ----
    # Adam：自适应学习率，大多数情况好用、调参简单
    # SGD + Momentum：经典选择，ResNet 论文推荐，需要仔细调参但上限更高
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.9, weight_decay=weight_decay,
                              nesterov=True)  # Nesterov 加速梯度
    else:
        raise ValueError(f"未知优化器: {optimizer_type}")

    # ---- 学习率调度器 ----
    # CosineAnnealing：学习率按余弦曲线衰减（从 lr → 0）
    #   → 这是现代训练的标准做法，比固定 lr 效果好得多
    # ReduceLROnPlateau：验证 loss 不降时自动降低 lr
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                       patience=5, verbose=True)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"未知调度器: {scheduler_type}")

    # ---- 训练循环 ----
    os.makedirs(save_dir, exist_ok=True)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }

    best_val_acc = 0.0       # 最佳验证准确率
    best_model_state = None  # 最佳模型权重
    no_improve = 0           # 连续未提升计数

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, num_epochs, scheduler
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 更新学习率（Cosine/None 在每个epoch后更新，Plateau 需要val_loss）
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        # 当前学习率（方便监控）
        current_lr = optimizer.param_groups[0]['lr']

        # 打印 epoch 总结
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            print(f"  ★ 新的最佳模型！Val Acc = {val_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⚠ 连续 {patience} 个 epoch 无提升，早停！")
                break

    # ---- 保存最佳权重 ----
    if best_model_state is not None:
        save_path = os.path.join(save_dir, f'{model_name}_best.pth')
        torch.save({
            'model_state_dict': best_model_state,
            'val_acc': best_val_acc,
            'history': history,
        }, save_path)
        print(f"\n[模型保存] 最佳权重 → {save_path} (Val Acc: {best_val_acc:.2f}%)")

    # 加载最佳权重到模型
    model.load_state_dict(best_model_state)
    return history


# ============================================================
# 专用：两阶段迁移学习训练
# ============================================================
def train_transfer_learning(model: nn.Module,
                            train_loader,
                            val_loader,
                            model_name: str,
                            phase1_epochs: int = 10,
                            phase2_epochs: int = 20,
                            phase1_lr: float = 0.01,
                            phase2_lr: float = 0.001,
                            device: torch.device | None = None,
                            save_dir: str = './checkpoints') -> dict:
    """
    迁移学习的两阶段训练。

    阶段1：只训练分类头（backbone 冻结）
      - 学习率较大 → 分类头快速收敛
      - epoch 较少 → 避免分类头在随机初始化下过拟合

    阶段2：全局微调（解冻全部参数）
      - 学习率较小 → 不破坏预训练特征
      - epoch 较多 → 整体适应新数据分布

    返回:
        合并后的 history 字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"迁移学习训练: {model_name}")
    print(f"{'='*60}")

    # ---- 阶段1：训练分类头 ----
    print("\n" + "─" * 40)
    print("阶段1：冻结 Backbone，训练分类头")
    print("─" * 40)

    # 只优化 fc 层的参数
    phase1_params = [p for name, p in model.named_parameters()
                     if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(phase1_params, lr=phase1_lr)
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=phase1_epochs)

    model = model.to(device)
    history_all = {'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': []}

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer1,
            device, epoch, phase1_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler1.step()

        history_all['train_loss'].append(train_loss)
        history_all['val_loss'].append(val_loss)
        history_all['train_acc'].append(train_acc)
        history_all['val_acc'].append(val_acc)

        print(f"Phase1 Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)

    # ---- 阶段2：全局微调 ----
    print("\n" + "─" * 40)
    print("阶段2：解冻全部参数，全局微调")
    print("─" * 40)

    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    # 判别性学习率：浅层用更小的 lr（特征更通用），深层用稍大的 lr
    optimizer2 = optim.SGD([
        {'params': [p for n, p in model.named_parameters()
                    if 'layer4' in n or 'fc' in n],
         'lr': phase2_lr},                               # 深层：标准 lr
        {'params': [p for n, p in model.named_parameters()
                    if 'layer3' in n],
         'lr': phase2_lr * 0.5},                         # 中层：一半 lr
        {'params': [p for n, p in model.named_parameters()
                    if 'layer4' not in n and 'layer3' not in n
                    and 'fc' not in n],
         'lr': phase2_lr * 0.1},                         # 浅层：1/10 lr
    ], momentum=0.9, weight_decay=1e-4)

    scheduler2 = CosineAnnealingLR(optimizer2, T_max=phase2_epochs)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, phase2_epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer2,
            device, epoch, phase2_epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler2.step()

        history_all['train_loss'].append(train_loss)
        history_all['val_loss'].append(val_loss)
        history_all['train_acc'].append(train_acc)
        history_all['val_acc'].append(val_acc)

        current_lr = optimizer2.param_groups[0]['lr']
        print(f"Phase2 Epoch {epoch:2d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
        save_path = os.path.join(save_dir, f'{model_name}_best.pth')
        torch.save({
            'model_state_dict': best_state,
            'val_acc': best_val_acc,
            'history': history_all,
        }, save_path)
        print(f"\n[模型保存] 最佳权重 → {save_path} (Val Acc: {best_val_acc:.2f}%)")

    return history_all
