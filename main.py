"""
================================================================================
基于深度学习的 CIFAR-10 图像分类 —— 主程序
================================================================================

项目目标：
  在 CIFAR-10 数据集上，训练并对比 3 种不同策略的图像分类模型，
  记录 Accuracy、Precision、Recall、F1-Score 等指标。

3 个模型（从简单到强大）：
  ┌────────────────────┬──────────────────┬──────────────┬─────────────────────┐
  │ 模型               │ 策略              │ 目标准确率    │ 学什么               │
  ├────────────────────┼──────────────────┼──────────────┼─────────────────────┤
  │ 1. SimpleCNN       │ 从零训练          │ ~78-82%      │ CNN 基础架构          │
  │ 2. ResNet-18 (从头) │ 从零训练          │ ~88-92%      │ 残差网络              │
  │ 3. ResNet-18 (微调) │ 迁移学习 + 微调    │ ~94-96% ★    │ 预训练 + 微调策略     │
  └────────────────────┴──────────────────┴──────────────┴─────────────────────┘

使用方法：
  python main.py

硬件需求：
  - GPU 推荐（RTX 4060 8GB 约 15-20 分钟完成全部训练）
  - CPU 也可以（但 ResNet 从零训练需要更长时间）

输出：
  - results/          → 图表（训练曲线、混淆矩阵、模型对比）
  - checkpoints/      → 训练好的模型权重
================================================================================
"""

import os
import sys

# ============================================================
# 代理配置（使用系统代理 127.0.0.1:7897 加速 CIFAR-10 下载）
# 必须在导入 torchvision 之前设置，因为数据集下载发生在导入时
# ============================================================
_PROXY = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = _PROXY
os.environ["HTTPS_PROXY"] = _PROXY
os.environ["http_proxy"] = _PROXY
os.environ["https_proxy"] = _PROXY

import torch

# 确保能找到项目内的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import get_dataloaders, get_dataloaders_transfer, CLASS_NAMES
from utils.plot import plot_training_curves
from models.simple_cnn import create_simple_cnn
from models.resnet_cifar import create_resnet18_scratch, create_resnet18_transfer
from train import train_model, train_transfer_learning
from evaluate import evaluate_model, run_full_evaluation


def main():
    """
    主流程：依次训练 3 个模型 → 分别评估 → 整体对比

    你可以通过设置下面的变量来跳过某些步骤（方便调试）：
    """
    # ======== 运行控制 ========
    SKIP_TRAIN_SIMPLE = True       # 已训练好（84.19%）
    SKIP_TRAIN_RESNET_SCRATCH = True  # 已训练好（94.31%）
    SKIP_TRAIN_RESNET_TRANSFER = False # ★ 重新训练迁移学习模型

    print("\n" + "="*70)
    print("  基于深度学习的 CIFAR-10 图像分类项目")
    print("  模型: SimpleCNN | ResNet-18 (从零) | ResNet-18 (迁移学习微调)")
    print("="*70)

    # ================================================================
    # 第0步：环境准备
    # ================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[设备] 使用: {device}")
    if device.type == 'cuda':
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 显存优化：如果显存不足可以降低 batch_size
        # torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法

    # ================================================================
    # 第1步：加载数据（所有模型共用）
    # ================================================================
    print(f"\n{'='*60}")
    print("第1步：加载 CIFAR-10 数据")
    print(f"{'='*60}")

    # batch_size=128 在 RTX 4060 8GB 上没问题
    # 如果显存不足，改成 64 或 32
    BATCH_SIZE = 128
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=2)

    # 创建结果和检查点目录
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)

    # ================================================================
    # 模型1：SimpleCNN —— 从零训练的基线模型
    # ================================================================
    print(f"\n{'='*60}")
    print("模型 1/3：SimpleCNN —— 从零训练的基线")
    print(f"{'='*60}")

    model1_name = "SimpleCNN"
    checkpoint1 = f'./checkpoints/{model1_name}_best.pth'

    if not SKIP_TRAIN_SIMPLE:
        model1 = create_simple_cnn(dropout=0.4)

        # 超参数说明：
        # - lr=0.001：Adam 的标准起步学习率
        # - weight_decay=1e-4：L2 正则化，轻微防止过拟合
        # - epochs=50：简单模型需要多跑几轮才能充分收敛
        history1 = train_model(
            model=model1,
            train_loader=train_loader,
            val_loader=test_loader,     # CIFAR-10 测试集当验证集用
            model_name=model1_name,
            num_epochs=50,
            lr=0.001,
            weight_decay=1e-4,
            optimizer_type='adam',
            scheduler_type='cosine',
            device=device,
            patience=10,
        )
        plot_training_curves(history1, model1_name)
    else:
        # 从检查点加载
        checkpoint = torch.load(checkpoint1, map_location=device,
                                weights_only=False)
        model1 = create_simple_cnn()
        model1.load_state_dict(checkpoint['model_state_dict'])
        model1 = model1.to(device)
        print(f"[加载] {model1_name} 权重已从 {checkpoint1} 恢复")

    # ================================================================
    # 模型2：ResNet-18 —— 从零训练
    # ================================================================
    print(f"\n{'='*60}")
    print("模型 2/3：ResNet-18 —— 从零训练")
    print(f"{'='*60}")

    model2_name = "ResNet18_Scratch"
    checkpoint2 = f'./checkpoints/{model2_name}_best.pth'

    if not SKIP_TRAIN_RESNET_SCRATCH:
        model2 = create_resnet18_scratch()

        # 超参数说明：
        # - SGD + momentum + nesterov：ResNet 论文推荐配置
        # - lr=0.1：SGD 用更大的初始学习率（和 Adam 不同）
        # - weight_decay=5e-4：ResNet 标准正则化强度
        # - epochs=50：从零训练 ResNet 需要足够轮数
        history2 = train_model(
            model=model2,
            train_loader=train_loader,
            val_loader=test_loader,
            model_name=model2_name,
            num_epochs=50,
            lr=0.1,
            weight_decay=5e-4,
            optimizer_type='sgd',
            scheduler_type='cosine',
            device=device,
            patience=10,
        )
        plot_training_curves(history2, model2_name)
    else:
        checkpoint = torch.load(checkpoint2, map_location=device,
                                weights_only=False)
        model2 = create_resnet18_scratch()
        model2.load_state_dict(checkpoint['model_state_dict'])
        model2 = model2.to(device)
        print(f"[加载] {model2_name} 权重已从 {checkpoint2} 恢复")

    # ================================================================
    # 模型3：ResNet-18 —— 迁移学习 + 微调（★ 目标最佳模型）
    # ================================================================
    print(f"\n{'='*60}")
    print("模型 3/3：ResNet-18 —— 迁移学习 + 微调（预期最佳）")
    print(f"{'='*60}")

    model3_name = "ResNet18_Transfer"
    checkpoint3 = f'./checkpoints/{model3_name}_best.pth'

    if not SKIP_TRAIN_RESNET_TRANSFER:
        # ★ 为迁移学习创建专用的 DataLoader：
        #   - 使用 ImageNet 归一化（匹配预训练权重）
        #   - 将图像放大到 224×224（预训练模型的标准输入尺寸）
        #   - 较小的 batch_size（224×224 图像占更多显存）
        print("\n[提示] 迁移学习需要 ImageNet 归一化 + 224×224 图像")
        trans_train_loader, trans_test_loader = get_dataloaders_transfer(
            batch_size=64, num_workers=2
        )

        # 阶段1：创建模型，冻结 backbone
        model3 = create_resnet18_transfer(num_classes=10, freeze_backbone=True)

        # 两阶段微调策略：
        #   阶段1: 只训练分类头（backbone冻结），lr=0.01，10 epochs
        #   阶段2: 全局微调（解冻全部），判别性lr，20 epochs
        history3 = train_transfer_learning(
            model=model3,
            train_loader=trans_train_loader,
            val_loader=trans_test_loader,
            model_name=model3_name,
            phase1_epochs=10,
            phase2_epochs=20,
            phase1_lr=0.01,
            phase2_lr=0.001,
            device=device,
        )
        plot_training_curves(history3, model3_name)
    else:
        # 从检查点加载
        model3 = create_resnet18_transfer(num_classes=10, freeze_backbone=False)
        checkpoint = torch.load(checkpoint3, map_location=device,
                                weights_only=False)
        # 迁移学习模型的 state_dict 可能有 key 不匹配，用 strict=False
        model3.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model3 = model3.to(device)
        print(f"[加载] {model3_name} 权重已从 {checkpoint3} 恢复")

    # ================================================================
    # 第4步：全面评估与对比
    # ================================================================
    print(f"\n{'='*60}")
    print("第4步：全面评估与模型对比")
    print(f"{'='*60}")

    models = {
        "SimpleCNN": model1,
        "ResNet18_Scratch": model2,
    }

    # 前两个模型用 CIFAR-10 归一化的测试集
    all_results = run_full_evaluation(models, test_loader, device)

    # ★ 迁移学习模型需要 ImageNet 归一化（单独评估）
    # 避免重复创建 DataLoader（训练时已创建）
    if 'trans_test_loader' not in dir():
        _, trans_test_loader = get_dataloaders_transfer(batch_size=64, num_workers=2)
    result3 = evaluate_model(model3, trans_test_loader, device, "ResNet18_Transfer")
    all_results["ResNet18_Transfer"] = {
        'accuracy': result3['accuracy'],
        'macro_precision': result3['macro_precision'],
        'macro_recall': result3['macro_recall'],
        'macro_f1': result3['macro_f1'],
    }
    from utils.plot import plot_confusion_matrix, plot_model_comparison
    plot_confusion_matrix(result3['y_true'], result3['y_pred'],
                          CLASS_NAMES, "ResNet18_Transfer")
    # 重新生成完整的模型对比图（含迁移学习模型）
    plot_model_comparison(all_results)

    # ================================================================
    # 第5步：结论总结
    # ================================================================
    print(f"\n{'='*60}")
    print("项目总结")
    print(f"{'='*60}")
    print(f"""
  通过对比三种模型策略，可以得出以下结论：

  1. SimpleCNN（{all_results['SimpleCNN']['accuracy']*100:.1f}%）：
     → 基础 CNN 架构可以学到一定特征，但容量有限，容易欠拟合。
     → 单纯堆叠卷积层无法有效提取深层语义。

  2. ResNet-18 从零训练（{all_results['ResNet18_Scratch']['accuracy']*100:.1f}%）：
     → 残差连接让 18 层网络也能有效训练，效果明显优于浅层 CNN。
     → 但从零训练需要大量 epochs，且受限于 CIFAR-10 的 50K 训练样本。

  3. ResNet-18 迁移学习微调（{all_results['ResNet18_Transfer']['accuracy']*100:.1f}%）：
     → ImageNet 预训练权重提供了强大的通用特征提取能力。
     → 两阶段微调策略（先训练分类头，再全局微调）是实用最佳实践。
     → 即使 CIFAR-10 图像只有 32×32，迁移学习依然显著提升了性能。

  迁移学习是实际项目中性价比最高的方案：
     - 不需要从零设计架构
     - 不需要海量训练数据
     - 不需要超长训练时间
     - 效果好、鲁棒性强
""")

    print(f"\n所有结果已保存到 ./results/ 目录")
    print(f"训练好的模型权重保存在 ./checkpoints/ 目录")
    print(f"\n{'='*70}")
    print("  项目完成！")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
