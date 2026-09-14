"""
================================================================================
  单张图片预测脚本 —— 用训练好的模型识别你自己的图片
================================================================================

使用方法：
  python predict.py <图片路径> [--model <模型名>] [--top_k <N>]

示例：
  python predict.py my_cat.jpg                                    # 默认用 ResNet18_Transfer
  python predict.py my_cat.jpg --model SimpleCNN                  # 用 SimpleCNN
  python predict.py my_cat.jpg --model ResNet18_Scratch           # 用 ResNet-18 从零训练
  python predict.py my_cat.jpg --model ResNet18_Transfer --top_k 3  # 显示 Top-3 预测

支持的模型：
  - SimpleCNN          → CIFAR-10 归一化，32×32，准确率 84.19%
  - ResNet18_Scratch   → CIFAR-10 归一化，32×32，准确率 94.31%
  - ResNet18_Transfer  → ImageNet 归一化，224×224，准确率 95.08% ★ 推荐

注意：
  - 本项目模型在 CIFAR-10 上训练，只能识别 10 个类别：
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  - 如果你输入的照片不属于这 10 类，模型也会强行预测成其中一类
  - 建议使用物体居中、背景简洁的照片，效果更好
================================================================================
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# 确保能找到项目内的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from models.simple_cnn import create_simple_cnn
from models.resnet_cifar import create_resnet18_scratch, create_resnet18_transfer


# ============================================================
# 模型注册表：每个模型的配置
# ============================================================
MODEL_CONFIG = {
    "SimpleCNN": {
        "create_fn": lambda: create_simple_cnn(),
        "checkpoint": "./checkpoints/SimpleCNN_best.pth",
        "input_size": 32,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "description": "从零训练的轻量 CNN（84.19%）",
    },
    "ResNet18_Scratch": {
        "create_fn": lambda: create_resnet18_scratch(),
        "checkpoint": "./checkpoints/ResNet18_Scratch_best.pth",
        "input_size": 32,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
        "description": "从零训练的 ResNet-18（94.31%）",
    },
    "ResNet18_Transfer": {
        "create_fn": lambda: create_resnet18_transfer(
            num_classes=10, freeze_backbone=False
        ),
        "checkpoint": "./checkpoints/ResNet18_Transfer_best.pth",
        "input_size": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "description": "ImageNet 预训练 + 微调 ★ 推荐（95.08%）",
    },
}


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    """加载指定模型和训练好的权重。"""
    if model_name not in MODEL_CONFIG:
        available = ", ".join(MODEL_CONFIG.keys())
        raise ValueError(f"未知模型 '{model_name}'。可选: {available}")

    config = MODEL_CONFIG[model_name]

    # 创建模型结构
    print(f"[模型] {model_name}: {config['description']}")
    model = config["create_fn"]()

    # 加载训练好的权重
    checkpoint_path = config["checkpoint"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"权重文件不存在: {checkpoint_path}\n"
            f"请先运行 main.py 训练模型，或检查 checkpoints/ 目录。"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 迁移学习模型的 state_dict key 可能不完全匹配，用 strict=False
    if model_name == "ResNet18_Transfer":
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    val_acc = checkpoint.get("val_acc", 0)
    print(f"[权重] 已加载: {checkpoint_path}")
    print(f"[精度] 训练时验证准确率: {val_acc:.2f}%")

    model = model.to(device)
    model.eval()
    return model


def preprocess_image(
    image_path: str, input_size: int, mean: tuple, std: tuple
) -> torch.Tensor:
    """
    预处理用户输入的图片，使其匹配训练时的输入格式。

    流程：
      1. 打开图片，转为 RGB（处理 PNG 的 RGBA、灰度图等情况）
      2. 缩放到模型需要的尺寸（ResNet_Transfer: 224×224，其他: 32×32）
      3. 中心裁剪（保证尺寸精确）
      4. 转为 Tensor + 归一化
      5. 添加 batch 维度
    """
    # 1. 打开图片
    image = Image.open(image_path).convert("RGB")
    print(f"[图片] 原始尺寸: {image.size}")

    # 2-3. 缩放 + 中心裁剪
    # 先按短边等比缩放，再中心裁剪，避免拉伸变形
    transform_pipeline = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    tensor = transform_pipeline(image)  # shape: (3, H, W)
    tensor = tensor.unsqueeze(0)  # 添加 batch 维度 → (1, 3, H, W)
    print(f"[预处理] 输入尺寸: {input_size}×{input_size}, 归一化完成")
    return tensor


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    对单张图片进行预测。

    参数:
        model: 训练好的模型
        image_tensor: 预处理后的图片张量 (1, 3, H, W)
        device: 计算设备
        top_k: 返回概率最高的 K 个类别

    返回:
        [(类别名, 概率), ...] 的列表，按概率从高到低排序
    """
    image_tensor = image_tensor.to(device)

    # 前向传播
    logits = model(image_tensor)  # shape: (1, 10)

    # Softmax → 概率
    probabilities = F.softmax(logits, dim=1)

    # 取 Top-K
    topk_probs, topk_indices = torch.topk(probabilities, top_k, dim=1)

    # 转为 Python 列表
    topk_probs = topk_probs.squeeze().cpu().numpy()
    topk_indices = topk_indices.squeeze().cpu().numpy()

    results = []
    for idx, prob in zip(topk_indices, topk_probs):
        class_name = CLASS_NAMES[idx]
        results.append((class_name, float(prob)))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="用训练好的 CIFAR-10 分类模型预测单张图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py my_cat.jpg
  python predict.py my_dog.png --model SimpleCNN --top_k 3
  python predict.py photo.jpg --model ResNet18_Scratch
        """,
    )
    parser.add_argument("image", type=str, help="图片文件路径")
    parser.add_argument(
        "--model", "-m", type=str, default="ResNet18_Transfer",
        choices=["SimpleCNN", "ResNet18_Scratch", "ResNet18_Transfer"],
        help="使用的模型（默认: ResNet18_Transfer）",
    )
    parser.add_argument(
        "--top_k", "-k", type=int, default=5,
        help="显示前 K 个预测结果（默认: 5）",
    )
    args = parser.parse_args()

    # ---- 检查图片是否存在 ----
    if not os.path.exists(args.image):
        print(f"[错误] 图片不存在: {args.image}")
        sys.exit(1)

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # ---- 加载模型 ----
    config = MODEL_CONFIG[args.model]
    model = load_model(args.model, device)

    # ---- 预处理图片 ----
    image_tensor = preprocess_image(
        args.image, config["input_size"], config["mean"], config["std"]
    )

    # ---- 预测 ----
    print(f"\n{'='*50}")
    print(f"  预测结果")
    print(f"{'='*50}")

    results = predict(model, image_tensor, device, top_k=args.top_k)

    # 格式化输出
    max_name_len = max(len(name) for name, _ in results)
    for rank, (class_name, prob) in enumerate(results, 1):
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        marker = " ← ★ 最可能" if rank == 1 else ""
        print(f"  {rank}. {class_name:<{max_name_len}s}  {bar}  {prob*100:5.1f}%{marker}")

    # ---- 对用户的提示 ----
    print(f"\n{'─'*50}")
    print("注意: 本模型在 CIFAR-10（10 个类别）上训练，只能识别：")
    print("  ", ", ".join(CLASS_NAMES))
    print("  如果你的图片不属于这些类别，模型仍会预测为其中之一。")

    # ---- 保存结果到文件 ----
    os.makedirs("./results", exist_ok=True)

    # 1) 为每张图片生成独立的结果文件
    image_basename = os.path.splitext(os.path.basename(args.image))[0]
    result_path = f"./results/{image_basename}_prediction.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"图片: {os.path.abspath(args.image)}\n")
        f.write(f"模型: {args.model} ({config['description']})\n")
        f.write(f"设备: {device}\n")
        f.write(f"{'='*50}\n")
        for rank, (class_name, prob) in enumerate(results, 1):
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            marker = " <-- best" if rank == 1 else ""
            f.write(f"{rank}. {class_name:<12s} {bar} {prob*100:5.1f}%{marker}\n")
    print(f"[保存] 预测结果 → {result_path}")

    # 2) 追加到全局预测日志
    log_path = "./results/prediction_history.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        from datetime import datetime
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
        f.write(f"图片={os.path.basename(args.image)}  ")
        f.write(f"模型={args.model}  ")
        f.write(f"结果={results[0][0]}({results[0][1]*100:.1f}%)")
        if len(results) > 1:
            f.write(f"  Top2={results[1][0]}({results[1][1]*100:.1f}%)")
        f.write("\n")
    print(f"[保存] 追加至历史日志 → {log_path}")

    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
