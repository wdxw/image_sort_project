"""
可视化模块
==========
功能：绘制训练曲线、混淆矩阵、模型对比图

使用 matplotlib + seaborn，所有图表自动保存到 results/ 目录。
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器/脚本环境
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

# 设置中文字体（如果可用），否则回退到英文
plt.rcParams['font.family'] = 'sans-serif'
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

# 全局风格
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

RESULT_DIR = './results'
os.makedirs(RESULT_DIR, exist_ok=True)


def plot_training_curves(history: dict, model_name: str, save: bool = True):
    """
    绘制训练/验证的 Loss 和 Accuracy 曲线。

    参数:
        history: 字典，包含 'train_loss', 'val_loss', 'train_acc', 'val_acc' 列表
        model_name: 模型名称（用于标题和文件名）
        save: 是否保存到文件
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图：Loss 曲线 ----
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- 右图：Accuracy 曲线 ----
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULT_DIR, f'{model_name}_training_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [保存] 训练曲线 → {path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list, model_name: str, save: bool = True):
    """
    绘制归一化的混淆矩阵。

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        model_name: 模型名称
        save: 是否保存
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 按行归一化（召回率视角）

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Recall Rate'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{model_name} - Confusion Matrix (Normalized by Row)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULT_DIR, f'{model_name}_confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  [保存] 混淆矩阵 → {path}")
    plt.close()


def plot_model_comparison(results: dict, save: bool = True):
    """
    绘制多模型指标对比柱状图。

    参数:
        results: 嵌套字典，格式如:
            {'SimpleCNN': {'accuracy': 0.80, 'macro_precision': 0.79, ...},
             'ResNet18_Scratch': {...},
             'ResNet18_Transfer': {...}}
        save: 是否保存
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']

    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results[m][metric] * 100 for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)

        # 在柱子上标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Comparison: Classification Metrics on CIFAR-10 Test Set')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if save:
        path = os.path.join(RESULT_DIR, 'model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n[保存] 模型对比图 → {path}")
    plt.close()
