"""
模型评估与对比模块
==================
功能：
  1. 在测试集上评估单个模型，输出完整的分类指标
  2. 对比多个模型的表现
  3. 输出分类报告、混淆矩阵、对比图表

评估指标详解（以"识别猫"为例）：
  ┌──────────────┬───────────────────────┬───────────────────────┐
  │              │   预测为猫             │   预测为非猫           │
  ├──────────────┼───────────────────────┼───────────────────────┤
  │ 实际是猫      │   TP (True Positive)  │   FN (False Negative)  │
  │ 实际不是猫    │   FP (False Positive) │   TN (True Negative)   │
  └──────────────┴───────────────────────┴───────────────────────┘

  Accuracy  = (TP+TN) / 全部        → 预测正确的比例（各类均衡时好用）
  Precision = TP / (TP+FP)          → 预测为"猫"的样本中，真的是猫的比例（宁缺毋滥）
  Recall    = TP / (TP+FN)          → 所有真猫中，被正确识别的比例（宁可错杀不可放过）
  F1-Score  = 2 × (P×R) / (P+R)    → Precision 和 Recall 的调和平均（综合指标）

  Macro平均：先对每个类别算指标，再取平均（每个类别权重相同）
  Weighted平均：按每类的样本数加权平均（样本多的类权重大）
"""

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)

from utils.data_loader import CLASS_NAMES
from utils.plot import plot_training_curves, plot_confusion_matrix, plot_model_comparison


@torch.no_grad()
def evaluate_model(model: nn.Module, test_loader, device: torch.device,
                   model_name: str = "Model") -> dict:
    """
    在测试集上全面评估模型。

    参数:
        model:      训练好的 PyTorch 模型
        test_loader: 测试集 DataLoader
        device:     设备
        model_name: 用于打印的模型名称

    返回:
        指标字典: {'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
                   'weighted_precision', 'weighted_recall', 'weighted_f1',
                   'y_true', 'y_pred', 'per_class_metrics'}
    """
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")

    model.eval()
    model = model.to(device)

    all_targets = []
    all_predictions = []

    pbar = tqdm(test_loader, desc=f'[评估] {model_name}', ncols=100, ascii=True)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_targets.extend(targets.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    # ---- 计算指标 ----
    # sklearn 的 precision_recall_fscore_support 一次性计算所有指标
    # average='macro': 每个类别权重相同
    # average='weighted': 按每类样本数加权
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    # 每个类别的详细指标
    per_class_prec, per_class_rec, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true, y_pred, average=None,
                                        zero_division=0,
                                        labels=range(len(CLASS_NAMES)))

    # ---- 打印结果 ----
    print(f"\n{'─'*50}")
    print(f"  {model_name} - 测试集评估结果")
    print(f"{'─'*50}")
    print(f"  Accuracy:         {accuracy*100:.2f}%")
    print(f"  ── Macro Average (每类权重相同) ──")
    print(f"  Precision:        {macro_prec*100:.2f}%")
    print(f"  Recall:           {macro_rec*100:.2f}%")
    print(f"  F1-Score:         {macro_f1*100:.2f}%")
    print(f"  ── Weighted Average (按样本数加权) ──")
    print(f"  Precision:        {weighted_prec*100:.2f}%")
    print(f"  Recall:           {weighted_rec*100:.2f}%")
    print(f"  F1-Score:         {weighted_f1*100:.2f}%")
    print(f"  ── Per-Class ──")
    print(f"  {'Class':<12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    print(f"  {'─'*50}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<12s} {per_class_prec[i]*100:>9.2f}% "
              f"{per_class_rec[i]*100:>9.2f}% {per_class_f1[i]*100:>9.2f}% "
              f"{per_class_support[i]:>8d}")

    # ---- 保存分类报告到文件 ----
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                   digits=4, zero_division=0)
    os.makedirs('./results', exist_ok=True)
    report_path = os.path.join('./results', f'{model_name}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
        f.write(report)
    print(f"\n  [保存] 分类报告 → {report_path}")

    return {
        'accuracy': accuracy,
        'macro_precision': macro_prec,
        'macro_recall': macro_rec,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_prec,
        'weighted_recall': weighted_rec,
        'weighted_f1': weighted_f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'per_class': {
            'precision': per_class_prec,
            'recall': per_class_rec,
            'f1': per_class_f1,
            'support': per_class_support,
        }
    }


def run_full_evaluation(models: dict, test_loader, device: torch.device):
    """
    对多个模型进行完整评估并生成对比结果。

    参数:
        models: {'模型名': model_instance} 的字典
        test_loader: 测试集 DataLoader
        device: 计算设备

    返回:
        results: {'模型名': metrics_dict} 的嵌套字典
    """
    all_results = {}

    for model_name, model in models.items():
        # 评估
        result = evaluate_model(model, test_loader, device, model_name)

        # 混淆矩阵
        plot_confusion_matrix(result['y_true'], result['y_pred'],
                              CLASS_NAMES, model_name)

        all_results[model_name] = {
            'accuracy': result['accuracy'],
            'macro_precision': result['macro_precision'],
            'macro_recall': result['macro_recall'],
            'macro_f1': result['macro_f1'],
        }

    # ---- 模型对比 ----
    print(f"\n{'='*60}")
    print("模型对比总结")
    print(f"{'='*60}")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Macro Prec':>10s} "
          f"{'Macro Rec':>10s} {'Macro F1':>10s}")
    print(f"  {'─'*65}")
    for name, m in all_results.items():
        print(f"  {name:<25s} {m['accuracy']*100:>9.2f}% "
              f"{m['macro_precision']*100:>9.2f}% {m['macro_recall']*100:>9.2f}% "
              f"{m['macro_f1']*100:>9.2f}%")

    # 找到最佳模型
    best_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    print(f"\n  ★ 最佳模型: {best_name} "
          f"(Accuracy: {all_results[best_name]['accuracy']*100:.2f}%)")

    # 画对比柱状图
    plot_model_comparison(all_results)

    # ---- 保存对比结果到 CSV ----
    import pandas as pd
    df_rows = []
    for name, m in all_results.items():
        df_rows.append({
            'Model': name,
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Macro_Precision': f"{m['macro_precision']*100:.2f}%",
            'Macro_Recall': f"{m['macro_recall']*100:.2f}%",
            'Macro_F1': f"{m['macro_f1']*100:.2f}%",
        })
    df = pd.DataFrame(df_rows)
    csv_path = os.path.join('./results', 'model_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] 对比表格 → {csv_path}")

    return all_results
