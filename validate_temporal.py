"""
验证2: RBF 激活时序结构定量分析

问题: 正确类的 RBF 激活 p_{t,correct} 是否有非平凡的时序结构？

三个定量指标:
  1. 峰值度: max_t(p_{t,correct}) / mean_t(p_{t,correct})
  2. 类内一致性: 同类样本 p_{t,correct} 曲线的 pairwise Pearson r
  3. 正确 vs 错误方差对比: var(p_{t,correct}) vs var(p_{t,wrong})
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model_v2 import FQN_HEDL_v2
from data_utils import load_and_split_dataset, LOCAL_UCR_PATH


def compute_rbf_activation(model, loader, device, num_classes):
    """对每个样本, 计算逐帧 RBF 激活 p_{t,k} = exp(-||x_t - c_k||^2 / (2*tau^2)).

    返回:
      all_p: list of (T, K) tensors  [每样本一个]
      all_labels: list of int
    """
    model.eval()
    prototypes = model.set_pooling.prototypes  # (K, K)
    tau = model.set_pooling.tau.abs().clamp(min=0.05)

    all_p = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))  # (N, K, T)
            x_flat = bb.permute(0, 2, 1)  # (N, T, K)
            dist_sq = ((x_flat.unsqueeze(2) - prototypes.to(device).unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=3)
            p_per_class = torch.exp(-dist_sq / (2 * tau ** 2))  # (N, T, K)

            for i in range(len(y_batch)):
                all_p.append(p_per_class[i].cpu())  # (T, K)
                all_labels.append(y_batch[i].item())

    return all_p, all_labels


def interpolate_to_fixed_len(p_list, target_len=None):
    """将所有样本的 p 曲线插值到相同长度 (用最小帧数)."""
    if target_len is None:
        target_len = min(p.shape[0] for p in p_list)
    interpolated = []
    for p in p_list:
        # p: (T, K) → 截断或下采样到 target_len
        if p.shape[0] == target_len:
            interpolated.append(p)
        else:
            # 线性插值
            idx = torch.linspace(0, p.shape[0] - 1, target_len)
            idx_floor = idx.long().clamp(0, p.shape[0] - 1)
            idx_ceil = (idx_floor + 1).clamp(0, p.shape[0] - 1)
            alpha = (idx - idx_floor.float()).unsqueeze(1)
            interp = p[idx_floor] * (1 - alpha) + p[idx_ceil] * alpha
            interpolated.append(interp)
    return torch.stack(interpolated, dim=0), target_len


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_code = "D1"
    save_dir = _current_dir / "saved_models"

    # ---- Load data ----
    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    num_classes = data['num_classes']
    batch_size = data['batch_size']

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['X_test']), torch.from_numpy(data['y_test'])),
        batch_size=batch_size, shuffle=False,
    )

    # ---- Load model ----
    model = FQN_HEDL_v2(
        num_channels=data['in_channels'], num_classes=num_classes,
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'],
    ).to(device)

    ckpt_path = save_dir / f"FQN_HEDL_v2_stage2_digamma_{dataset_code}.pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"=== 验证2: RBF 激活时序结构 ({dataset_code}, {data['name']}) ===")
    print(f"  K={num_classes}, tau={model.set_pooling.tau.item():.4f}")

    # ---- Compute per-frame RBF activation ----
    all_p, all_labels = compute_rbf_activation(model, test_loader, device, num_classes)
    print(f"  测试样本数: {len(all_p)}")
    print(f"  帧数范围: [{min(p.shape[0] for p in all_p)}, {max(p.shape[0] for p in all_p)}]")

    # ---- Metric 1: Peak ratio ----
    peak_ratios = {k: [] for k in range(num_classes)}
    for p, label in zip(all_p, all_labels):
        p_correct = p[:, label]  # (T,)
        peak_ratio = (p_correct.max() / (p_correct.mean() + 1e-8)).item()
        peak_ratios[label].append(peak_ratio)

    print(f"\n  --- 指标1: 峰值度 (max/mean) ---")
    all_peak_ratios = []
    for k in range(num_classes):
        vals = peak_ratios[k]
        mean_pr = np.mean(vals)
        all_peak_ratios.extend(vals)
        print(f"    类 {k}: mean={mean_pr:.2f}, std={np.std(vals):.2f}, "
              f"n={len(vals)}, frac>2={np.mean(np.array(vals)>2):.2%}")
    overall_pr = np.mean(all_peak_ratios)
    print(f"    总体 mean={overall_pr:.2f}")

    # ---- Metric 2: Within-class correlation ----
    print(f"\n  --- 指标2: 类内一致性 (pairwise Pearson r) ---")
    for k in range(num_classes):
        # Get p_correct curves for class k, interpolated to common length
        class_p = [all_p[i][:, k] for i in range(len(all_p)) if all_labels[i] == k]
        if len(class_p) < 2:
            print(f"    类 {k}: 样本不足 (<2)")
            continue
        class_p_interp, _ = interpolate_to_fixed_len(class_p)
        # All pairwise correlations
        corrs = []
        for i in range(len(class_p_interp)):
            for j in range(i + 1, len(class_p_interp)):
                r = np.corrcoef(class_p_interp[i].numpy(), class_p_interp[j].numpy())[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
        mean_r = np.mean(corrs) if corrs else 0
        print(f"    类 {k}: mean pairwise r={mean_r:.4f}, n_pairs={len(corrs)}")

    # ---- Metric 3: Correct vs wrong variance ----
    print(f"\n  --- 指标3: 正确类 vs 错误类激活方差 ---")
    correct_vars = []
    wrong_vars = []
    for p, label in zip(all_p, all_labels):
        p_correct = p[:, label]
        correct_vars.append(p_correct.var().item())
        if num_classes == 2:
            wrong_label = 1 - label
            p_wrong = p[:, wrong_label]
            wrong_vars.append(p_wrong.var().item())
    mean_cvar = np.mean(correct_vars)
    mean_wvar = np.mean(wrong_vars) if wrong_vars else float('nan')
    print(f"    正确类方差 mean={mean_cvar:.6f}, std={np.std(correct_vars):.6f}")
    if wrong_vars:
        print(f"    错误类方差 mean={mean_wvar:.6f}, std={np.std(wrong_vars):.6f}")
        print(f"    var(correct)/var(wrong) = {mean_cvar/(mean_wvar+1e-8):.2f}")

    # ---- Summary ----
    print(f"\n{'='*50}")
    verdicts = []
    verdicts.append(overall_pr > 2.0)
    print(f"  峰值度 > 2.0: {overall_pr:.2f} -> {'Y' if overall_pr > 2.0 else 'N'}")

    # Cross-class correlation: p_correct vs p_wrong (opposite trend = structure)
    cross_corrs = []
    for p, label in zip(all_p, all_labels):
        if num_classes == 2:
            pc = p[:, label]
            pw = p[:, 1 - label]
            r = np.corrcoef(pc.numpy(), pw.numpy())[0, 1]
            if not np.isnan(r):
                cross_corrs.append(r)
    if cross_corrs:
        mean_cc = np.mean(cross_corrs)
        print(f"  p_correct 与 p_wrong 互相关: {mean_cc:.4f} (负值=互补结构)")

    n_pass = sum(verdicts)
    if n_pass >= 2:
        verdict = "✓ 源分布假设成立 — 逐帧 RBF 激活有非平凡时序结构"
    elif n_pass == 1:
        verdict = "? 部分成立 — 需结合验证1/3综合判断"
    else:
        verdict = "✗ 源分布假设不成立 — 逐帧 RBF 激活无结构"

    print(f"\n  >>> 判定: {verdict} (通过 {n_pass}/2 项)")

    # Save
    np.savez(
        _current_dir / "validate_temporal_results.npz",
        peak_ratios=all_peak_ratios, overall_peak_ratio=overall_pr,
        correct_vars=correct_vars, wrong_vars=wrong_vars,
    )
    print("  结果已保存到 validate_temporal_results.npz")


if __name__ == "__main__":
    main()
