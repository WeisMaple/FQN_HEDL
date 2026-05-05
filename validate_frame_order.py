"""
验证3+4: 逐帧向量的时间结构 + 原型可视化

问题(修正版): 直接测量 x_t 向量的时间依赖性,而非通过顺序无关的聚合算子间接推断。

方法:
  A. ||x_t|| 沿 T 的自相关函数 (lag 1-5)
  B. 帧位置分类器: MLP(x_t) → 帧位置桶 (前/中/后 1/3)
  C. 原型可视化: c_0 vs c_1 的 K 维柱状对比
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model_v2 import FQN_HEDL_v2
from data_utils import load_and_split_dataset, LOCAL_UCR_PATH


def collect_frame_vectors(model, loader, device):
    """收集所有样本的逐帧向量 x_t.

    返回:
      all_frames: (total_T, K) — 所有帧向量堆叠
      all_positions: (total_T,) — 每帧在序列中的相对位置 [0, 1]
      per_sample_norms: list of (T,) arrays — 每样本的 ||x_t|| 序列
      per_sample_positions: list of (T,) arrays
    """
    model.eval()
    all_frames_list = []
    all_positions_list = []
    per_sample_norms = []
    per_sample_positions = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))  # (N, K, T)
            for i in range(len(y_batch)):
                frames = bb[i].permute(1, 0).cpu()  # (T, K)
                T = frames.shape[0]
                norms = frames.norm(dim=1)  # (T,)
                positions = torch.linspace(0, 1, T)  # (T,)

                all_frames_list.append(frames)
                all_positions_list.append(torch.full((T,), positions.mean().item()))
                per_sample_norms.append(norms.numpy())
                per_sample_positions.append(positions.numpy())

    all_frames = torch.cat(all_frames_list, dim=0)
    return all_frames, per_sample_norms, per_sample_positions


def compute_autocorrelation(x):
    """计算序列 x 的 lag-1 到 lag-5 自相关."""
    x = np.array(x)
    x = x - x.mean()
    denom = np.sum(x ** 2) + 1e-12
    acfs = []
    for lag in range(1, min(6, len(x) - 1)):
        acf = np.sum(x[:-lag] * x[lag:]) / denom
        acfs.append(acf)
    return acfs


class PositionMLP(nn.Module):
    """简单 MLP: K 维帧向量 → 3 个位置桶 (前/中/后)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.net(x)


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

    print(f"=== 验证3+4: 逐帧时间结构 + 原型可视化 ({dataset_code}, {data['name']}) ===")
    print(f"  K={num_classes}, tau={model.set_pooling.tau.item():.4f}")

    # ---- Collect frame vectors ----
    all_frames, per_sample_norms, per_sample_positions = collect_frame_vectors(
        model, test_loader, device
    )
    print(f"  总帧数: {all_frames.shape[0]}, 帧维度: {all_frames.shape[1]}")
    print(f"  样本数: {len(per_sample_norms)}")

    # ---- A: Autocorrelation ----
    print(f"\n  --- A: ||x_t|| 自相关分析 ---")
    all_acfs = {lag: [] for lag in range(1, 6)}
    for norms in per_sample_norms:
        acfs = compute_autocorrelation(norms)
        for lag_i, acf in enumerate(acfs):
            all_acfs[lag_i + 1].append(acf)

    for lag in range(1, 6):
        vals = all_acfs[lag]
        mean_acf = np.mean(vals)
        print(f"    lag-{lag}: mean={mean_acf:.4f}, std={np.std(vals):.4f}, "
              f"frac>0.5={np.mean(np.array(vals)>0.5):.2%}")

    lag1_mean = np.mean(all_acfs[1])
    has_autocorr = lag1_mean > 0.5

    # ---- B: Position classifier ----
    print(f"\n  --- B: 帧位置分类器 ---")
    # Assign each frame to one of 3 position buckets (T/3)
    all_frames_np = all_frames.numpy()
    K_dim = all_frames.shape[1]

    # For each sample, compute each frame's position bucket
    position_buckets = []
    for norms, positions in zip(per_sample_norms, per_sample_positions):
        T = len(norms)
        bucket_size = max(T // 3, 1)
        buckets = np.clip(np.arange(T) // bucket_size, 0, 2)
        position_buckets.append(buckets)

    all_buckets = np.concatenate(position_buckets, axis=0)

    # Train/test split for position classifier (80/20)
    ds = TensorDataset(torch.from_numpy(all_frames_np).float(),
                       torch.from_numpy(all_buckets).long())
    n_train = int(0.8 * len(ds))
    n_test = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    train_loader_pos = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader_pos = DataLoader(test_ds, batch_size=256, shuffle=False)

    pos_model = PositionMLP(K_dim).to(device)
    optimizer = torch.optim.Adam(pos_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        pos_model.train()
        for X_b, y_b in train_loader_pos:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(pos_model(X_b), y_b)
            loss.backward()
            optimizer.step()

    pos_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader_pos:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = pos_model(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
    pos_acc = correct / total
    random_baseline = 1/3
    print(f"    测试准确率: {pos_acc:.4f} (随机基线: {random_baseline:.4f})")
    print(f"    高于基线: {pos_acc - random_baseline:.4f}")
    pos_predictable = pos_acc > random_baseline + 0.1

    # ---- C: Prototype visualization ----
    print(f"\n  --- C: 原型可视化 ---")
    prototypes = model.set_pooling.prototypes.cpu().numpy()  # (K, K)
    for k in range(num_classes):
        proto_k = prototypes[k]
        print(f"    c_{k} = [{', '.join(f'{v:.4f}' for v in proto_k)}]")
    if num_classes == 2:
        diff = prototypes[0] - prototypes[1]
        print(f"    c_0 - c_1 = [{', '.join(f'{v:.4f}' for v in diff)}]")
        print(f"    ||c_0 - c_1|| = {np.linalg.norm(diff):.4f}")

        # Channel-wise comparison
        print(f"\n    Channel |   c_0   |   c_1   |  diff  | dominant")
        print(f"    " + "-" * 55)
        for j in range(num_classes):
            d = prototypes[0, j] - prototypes[1, j]
            dom = "class 0" if d > 0 else "class 1"
            print(f"    {j:7d} | {prototypes[0,j]:7.4f} | {prototypes[1,j]:7.4f} | {d:+7.4f} | {dom}")

    # ---- Summary ----
    print(f"\n{'='*50}")
    n_pass = int(has_autocorr) + int(pos_predictable)
    print(f"  lag-1 自相关 > 0.5: {'Y' if has_autocorr else 'N'} ({lag1_mean:.4f})")
    print(f"  位置可预测 (acc > 0.43): {'Y' if pos_predictable else 'N'} ({pos_acc:.4f})")

    if n_pass == 2:
        verdict = "✓ 时间结构存在 — x_t 有明显的时间依赖性"
    elif n_pass == 1:
        verdict = "? 部分成立 — 时间结构弱, 需结合验证1/2综合判断"
    else:
        verdict = "✗ 时间结构不存在 — x_t 近似帧间独立"

    print(f"\n  >>> 判定: {verdict} (通过 {n_pass}/2 项)")

    # Save
    np.savez(
        _current_dir / "validate_frame_order_results.npz",
        lag1_acfs=all_acfs[1], lag1_mean=lag1_mean,
        pos_acc=pos_acc, random_baseline=random_baseline,
        prototypes=prototypes,
    )
    print("  结果已保存到 validate_frame_order_results.npz")


if __name__ == "__main__":
    main()
