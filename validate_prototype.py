"""
验证1: 原型模板测试

问题: c_k 是否具有"类别模板"意义——能否用它来对单帧做独立分类？

方法:
- 加载 D1 Stage 2 checkpoint
- 用训练集计算硬标签原型 c_0, c_1
- 对测试集每帧用最近原型做独立分类
- 100 次 shuffled-labels 基线计算 z-score
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


def compute_prototypes(model, loader, device, num_classes):
    """硬标签原型: 每类所有帧向量的均值."""
    model.eval()
    bb_by_class = {k: [] for k in range(num_classes)}
    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))  # (N, K, T)
            for i in range(len(y_batch)):
                lbl = y_batch[i].item()
                frames = bb[i].permute(1, 0).cpu()  # (T, K)
                bb_by_class[lbl].append(frames)
    prototypes = torch.zeros(num_classes, num_classes)
    for k in range(num_classes):
        if bb_by_class[k]:
            all_k = torch.cat(bb_by_class[k], dim=0)  # (total_T, K)
            prototypes[k] = all_k.mean(dim=0)
    return prototypes


def per_frame_accuracy(model, loader, device, prototypes):
    """逐帧最近原型分类, 返回每帧是否被正确分配到样本真实类别."""
    model.eval()
    all_correct = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))  # (N, K, T)
            for i in range(len(y_batch)):
                true_label = y_batch[i].item()
                frames = bb[i].permute(1, 0)  # (T, K)
                dists = torch.cdist(frames, prototypes.to(device))  # (T, num_classes)
                preds = dists.argmin(dim=1)  # (T,)
                all_correct.append((preds == true_label).cpu())
    return torch.cat(all_correct, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_code = "D1"
    save_dir = _current_dir / "saved_models"

    # ---- Load data ----
    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    num_classes = data['num_classes']
    batch_size = data['batch_size']

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['X_train']), torch.from_numpy(data['y_train'])),
        batch_size=batch_size, shuffle=False,
    )
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Info] Missing keys: {missing}")
    model.eval()

    # ---- Real prototypes ----
    real_proto = compute_prototypes(model, train_loader, device, num_classes)
    real_correct = per_frame_accuracy(model, test_loader, device, real_proto)
    real_acc = real_correct.float().mean().item()

    print(f"=== 验证1: 原型模板测试 ({dataset_code}, {data['name']}) ===")
    print(f"  K={num_classes}, tau={model.set_pooling.tau.item():.4f}")
    print(f"  测试集帧总数: {len(real_correct)}")
    print(f"  真实原型逐帧分类正确率: {real_acc:.4f} ({real_acc*100:.2f}%)")
    for k in range(num_classes):
        proto_k = real_proto[k].tolist()
        print(f"  c_{k} = [{', '.join(f'{v:.4f}' for v in proto_k)}]")
    inter_class_dist = (real_proto[0] - real_proto[1]).norm().item() if num_classes == 2 else 0
    print(f"  ||c_0 - c_1|| = {inter_class_dist:.4f}")

    # ---- Random baselines (shuffled labels) ----
    y_train_np = data['y_train'].copy()
    rand_accs = []
    for seed in range(100):
        rng = np.random.RandomState(seed)
        y_shuffled = rng.permutation(y_train_np)
        rand_loader = DataLoader(
            TensorDataset(torch.from_numpy(data['X_train']), torch.from_numpy(y_shuffled)),
            batch_size=batch_size, shuffle=False,
        )
        rand_proto = compute_prototypes(model, rand_loader, device, num_classes)
        rand_correct = per_frame_accuracy(model, test_loader, device, rand_proto)
        rand_accs.append(rand_correct.float().mean().item())

    rand_mean = np.mean(rand_accs)
    rand_std = np.std(rand_accs)
    z_score = (real_acc - rand_mean) / rand_std if rand_std > 0 else float('inf')

    print(f"\n  随机基线 (100次 shuffled labels):")
    print(f"    mean={rand_mean:.4f}, std={rand_std:.4f}")
    print(f"    range=[{min(rand_accs):.4f}, {max(rand_accs):.4f}]")
    print(f"  z-score = {z_score:.2f}")

    if z_score > 3:
        verdict = "✓ 源分布假设成立 — 原型有显著的类别特异性"
    elif z_score > 2:
        verdict = "? 灰色地带 — z 在 2-3 之间, 需结合验证2/3综合判断"
    else:
        verdict = "✗ 源分布假设不成立 — 原型无类别特异性"

    print(f"\n  >>> 判定: {verdict}")

    # Save results
    np.savez(
        _current_dir / "validate_prototype_results.npz",
        real_acc=real_acc, rand_accs=rand_accs,
        rand_mean=rand_mean, rand_std=rand_std, z_score=z_score,
    )
    print("  结果已保存到 validate_prototype_results.npz")


if __name__ == "__main__":
    main()
