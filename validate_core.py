"""
源分布假设 — 核心验证（跨数据集）

分析项:
  A. 逐帧分类准确率 (D1, D3, D5, D8) — 原型是否有类别模板意义
  B. 原型距离矩阵 — 各类原型之间的几何分离度
  C. 帧-原型距离分布 (ID vs OOD) — RBF 激活能否区分 ID/OOD
  D. 帧数 (T2) 报告 — 评估时序结构分析的基础条件
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
from data_utils import load_and_split_dataset, prepare_ood_data, get_dataset_config, LOCAL_UCR_PATH


def compute_prototypes(model, loader, device, num_classes):
    """硬标签原型: 每类所有帧向量的均值."""
    model.eval()
    bb_by_class = {k: [] for k in range(num_classes)}
    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))
            for i in range(len(y_batch)):
                lbl = y_batch[i].item()
                frames = bb[i].permute(1, 0).cpu()
                bb_by_class[lbl].append(frames)
    prototypes = torch.zeros(num_classes, num_classes)
    for k in range(num_classes):
        if bb_by_class[k]:
            all_k = torch.cat(bb_by_class[k], dim=0)
            prototypes[k] = all_k.mean(dim=0)
    return prototypes


def per_frame_accuracy(model, loader, device, prototypes):
    """逐帧最近原型分类正确率."""
    model.eval()
    all_correct = []
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))
            for i in range(len(y_batch)):
                true_label = y_batch[i].item()
                frames = bb[i].permute(1, 0)
                dists = torch.cdist(frames, prototypes.to(device))
                preds = dists.argmin(dim=1)
                correct = (preds == true_label)
                all_correct.append(correct.cpu())
                per_class_correct.setdefault(true_label, []).append(correct.sum().item())
                per_class_total.setdefault(true_label, []).append(correct.shape[0])
    all_correct = torch.cat(all_correct, dim=0)
    per_class_acc = {}
    for k in per_class_correct:
        per_class_acc[k] = sum(per_class_correct[k]) / sum(per_class_total[k])
    return all_correct.float().mean().item(), per_class_acc


def frame_distance_distribution(model, loader, device, prototypes, num_classes):
    """收集每帧到各类原型的距离分布.

    返回:
      dist_to_correct: list of float — ID 帧到其正确原型的距离
      dist_to_nearest_wrong: list of float — ID 帧到最近错误原型的距离
      margin: list of float — dist_to_nearest_wrong - dist_to_correct (正=可分)
    """
    model.eval()
    dist_to_correct = []
    dist_to_nearest_wrong = []
    margins = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            bb = model.forward_backbone(X_batch.to(device))
            for i in range(len(y_batch)):
                true_label = y_batch[i].item()
                frames = bb[i].permute(1, 0)  # (T, K)
                dists = torch.cdist(frames, prototypes.to(device))  # (T, num_classes)
                d_correct = dists[:, true_label]
                dist_to_correct.extend(d_correct.cpu().tolist())
                # distance to nearest WRONG prototype
                mask = torch.ones(num_classes, dtype=torch.bool)
                mask[true_label] = False
                d_wrong = dists[:, mask].min(dim=1).values
                dist_to_nearest_wrong.extend(d_wrong.cpu().tolist())
                margins.extend((d_wrong - d_correct).cpu().tolist())
    return (np.array(dist_to_correct), np.array(dist_to_nearest_wrong), np.array(margins))


def ood_distance_to_nearest(model, ood_loader, device, prototypes):
    """OOD 每帧到最近的 ID 原型的距离."""
    model.eval()
    distances = []
    with torch.no_grad():
        for X_batch, _ in ood_loader:
            bb = model.forward_backbone(X_batch.to(device))
            for i in range(bb.shape[0]):
                frames = bb[i].permute(1, 0)  # (T, K)
                dists = torch.cdist(frames, prototypes.to(device))  # (T, num_classes)
                nearest = dists.min(dim=1).values
                distances.extend(nearest.cpu().tolist())
    return np.array(distances)


def prototype_distance_matrix(prototypes):
    """原型的 pairwise L2 距离矩阵."""
    K = prototypes.shape[0]
    full_dists = torch.cdist(prototypes, prototypes).cpu().numpy()
    # 只取上三角 (不含对角线)
    triu = full_dists[np.triu_indices(K, k=1)]
    return full_dists, triu


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["D1", "D3", "D5", "D8"]
    save_dir = _current_dir / "saved_models"

    all_results = {}

    for dataset_code in datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset_code}")
        print(f"{'='*60}")

        data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
        num_classes = data['num_classes']
        batch_size = data['batch_size']
        name = data['name']

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(data['X_train']), torch.from_numpy(data['y_train'])),
            batch_size=batch_size, shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(data['X_test']), torch.from_numpy(data['y_test'])),
            batch_size=batch_size, shuffle=False,
        )

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

        tau = model.set_pooling.tau.item()
        prototypes = model.set_pooling.prototypes.clone()

        # ---- Check T2 for one batch ----
        with torch.no_grad():
            sample_batch = torch.from_numpy(data['X_test'][:1]).to(device)
            bb_out = model.forward_backbone(sample_batch)
            T2 = bb_out.shape[2]
        print(f"  {name}: K={num_classes}, tau={tau:.4f}, T2={T2} (input_len={data['X_test'].shape[2]})")

        # ---- A: Per-frame accuracy ----
        real_acc, per_class_acc = per_frame_accuracy(model, test_loader, device, prototypes)
        print(f"\n  [A] 逐帧分类正确率: {real_acc:.4f} ({real_acc*100:.2f}%)")
        for k in sorted(per_class_acc.keys()):
            print(f"      类 {k}: {per_class_acc[k]:.4f}")

        # Shuffled-labels baseline
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
            rand_correct, _ = per_frame_accuracy(model, test_loader, device, rand_proto)
            rand_accs.append(rand_correct)
        rand_mean = np.mean(rand_accs)
        rand_std = np.std(rand_accs)
        z_score = (real_acc - rand_mean) / rand_std if rand_std > 0 else float('inf')
        print(f"  随机基线: mean={rand_mean:.4f}, std={rand_std:.4f}, range=[{min(rand_accs):.4f}, {max(rand_accs):.4f}]")
        print(f"  z-score={z_score:.2f}  {'(K=2 时双峰分布, z-score 被低估)' if num_classes == 2 else ''}")

        # ---- B: Prototype distance matrix ----
        full_dists, triu_dists = prototype_distance_matrix(prototypes)
        print(f"\n  [B] 原型距离矩阵:")
        print(f"      pairwise L2 distances:")
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                print(f"      c_{i} - c_{j}: {full_dists[i,j]:.4f}")
        print(f"      min={triu_dists.min():.4f}, max={triu_dists.max():.4f}, mean={triu_dists.mean():.4f}")
        print(f"      tau={tau:.4f} (RBF 激活半径)")
        # 多少对原型的距离 < tau (即 RBF 无法有效区分)
        close_pairs = (triu_dists < tau).sum()
        print(f"      距离 < τ 的类对数: {close_pairs}/{len(triu_dists)} (这对 RBF 区分是致命的)")

        # ---- C: ID frame-to-prototype distance distribution ----
        d_correct, d_wrong, margins = frame_distance_distribution(
            model, test_loader, device, prototypes, num_classes
        )
        print(f"\n  [C] 帧-原型距离分布 (ID 测试集, {len(d_correct)} 帧):")
        print(f"      d_correct:         mean={d_correct.mean():.4f}, std={d_correct.std():.4f}, "
              f"median={np.median(d_correct):.4f}")
        print(f"      d_nearest_wrong:   mean={d_wrong.mean():.4f}, std={d_wrong.std():.4f}, "
              f"median={np.median(d_wrong):.4f}")
        print(f"      margin (wrong-correct): mean={margins.mean():.4f}, "
              f"median={np.median(margins):.4f}")
        frac_negative_margin = (margins < 0).mean()
        print(f"      margin<0 的帧比例: {frac_negative_margin:.4f} ({frac_negative_margin*100:.1f}%) "
              f"— 这些帧被归错类")

        # p_correct = exp(-d_correct^2 / (2*tau^2))
        p_correct = np.exp(-d_correct**2 / (2 * tau**2))
        p_wrong_min = np.exp(-d_wrong**2 / (2 * tau**2))
        print(f"      p_correct (RBF):   mean={p_correct.mean():.4f}, min={p_correct.min():.4f}")
        print(f"      p_nearest_wrong:   mean={p_wrong_min.mean():.4f}")

        # ---- D: OOD distance analysis ----
        ood_codes = [c for c in [f"D{i}" for i in range(1, 21)] if c != dataset_code]
        ood_distances_all = []
        ood_names_checked = []
        for ood_code in ood_codes:
            try:
                ood_name, ood_channels, _, _, _, _, _, _, _, _ = get_dataset_config(ood_code)
                if ood_channels != data['in_channels']:
                    continue
                X_ood, y_ood = prepare_ood_data(
                    ood_name, data['global_means'], data['global_stds'],
                    target_len=data['X_train'].shape[2], local_path=LOCAL_UCR_PATH,
                )
                ood_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_ood), torch.from_numpy(y_ood)),
                    batch_size=batch_size, shuffle=False,
                )
                ood_dists = ood_distance_to_nearest(model, ood_loader, device, prototypes)
                ood_distances_all.append(ood_dists)
                ood_names_checked.append(ood_code)
            except Exception:
                continue

        if ood_distances_all:
            all_ood_dists = np.concatenate(ood_distances_all)
            p_ood = np.exp(-all_ood_dists**2 / (2 * tau**2))
            print(f"\n  [D] OOD 帧-原型距离 (合并 {len(ood_names_checked)} 个 OOD 数据集, "
                  f"{len(all_ood_dists)} 帧):")
            print(f"      d_nearest:  mean={all_ood_dists.mean():.4f}, "
                  f"median={np.median(all_ood_dists):.4f}, std={all_ood_dists.std():.4f}")
            print(f"      p_nearest (RBF): mean={p_ood.mean():.4f}")
            print(f"      vs ID d_correct: mean={d_correct.mean():.4f}")

            # 关键对比: OOD 帧到最近原型的距离 vs ID 帧到正确原型的距离
            overlap_ratio = (all_ood_dists < d_correct.mean()).mean()
            print(f"      OOD 帧比 ID 平均 d_correct 更近的比例: {overlap_ratio:.4f} "
                  f"({overlap_ratio*100:.1f}%)")

            # OOD per-dataset breakdown
            print(f"\n      各 OOD 数据集的 mean d_nearest:")
            for ood_name, ood_dists_single in zip(ood_names_checked, ood_distances_all):
                p_single = np.exp(-ood_dists_single**2 / (2 * tau**2))
                print(f"        {ood_name}: d={ood_dists_single.mean():.4f}, "
                      f"p={p_single.mean():.4f}, n_frames={len(ood_dists_single)}")

        # ---- E: Directional analysis (cosine similarity to prototype) ----
        print(f"\n  [E] 方向分析 (帧向量与各类原型的 cosine similarity):")
        with torch.no_grad():
            cos_to_correct = []
            cos_to_nearest_wrong = []
            for X_batch, y_batch in test_loader:
                bb = model.forward_backbone(X_batch.to(device))
                for i in range(len(y_batch)):
                    true_label = y_batch[i].item()
                    frames = bb[i].permute(1, 0)  # (T, K)
                    frames_n = frames / (frames.norm(dim=1, keepdim=True) + 1e-8)
                    proto_n = prototypes / (prototypes.norm(dim=1, keepdim=True) + 1e-8)
                    cos_all = frames_n @ proto_n.T  # (T, num_classes)
                    cos_to_correct.extend(cos_all[:, true_label].cpu().tolist())
                    mask = torch.ones(num_classes, dtype=torch.bool)
                    mask[true_label] = False
                    cos_to_nearest_wrong.extend(cos_all[:, mask].max(dim=1).values.cpu().tolist())
        cos_correct = np.array(cos_to_correct)
        cos_wrong = np.array(cos_to_nearest_wrong)
        print(f"      cos(correct):   mean={cos_correct.mean():.4f}, "
              f"frac>0={np.mean(cos_correct>0):.2%}")
        print(f"      cos(wrong_max): mean={cos_wrong.mean():.4f}, "
              f"frac>0={np.mean(cos_wrong>0):.2%}")

        # Store
        all_results[dataset_code] = {
            'name': name, 'K': num_classes, 'T2': T2, 'tau': tau,
            'per_frame_acc': real_acc, 'z_score': z_score,
            'rand_mean': rand_mean, 'rand_std': rand_std,
            'proto_dists': triu_dists, 'proto_min': triu_dists.min(),
            'd_correct_mean': d_correct.mean(), 'd_wrong_mean': d_wrong.mean(),
            'margin_mean': margins.mean(),
        }

    # ========== Summary ==========
    print(f"\n\n{'='*60}")
    print(f"  汇总对比")
    print(f"{'='*60}")
    print(f"{'DS':<4} {'K':<3} {'T2':<4} {'τ':<8} {'帧ACC':<8} {'z':<6} "
          f"{'proto_min':<10} {'d_correct':<10} {'margin':<10}")
    print("-" * 70)
    for ds in datasets:
        r = all_results[ds]
        print(f"{ds:<4} {r['K']:<3} {r['T2']:<4} {r['tau']:<8.4f} {r['per_frame_acc']:<8.4f} "
              f"{r['z_score']:<6.2f} {r['proto_min']:<10.4f} {r['d_correct_mean']:<10.4f} "
              f"{r['margin_mean']:<10.4f}")

    # Save
    np.savez(_current_dir / "validate_core_results.npz", **all_results)
    print("\n结果已保存到 validate_core_results.npz")


if __name__ == "__main__":
    main()
