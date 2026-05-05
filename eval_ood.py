"""
FQN+HEDL OOD Detection 评估

使用 vacuity = W/Σα 作为 OOD score。
ID 样本应有低 vacuity，OOD 样本应有高 vacuity。

指标: FPR95(%)↓, AUROC↑, AUPRC↑

用法:
    python -m FQN_HEDL.eval_ood --dataset D3 [--device cuda]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model import FQN_HEDL
from data_utils import load_and_split_dataset, prepare_ood_data, get_dataset_config, LOCAL_UCR_PATH

_ALL_DATASETS = [f"D{i}" for i in range(1, 21)]


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
    """
    计算 OOD 检测指标。

    id_scores: ID 样本的 vacuity（应较低）
    ood_scores: OOD 样本的 vacuity（应较高）

    "Positive" = OOD（高 score → 被预测为 OOD）
    "Negative" = ID  （低 score → 被预测为 ID）

    返回 fpr95 (fraction 0-1), auroc, auprc
    """
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([
        np.zeros(len(id_scores), dtype=int),
        np.ones(len(ood_scores), dtype=int),
    ])

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]

    n_pos = int(np.sum(labels))
    n_neg = int(len(labels) - n_pos)

    tp, fp = 0, 0
    tpr_list, fpr_list = [0.0], [0.0]
    precision_list, recall_list = [1.0], [0.0]

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)
        precision_list.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recall_list.append(tp / n_pos if n_pos > 0 else 0)

    tpr = np.array(tpr_list, dtype=np.float64)
    fpr = np.array(fpr_list, dtype=np.float64)

    # FPR95: FPR at TPR=0.95
    fpr95 = 1.0
    if n_pos > 0:
        for i in range(len(tpr) - 1):
            if tpr[i] <= 0.95 <= tpr[i + 1]:
                if tpr[i + 1] - tpr[i] > 1e-12:
                    alpha = (0.95 - tpr[i]) / (tpr[i + 1] - tpr[i])
                    fpr95 = float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))
                else:
                    fpr95 = float(fpr[i])
                break

    auroc = float(np.trapz(tpr, fpr))

    prec_arr = np.array(precision_list, dtype=np.float64)
    rec_arr = np.array(recall_list, dtype=np.float64)
    for i in range(len(prec_arr) - 1, 0, -1):
        prec_arr[i - 1] = max(prec_arr[i - 1], prec_arr[i])
    auprc = float(np.trapz(prec_arr, rec_arr))

    return fpr95, auroc, auprc


def collect_vacuity(model: FQN_HEDL, loader: DataLoader,
                    device: torch.device,
                    method: str = 'logits') -> np.ndarray:
    """对 DataLoader 中的数据计算 vacuity

    method='logits': 标准 EDL, evidence=relu(logits), alpha=evidence+W/K
    method='henn':   HENN 正权重 masking, 与 edl_HENN loss 训练一致
    """
    model.eval()
    vacuity_list = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            vac = model.compute_vacuity(X_batch, method=method)
            vacuity_list.append(vac.cpu().numpy())
    return np.concatenate(vacuity_list, axis=0)


def evaluate_ood(
    dataset_code: str = "D3",
    device: torch.device = None,
    save_dir: str = None,
    W: float = 2.0,
    loss_type: str = 'digamma',
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)

    # HENN loss 训练的模型，OOD 评估也用 HENN 归一化
    vacuity_method = 'henn' if loss_type == 'henn' else 'logits'

    # ---- 加载 ID 数据 ----
    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    num_classes = data['num_classes']
    batch_size = data['batch_size']
    id_test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['X_test']), torch.from_numpy(data['y_test'])),
        batch_size=batch_size, shuffle=False,
    )
    target_len = data['X_train'].shape[2]

    print(f"FQN+HEDL OOD Evaluation — ID: {name} ({dataset_code}), loss={loss_type}, vacuity={vacuity_method}")
    print(f"  Classes: {num_classes}, Feature dim: {data['dim']}, "
          f"Time length: {target_len}")

    # ---- 加载模型 ----
    model = FQN_HEDL(
        num_channels=data['in_channels'], num_classes=num_classes,
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'], W=W,
    ).to(device)

    ckpt_path = save_dir / f"FQN_HEDL_stage2_{loss_type}_{dataset_code}.pth"
    if not ckpt_path.exists():
        # fallback: old naming convention without loss_type
        ckpt_path = save_dir / f"FQN_HEDL_stage2_{dataset_code}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt_path}. "
            f"Run training on {dataset_code} first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ---- ID vacuity ----
    print(f"\nComputing ID vacuity for {name} ({dataset_code})...")
    id_scores = collect_vacuity(model, id_test_loader, device, method=vacuity_method)
    print(f"  ID samples: {len(id_scores)}, "
          f"mean vacuity: {id_scores.mean():.6f} ± {id_scores.std():.6f}")

    # ---- OOD evaluation ----
    ood_codes = [c for c in _ALL_DATASETS if c != dataset_code]
    print(f"\nOOD datasets: {ood_codes}")
    print(f"{'OOD':<6} {'Name':<28} {'FPR95%↓':>8} {'AUROC%↑':>8} {'AUPRC%↑':>8}")
    print("-" * 70)

    id_channels = data['in_channels']
    all_results = {}
    for ood_code in ood_codes:
        try:
            ood_name, ood_channels, _, _, _, _, _, _, _, _ = get_dataset_config(ood_code)
            if ood_channels != id_channels:
                print(f"{ood_code:<6} {ood_name:<28} {'SKIP':>8} (ch mismatch)")
                continue
            X_ood, y_ood = prepare_ood_data(
                ood_name, data['global_means'], data['global_stds'],
                target_len=target_len, local_path=LOCAL_UCR_PATH,
            )
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_ood), torch.from_numpy(y_ood)),
                batch_size=batch_size, shuffle=False,
            )
            ood_scores = collect_vacuity(model, ood_loader, device, method=vacuity_method)
            fpr95, auroc, auprc = compute_ood_metrics(id_scores, ood_scores)
            fpr95_pct = fpr95 * 100
            all_results[ood_code] = {
                "fpr95": fpr95_pct, "auroc": auroc, "auprc": auprc
            }
            print(f"{ood_code:<6} {ood_name:<28} {fpr95_pct:>8.2f} {auroc*100:>8.2f} {auprc*100:>8.2f}")
        except Exception as e:
            print(f"{ood_code:<6} {'FAILED':<28} — {e}")

    # ---- Summary ----
    if all_results:
        fpr95s = [r["fpr95"] for r in all_results.values()]
        aurocs = [r["auroc"] for r in all_results.values()]
        auprcs = [r["auprc"] for r in all_results.values()]
        print("-" * 70)
        print(f"{'AVG':<6} {'':<28} {np.mean(fpr95s):>8.2f} {np.mean(aurocs)*100:>8.2f} {np.mean(auprcs)*100:>8.2f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FQN+HEDL OOD Evaluation")
    parser.add_argument("--dataset", type=str, default="D3")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--W", type=float, default=2.0)
    parser.add_argument("--loss_type", type=str, default="digamma",
                        choices=["henn", "digamma", "mse", "log"])
    args = parser.parse_args()
    evaluate_ood(
        dataset_code=args.dataset, device=args.device,
        save_dir=args.save_dir, W=args.W, loss_type=args.loss_type,
    )
