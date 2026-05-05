"""
FQN_HEDL_v2 OOD Detection 评估

使用 vacuity = K/Σα 作为 OOD score。
ID 样本应有低 vacuity，OOD 样本应有高 vacuity。

用法:
    python eval_ood_v2.py --dataset D1
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

from model_v2 import FQN_HEDL_v2
from data_utils import load_and_split_dataset, prepare_ood_data, get_dataset_config, LOCAL_UCR_PATH

_ALL_DATASETS = [f"D{i}" for i in range(1, 21)]


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray):
    """FPR95 (fraction), AUROC, AUPRC. Positive=OOD(high score)."""
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


def collect_vacuity_v2(model, loader, device):
    """Compute vacuity = K / Sigma(alpha) for each sample."""
    model.eval()
    vacuity_list = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            vac = model.compute_vacuity(X_batch)
            vacuity_list.append(vac.cpu().numpy())
    return np.concatenate(vacuity_list, axis=0)


def evaluate_ood_v2(
    dataset_code: str = "D1",
    device: torch.device = None,
    save_dir: str = None,
    W: float = 2.0,
    tau_init: float = 0.5,
    loss_type: str = 'digamma',
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)

    # ---- Load ID data ----
    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    num_classes = data['num_classes']
    batch_size = data['batch_size']
    id_test_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['X_test']), torch.from_numpy(data['y_test'])),
        batch_size=batch_size, shuffle=False,
    )
    target_len = data['X_train'].shape[2]

    print(f"FQN_HEDL_v2 OOD Evaluation - ID: {name} ({dataset_code})")
    print(f"  K={num_classes}, dim={data['dim']}, depth={data['depth']}, "
          f"W={W}, tau_init={tau_init}")

    # ---- Load model ----
    model = FQN_HEDL_v2(
        num_channels=data['in_channels'], num_classes=num_classes,
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'], W=W, tau_init=tau_init,
    ).to(device)

    ckpt_path = save_dir / f"FQN_HEDL_v2_stage2_{loss_type}_{dataset_code}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run Stage 2 training first."
        )
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [Info] Missing keys: {missing}")
    if unexpected:
        print(f"  [Warn] Unexpected keys: {unexpected}")
    model.eval()
    print(f"  Loaded: {ckpt_path}")
    print(f"  tau={model.set_pooling.tau.item():.4f}")

    # ---- ID vacuity ----
    print(f"\nComputing ID vacuity...")
    id_scores = collect_vacuity_v2(model, id_test_loader, device)
    print(f"  ID samples: {len(id_scores)}, "
          f"mean={id_scores.mean():.6f}, std={id_scores.std():.6f}, "
          f"min={id_scores.min():.6f}, max={id_scores.max():.6f}")

    # ---- OOD evaluation ----
    ood_codes = [c for c in _ALL_DATASETS if c != dataset_code]
    id_channels = data['in_channels']

    print(f"\n{'OOD':<6} {'Name':<28} {'FPR95%':>8} {'AUROC%':>8} {'AUPRC%':>8}")
    print("-" * 70)

    all_results = {}
    for ood_code in ood_codes:
        try:
            ood_name, ood_channels, _, _, _, _, _, _, _, _ = get_dataset_config(ood_code)
            if ood_channels != id_channels:
                continue
            X_ood, y_ood = prepare_ood_data(
                ood_name, data['global_means'], data['global_stds'],
                target_len=target_len, local_path=LOCAL_UCR_PATH,
            )
            ood_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_ood), torch.from_numpy(y_ood)),
                batch_size=batch_size, shuffle=False,
            )
            ood_scores = collect_vacuity_v2(model, ood_loader, device)
            fpr95, auroc, auprc = compute_ood_metrics(id_scores, ood_scores)
            fpr95_pct = fpr95 * 100
            all_results[ood_code] = {"fpr95": fpr95_pct, "auroc": auroc, "auprc": auprc}
            print(f"{ood_code:<6} {ood_name:<28} {fpr95_pct:>8.2f} {auroc*100:>8.2f} {auprc*100:>8.2f}")
        except Exception as e:
            print(f"{ood_code:<6} {'ERROR':<28} {str(e)[:50]}")

    # ---- Summary ----
    if all_results:
        fpr95s = [r["fpr95"] for r in all_results.values()]
        aurocs = [r["auroc"] for r in all_results.values()]
        auprcs = [r["auprc"] for r in all_results.values()]
        print("-" * 70)
        print(f"{'AVG':<6} {'':<28} {np.mean(fpr95s):>8.2f} {np.mean(aurocs)*100:>8.2f} {np.mean(auprcs)*100:>8.2f}")
        print(f"{'STD':<6} {'':<28} {np.std(fpr95s):>8.2f} {np.std(aurocs)*100:>8.2f} {np.std(auprcs)*100:>8.2f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FQN_HEDL_v2 OOD Evaluation")
    parser.add_argument("--dataset", type=str, default="D1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--W", type=float, default=2.0)
    parser.add_argument("--tau_init", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="digamma")
    args = parser.parse_args()
    evaluate_ood_v2(
        dataset_code=args.dataset, device=args.device,
        save_dir=args.save_dir, W=args.W, tau_init=args.tau_init,
        loss_type=args.loss_type,
    )
