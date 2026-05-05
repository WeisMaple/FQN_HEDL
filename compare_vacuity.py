"""对比 method='henn' vs method='logits' 的 OOD 检测性能"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model import FQN_HEDL
from eval_ood import compute_ood_metrics, collect_vacuity
from data_utils import (
    load_and_split_dataset, prepare_ood_data,
    get_dataset_config, LOCAL_UCR_PATH,
)

_ALL_DATASETS = [f"D{i}" for i in range(1, 21)]
_TARGETS = ["D1", "D3", "D5"]


def collect_vacuity_method(model, loader, device, method):
    model.eval()
    vacuity_list = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            vac = model.compute_vacuity(X_batch, method=method)
            vacuity_list.append(vac.cpu().numpy())
    return np.concatenate(vacuity_list, axis=0)


def compare_one_dataset(dataset_code, device, save_dir):
    print(f"\n{'='*80}")
    print(f"  ID: {dataset_code}")
    print(f"{'='*80}")

    # Load model
    ckpt_path = save_dir / f"FQN_HEDL_stage2_henn_{dataset_code}.pth"
    if not ckpt_path.exists():
        print(f"  SKIP: no checkpoint at {ckpt_path}")
        return

    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    batch_size = data['batch_size']

    model = FQN_HEDL(
        num_channels=data['in_channels'], num_classes=data['num_classes'],
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'], W=2.0,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ID loader
    id_loader = DataLoader(
        TensorDataset(torch.from_numpy(data['X_test']),
                      torch.from_numpy(data['y_test'])),
        batch_size=batch_size, shuffle=False,
    )

    id_henn = collect_vacuity_method(model, id_loader, device, 'henn')
    id_logits = collect_vacuity_method(model, id_loader, device, 'logits')

    target_len = data['X_train'].shape[2]
    id_channels = data['in_channels']

    ood_codes = [c for c in _ALL_DATASETS if c != dataset_code]

    print(f"  {'OOD':<6} {'Name':<28} {'HENN FPR95%':>11} {'Logits FPR95%':>13} | {'HENN AUROC':>10} {'Logits AUROC':>12} | {'HENN AUPRC':>10} {'Logits AUPRC':>12}")
    print(f"  {'-'*6} {'-'*28} {'-'*11} {'-'*13} | {'-'*10} {'-'*12} | {'-'*10} {'-'*12}")

    results = {'henn': {'fpr95': [], 'auroc': [], 'auprc': []},
               'logits': {'fpr95': [], 'auroc': [], 'auprc': []}}

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
                TensorDataset(torch.from_numpy(X_ood),
                              torch.from_numpy(y_ood)),
                batch_size=batch_size, shuffle=False,
            )

            ood_henn = collect_vacuity_method(model, ood_loader, device, 'henn')
            ood_logits = collect_vacuity_method(model, ood_loader, device, 'logits')

            h_fpr, h_auroc, h_auprc = compute_ood_metrics(id_henn, ood_henn)
            l_fpr, l_auroc, l_auprc = compute_ood_metrics(id_logits, ood_logits)

            results['henn']['fpr95'].append(h_fpr * 100)
            results['henn']['auroc'].append(h_auroc)
            results['henn']['auprc'].append(h_auprc)
            results['logits']['fpr95'].append(l_fpr * 100)
            results['logits']['auroc'].append(l_auroc)
            results['logits']['auprc'].append(l_auprc)

            # Mark which is better
            better = "HENN" if h_auroc > l_auroc else "Logits"
            print(f"  {ood_code:<6} {ood_name:<28} {h_fpr*100:>10.2f}% {l_fpr*100:>12.2f}% | {h_auroc:>10.4f} {l_auroc:>12.4f} | {h_auprc:>10.4f} {l_auprc:>12.4f}  ({better})")
        except Exception as e:
            print(f"  {ood_code:<6} {'FAILED':<28} — {e}")

    # Summary
    print(f"  {'-'*6} {'-'*28} {'-'*11} {'-'*13} | {'-'*10} {'-'*12} | {'-'*10} {'-'*12}")
    for method in ['henn', 'logits']:
        r = results[method]
        if r['fpr95']:
            print(f"  {method.upper():<6} {'AVG':<28} {np.mean(r['fpr95']):>10.2f}%           | {np.mean(r['auroc']):>10.4f}            | {np.mean(r['auprc']):>10.4f}")

    # Count wins
    n = len(results['henn']['auroc'])
    henn_wins = sum(1 for i in range(n) if results['henn']['auroc'][i] > results['logits']['auroc'][i])
    logits_wins = n - henn_wins
    print(f"\n  Winner count (by AUROC): HENN={henn_wins}, Logits={logits_wins}")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = _current_dir / "saved_models"
    print(f"Device: {device}")
    print(f"Comparing method='henn' vs method='logits' on HENN-trained models\n")

    for code in _TARGETS:
        compare_one_dataset(code, device, save_dir)

    print(f"\n{'='*80}")
    print("Done.")
