"""
FQN+HEDL 全量 OOD Benchmark — 20 UCR 数据集 D1-D20

Protocol（与 HEDL / REEDL / RSNN 一致）:
  - 60/20/20 split, random_state=42
  - OOD score: vacuity = W/Σα (越高越 OOD)
  - 指标: FPR95(%)↓, AUROC↑, AUPRC↑

用法:
    python -m FQN_HEDL.run_benchmark [--device cuda] [--stage1_epochs 200] [--stage2_epochs 50] [--resume]
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model import FQN_HEDL
from train import train_full_pipeline
from eval_ood import compute_ood_metrics, collect_vacuity
from data_utils import (
    load_and_split_dataset, prepare_ood_data,
    get_dataset_config, LOCAL_UCR_PATH,
)

_ALL_DATASETS = [f"D{i}" for i in range(1, 21)]


def run_benchmark(
    device: str = "cuda",
    stage1_epochs: int = 200,
    stage2_epochs: int = 50,
    resume: bool = False,
    W: float = 2.0,
    loss_type: str = 'henn',
    skip_datasets: list = None,
    only_datasets: list = None,
):
    if skip_datasets is None:
        skip_datasets = []
    skip_set = set(skip_datasets)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    vacuity_method = 'henn' if loss_type == 'henn' else 'logits'
    print(f"FQN+HEDL OOD Benchmark")
    print(f"  Device: {device}, Stage1 epochs: {stage1_epochs}, "
          f"Stage2 epochs: {stage2_epochs}, W: {W}, loss: {loss_type}, vacuity: {vacuity_method}")
    print(f"  Resume: {resume}")
    if skip_set:
        print(f"  Skip: {sorted(skip_set)}")
    if only_datasets:
        print(f"  Only: {only_datasets}")

    save_dir = _current_dir / "saved_models"
    save_dir.mkdir(exist_ok=True)

    all_id_results = {}
    datasets = only_datasets if only_datasets else _ALL_DATASETS
    datasets = [d for d in datasets if d not in skip_set]
    total = len(datasets)

    for idx, id_code in enumerate(datasets):
        print(f"\n{'='*70}")
        print(f"[{idx+1}/{total}] ID Dataset: {id_code}")
        print(f"{'='*70}")

        try:
            # ---- Train ----
            t0 = time.time()
            model = train_full_pipeline(
                dataset_code=id_code, device=device,
                stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs,
                W=W, save_dir=str(save_dir),
                skip_stage1_if_exists=resume,
                loss_type=loss_type,
            )
            print(f"  Training time: {time.time() - t0:.0f}s")

            # ---- Load ID data ----
            data = load_and_split_dataset(id_code, local_path=LOCAL_UCR_PATH)
            name = data['name']
            batch_size = data['batch_size']
            target_len = data['X_train'].shape[2]

            id_loader = DataLoader(
                TensorDataset(torch.from_numpy(data['X_test']),
                              torch.from_numpy(data['y_test'])),
                batch_size=batch_size, shuffle=False,
            )

            # ---- ID vacuity ----
            id_scores = collect_vacuity(model, id_loader, device, method=vacuity_method)
            print(f"  ID samples: {len(id_scores)}, "
                  f"mean vacuity: {id_scores.mean():.6f} ± {id_scores.std():.6f}")

            # ---- OOD evaluation ----
            ood_results = {}
            ood_codes = [c for c in _ALL_DATASETS if c != id_code]
            id_channels = data['in_channels']

            n_ood_available = sum(1 for c in ood_codes if get_dataset_config(c)[1] == id_channels)
            print(f"\n  OOD pairs: {n_ood_available} (same-channel only, ch={id_channels})")
            print(f"  {'OOD':<6} {'Name':<28} {'FPR95%↓':>8} {'AUROC%↑':>9} {'AUPRC%↑':>9}")
            print(f"  {'-'*65}")

            for ood_code in ood_codes:
                try:
                    ood_name, ood_channels, _, _, _, _, _, _, _, _ = get_dataset_config(ood_code)
                    if ood_channels != id_channels:
                        print(f"  {ood_code:<6} {ood_name:<28} {'SKIP':>8} (ch mismatch)")
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
                    ood_scores = collect_vacuity(model, ood_loader, device, method=vacuity_method)
                    fpr95, auroc, auprc = compute_ood_metrics(id_scores, ood_scores)
                    fpr95_pct = fpr95 * 100
                    ood_results[ood_code] = {
                        "fpr95": fpr95_pct, "auroc": auroc, "auprc": auprc,
                    }
                    print(f"  {ood_code:<6} {ood_name:<28} {fpr95_pct:>8.2f} {auroc*100:>9.2f} {auprc*100:>9.2f}")
                except Exception as e:
                    print(f"  {ood_code:<6} {'FAILED':<28} — {e}")

            if ood_results:
                avg_fpr95 = np.mean([r["fpr95"] for r in ood_results.values()])
                avg_auroc = np.mean([r["auroc"] for r in ood_results.values()])
                avg_auprc = np.mean([r["auprc"] for r in ood_results.values()])
                print(f"  {'-'*65}")
                print(f"  {'AVG':<6} {'':<28} {avg_fpr95:>8.2f} {avg_auroc*100:>9.2f} {avg_auprc*100:>9.2f}")
                all_id_results[id_code] = {
                    "name": name,
                    "avg_fpr95": avg_fpr95,
                    "avg_auroc": avg_auroc,
                    "avg_auprc": avg_auprc,
                    "per_ood": ood_results,
                }

        except Exception as e:
            print(f"  FAILED for {id_code}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Final summary ----
    print(f"\n\n{'='*80}")
    print(f"FQN+HEDL FINAL SUMMARY (60/20/20 split, W={W})")
    print(f"{'='*80}")
    print(f"{'ID':<6} {'Name':<28} {'FPR95%↓':>10} {'AUROC%↑':>10} {'AUPRC%↑':>10}  {'Pairs':>5}")
    print(f"{'-'*74}")

    fpr95s, aurocs, auprcs = [], [], []
    for id_code in _ALL_DATASETS:
        if id_code in all_id_results:
            r = all_id_results[id_code]
            fpr95s.append(r["avg_fpr95"])
            aurocs.append(100 * r["avg_auroc"])
            auprcs.append(100 * r["avg_auprc"])
            n_pairs = len(r.get("per_ood", {}))
            print(f"{id_code:<6} {r['name']:<28} {r['avg_fpr95']:>9.2f}% {100*r['avg_auroc']:>9.2f}% {100*r['avg_auprc']:>9.2f}%  {n_pairs:>5}")

    if fpr95s:
        print(f"{'-'*74}")
        print(f"{'AVG':<6} {'':<28} {np.mean(fpr95s):>9.2f}% {np.mean(aurocs):>9.2f}% {np.mean(auprcs):>9.2f}%")

    # ---- Save ----
    results_path = save_dir / "benchmark_results_fqn_hedl.json"
    with open(results_path, 'w') as f:
        json.dump(all_id_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\nResults saved to {results_path}")

    return all_id_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FQN+HEDL OOD Benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stage1_epochs", type=int, default=200)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--W", type=float, default=2.0)
    parser.add_argument("--loss_type", type=str, default="henn",
                        choices=["henn", "digamma", "mse", "log"])
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Datasets to skip, e.g. --skip D11 D13")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Only run these datasets, e.g. --only D1 D3 D5")
    args = parser.parse_args()
    run_benchmark(
        device=args.device,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        resume=args.resume,
        W=args.W,
        loss_type=args.loss_type,
        skip_datasets=args.skip if args.skip else None,
        only_datasets=args.only,
    )
