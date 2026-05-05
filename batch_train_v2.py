"""Batch train & eval all D1-D13 datasets with RBF Scheme E."""
import json
import torch
import numpy as np
from pathlib import Path
from train_v2 import train_full_pipeline_v2
from eval_ood_v2 import evaluate_ood_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = Path(__file__).resolve().parent / "saved_models"

datasets = [f"D{i}" for i in range(1, 14) if i != 11]  # D1-D13, skip D11 (too slow)

all_results = {}
for ds in datasets:
    print(f"\n{'='*60}")
    print(f"Training {ds}")
    print(f"{'='*60}")
    try:
        # Delete old Stage 2 checkpoint (from ReLU era)
        old_ckpt = save_dir / f"FQN_HEDL_v2_stage2_digamma_{ds}.pth"
        if old_ckpt.exists():
            old_ckpt.unlink()
            print(f"Deleted old Stage 2 checkpoint: {old_ckpt}")

        model = train_full_pipeline_v2(
            ds, device=device, stage1_epochs=200, stage2_epochs=50,
            skip_stage1_if_exists=True,
        )
        print(f"\n--- OOD Evaluation: {ds} ---")
        results = evaluate_ood_v2(ds, device=device)
        all_results[ds] = {
            "fpr95_mean": float(np.mean([r["fpr95"] for r in results.values()])),
            "auroc_mean": float(np.mean([r["auroc"] for r in results.values()])),
            "auprc_mean": float(np.mean([r["auprc"] for r in results.values()])),
            "details": {k: {"fpr95": v["fpr95"], "auroc": v["auroc"], "auprc": v["auprc"]}
                       for k, v in results.items()},
        }
    except Exception as e:
        print(f"ERROR for {ds}: {e}")
        all_results[ds] = {"error": str(e)}

# Summary
print(f"\n{'='*60}")
print("BENCHMARK SUMMARY (Scheme E: RBF Prototype)")
print(f"{'='*60}")
print(f"{'ID':<6} {'FPR95%':>8} {'AUROC%':>8} {'AUPRC%':>8}")
print("-" * 40)
fpr95s, aurocs, auprcs = [], [], []
for ds in datasets:
    r = all_results.get(ds, {})
    if "fpr95_mean" in r:
        fpr95s.append(r["fpr95_mean"])
        aurocs.append(r["auroc_mean"])
        auprcs.append(r["auprc_mean"])
        print(f"{ds:<6} {r['fpr95_mean']:>8.2f} {r['auroc_mean']*100:>8.2f} {r['auprc_mean']*100:>8.2f}")
    elif "error" in r:
        print(f"{ds:<6} {'ERROR':>8}")
print("-" * 40)
if fpr95s:
    print(f"{'AVG':<6} {np.mean(fpr95s):>8.2f} {np.mean(aurocs)*100:>8.2f} {np.mean(auprcs)*100:>8.2f}")

# Save
result_path = save_dir / "benchmark_results_v2_rbf.json"
with open(result_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved: {result_path}")
