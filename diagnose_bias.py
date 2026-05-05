"""Quick check: what did bias learn, what do per-frame activations look like after bias+ReLU."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_current_dir))

from model_v2 import FQN_HEDL_v2
from data_utils import load_and_split_dataset, prepare_ood_data, get_dataset_config, LOCAL_UCR_PATH


def diagnose_bias(dataset_code: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)

    model = FQN_HEDL_v2(
        num_channels=data['in_channels'], num_classes=data['num_classes'],
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'],
    ).to(device)

    ckpt = _current_dir / "saved_models" / f"FQN_HEDL_v2_stage2_digamma_{dataset_code}.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    K = model.K
    tau = model.set_pooling.tau.item()
    bias = model.set_pooling.bias.data.cpu().numpy().squeeze()  # (K,)

    print(f"=== Bias Diagnosis: {dataset_code} (K={K}) ===")
    print(f"τ = {tau:.6f}")
    print(f"bias (per class): {bias}")
    print(f"bias range: [{bias.min():.4f}, {bias.max():.4f}]")

    # ---- ID test ----
    X_id = torch.from_numpy(data['X_test']).to(device)
    with torch.no_grad():
        backbone_out = model.forward_backbone(X_id)  # (N, K, T)
        alpha, evidence = model.forward(X_id)
        vacuity_id = model.compute_vacuity(X_id)

    print(f"\n--- ID (test) ---")
    print(f"Backbone output: min={backbone_out.min():.4f}, max={backbone_out.max():.4f}, "
          f"mean={backbone_out.mean():.4f}, std={backbone_out.std():.4f}")

    # After bias shift, before ReLU
    shifted = backbone_out - model.set_pooling.bias  # (N, K, T)
    print(f"After bias shift (x - bias): min={shifted.min():.4f}, max={shifted.max():.4f}, "
          f"mean={shifted.mean():.4f}, std={shifted.std():.4f}")

    # After ReLU
    p = torch.relu(shifted / abs(tau))
    frac_zero = (p == 0).float().mean().item()
    frac_positive = 1 - frac_zero
    print(f"After ReLU((x-bias)/τ): frac_zero={frac_zero:.4f}, frac_positive={frac_positive:.4f}")
    print(f"  p>0 values: min={p[p>0].min():.6f}, max={p[p>0].max():.4f}, mean={p[p>0].mean():.6f}")

    # Per-frame: how many frames per sample have at least one positive class?
    any_positive = (p > 0).any(dim=1).float().sum(dim=1)  # (N,) count of positive classes per frame
    frames_with_signal = (any_positive > 0).float().mean().item()
    print(f"Frames with ≥1 positive class: {frames_with_signal:.4f}")
    avg_pos_classes_per_frame = any_positive.mean().item()
    print(f"Avg positive classes per frame: {avg_pos_classes_per_frame:.4f} (out of {K})")

    # Evidence stats
    print(f"\nEvidence per set (first 8):")
    for i in range(min(model.N_sets, 8)):
        e = evidence[:, i]
        mask = model.set_pooling.masks[i]
        print(f"  set {i} |s|={mask.sum().item()}: mean={e.mean():.4f}, std={e.std():.4f}, "
              f"frac>0={(e>1e-6).float().mean():.4f}")

    total_evidence = evidence.sum(dim=1)
    print(f"\nTotal evidence: mean={total_evidence.mean():.4f}, std={total_evidence.std():.4f}")
    print(f"Σα: mean={alpha.sum(dim=1).mean():.4f}, std={alpha.sum(dim=1).std():.4f}")
    print(f"Vacuity: mean={vacuity_id.mean():.4f}, std={vacuity_id.std():.4f}, "
          f"min={vacuity_id.min():.4f}, max={vacuity_id.max():.4f}")
    print(f"Theoretical max vacuity (all-zero evidence) = K/W = {K}/{model.W} = {K/model.W:.4f}")

    # ---- Sample-level detail (first 5 ID samples) ----
    print(f"\n--- Per-sample vacuity (first 10 ID) ---")
    for i in range(min(10, len(vacuity_id))):
        print(f"  sample {i}: vacuity={vacuity_id[i]:.6f}, Σα={alpha[i].sum():.6f}")

    # ---- OOD check ----
    ood_codes = [f"D{i}" for i in range(1, 21) if f"D{i}" != dataset_code]
    id_ch = data['in_channels']
    target_len = data['X_train'].shape[2]
    checked = 0
    for ood_code in ood_codes:
        if checked >= 3:
            break
        try:
            ood_name, ood_ch, _, _, _, _, _, _, _, _ = get_dataset_config(ood_code)
            if ood_ch != id_ch:
                continue
            X_ood, _ = prepare_ood_data(ood_name, data['global_means'], data['global_stds'],
                                        target_len=target_len, local_path=LOCAL_UCR_PATH)
            X_ood = torch.from_numpy(X_ood[:min(50, len(X_ood))]).to(device)
            with torch.no_grad():
                vac_ood = model.compute_vacuity(X_ood)
                ood_bb = model.forward_backbone(X_ood)
                ood_p = torch.relu((ood_bb - model.set_pooling.bias) / abs(tau))
                ood_frac_zero = (ood_p == 0).float().mean().item()
                ood_ev, ood_alpha = model.forward(X_ood)
            print(f"\nOOD {ood_code} ({ood_name}):")
            print(f"  vacuity: mean={vac_ood.mean():.4f}, std={vac_ood.std():.4f}")
            print(f"  backbone out: mean={ood_bb.mean():.4f}, std={ood_bb.std():.4f}")
            print(f"  after bias+ReLU: frac_zero={ood_frac_zero:.4f}")
            print(f"  total evidence: mean={ood_ev.sum(dim=1).mean():.4f}")
            checked += 1
        except Exception as e:
            pass

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="D8")
    args = parser.parse_args()
    diagnose_bias(args.dataset)
