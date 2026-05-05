"""D1 vs D5 深度诊断：特征、梯度、vacuity 分离度"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model import FQN_HEDL
from data_utils import (
    load_and_split_dataset, prepare_ood_data,
    get_dataset_config, LOCAL_UCR_PATH,
)
from HEDL_core.losses import get_henn_fc

_ALL_DATASETS = [f"D{i}" for i in range(1, 21)]


def diagnose(dataset_code, device, save_dir):
    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS: {dataset_code}")
    print(f"{'='*60}")

    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    batch_size = data['batch_size']
    num_classes = data['num_classes']
    in_channels = data['in_channels']

    ckpt_path = save_dir / f"FQN_HEDL_stage2_henn_{dataset_code}.pth"
    ckpt1_path = save_dir / f"FQN_HEDL_stage1_{dataset_code}.pth"

    # ---- Dataset stats ----
    print(f"\n  [Dataset] name={name}, ch={in_channels}, classes={num_classes}")
    for split, (X, y) in [("train", (data['X_train'], data['y_train'])),
                            ("val", (data['X_val'], data['y_val'])),
                            ("test", (data['X_test'], data['y_test']))]:
        unique, counts = np.unique(y, return_counts=True)
        cls_str = ", ".join(f"c{k}:{v}" for k, v in zip(unique, counts))
        print(f"    {split}: {X.shape}, labels: {cls_str}")
    print(f"    target_len={data['X_train'].shape[2]}, dim={data['dim']}, depth={data['depth']}, ks={data['ks']}")

    # ---- Load models ----
    model = FQN_HEDL(
        num_channels=in_channels, num_classes=num_classes,
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'], W=2.0,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    if ckpt1_path.exists():
        model_s1 = FQN_HEDL(
            num_channels=in_channels, num_classes=num_classes,
            device=device, dim=data['dim'], depth=data['depth'],
            input_window=data['ks'], input_scale=data['input_scale'],
            hidden_window=data['kprop'], W=2.0,
        ).to(device)
        model_s1.load_state_dict(torch.load(ckpt1_path, map_location=device))
        model_s1.eval()
        has_s1 = True
    else:
        has_s1 = False

    # ---- Extract features & logits on test set ----
    X_test_t = torch.from_numpy(data['X_test']).to(device)
    with torch.no_grad():
        # backbone features (before expander)
        backbone_feats = model.backbone(X_test_t)
        # expanded features
        expander_feats, logits = model(X_test_t)
        # alpha from both methods
        evidence = F.relu(logits)
        alpha_logits = evidence + 2.0 / num_classes
        S_logits = alpha_logits.sum(dim=1)
        vacuity_logits = (num_classes / (S_logits + 1e-12)).cpu().numpy()

        alpha_henn = get_henn_fc(expander_feats, model.fc.weight, 2.0, num_classes, device)
        S_henn = alpha_henn.sum(dim=1)
        vacuity_henn = (num_classes / (S_henn + 1e-12)).cpu().numpy()

        # FC weights
        fc_w = model.fc.weight.data.clone()
        fc_b = model.fc.bias.data.clone()

        # predictions
        preds = logits.argmax(dim=1).cpu().numpy()

    # ---- Feature analysis ----
    backbone_np = backbone_feats.cpu().numpy()
    expander_np = expander_feats.cpu().numpy()

    print(f"\n  [Backbone features] shape={backbone_np.shape}")
    print(f"    mean={backbone_np.mean():.4f}, std={backbone_np.std():.4f}")
    print(f"    min={backbone_np.min():.4f}, max={backbone_np.max():.4f}")
    print(f"    frac negative: {(backbone_np < 0).mean():.3f}")
    print(f"    frac zero: {(backbone_np == 0).mean():.3f}")

    print(f"\n  [Expander features] shape={expander_np.shape}")
    print(f"    mean={expander_np.mean():.4f}, std={expander_np.std():.4f}")
    print(f"    min={expander_np.min():.4f}, max={expander_np.max():.4f}")
    print(f"    frac negative: {(expander_np < 0).mean():.3f}")
    print(f"    frac zero: {(expander_np == 0).mean():.3f}")

    # ---- FC weight analysis ----
    print(f"\n  [FC layer] in={fc_w.shape[1]}, out={fc_w.shape[0]}")
    print(f"    weight: mean={fc_w.mean():.4f}, std={fc_w.std():.4f}")
    print(f"    weight: min={fc_w.min():.4f}, max={fc_w.max():.4f}")
    frac_pos = (fc_w > 0).float().mean().item()
    print(f"    frac weights > 0: {frac_pos:.3f}")
    print(f"    bias: mean={fc_b.mean():.4f}, range=[{fc_b.min():.4f}, {fc_b.max():.4f}]")

    # ---- Vacuity analysis ----
    print(f"\n  [Vacuity - logits] mean={vacuity_logits.mean():.4f}, std={vacuity_logits.std():.4f}, range=[{vacuity_logits.min():.4f}, {vacuity_logits.max():.4f}]")
    print(f"  [Vacuity - henn]   mean={vacuity_henn.mean():.4f}, std={vacuity_henn.std():.4f}, range=[{vacuity_henn.min():.4f}, {vacuity_henn.max():.4f}]")

    # Per-class vacuity
    y_test = data['y_test']
    for c in range(num_classes):
        mask = y_test == c
        if mask.sum() > 0:
            print(f"    class {c} (n={mask.sum()}): vacuity_logits={vacuity_logits[mask].mean():.4f}±{vacuity_logits[mask].std():.4f}")

    # ---- Logits analysis ----
    logits_np = logits.cpu().numpy()
    print(f"\n  [Logits] mean={logits_np.mean():.4f}, std={logits_np.std():.4f}")
    print(f"    max per sample: mean={logits_np.max(axis=1).mean():.4f}, std={logits_np.max(axis=1).std():.4f}")
    print(f"    evidence (relu): mean={evidence.mean():.4f}, std={evidence.std():.4f}")
    frac_neg = (logits_np < 0).mean()
    print(f"    frac logits < 0: {frac_neg:.3f}")
    print(f"    acc (test): {(preds == y_test).mean():.4f}")

    # ---- Compare with Stage 1 ----
    if has_s1:
        with torch.no_grad():
            _, logits_s1 = model_s1(X_test_t)
        preds_s1 = logits_s1.argmax(dim=1).cpu().numpy()
        fc_w_s1 = model_s1.fc.weight.data.clone()
        frac_pos_s1 = (fc_w_s1 > 0).float().mean().item()
        print(f"\n  [Stage 1 comparison]")
        print(f"    test acc (s1): {(preds_s1 == y_test).mean():.4f}")
        print(f"    FC frac >0 (s1): {frac_pos_s1:.3f} → (s2): {frac_pos:.3f}")
        print(f"    FC weight mean (s1): {fc_w_s1.mean():.4f} → (s2): {fc_w.mean():.4f}")
        print(f"    FC weight std (s1): {fc_w_s1.std():.4f} → (s2): {fc_w.std():.4f}")

    # ---- OOD separation (pick 3 representative OODs) ----
    id_channels = in_channels
    target_len = data['X_train'].shape[2]

    print(f"\n  [OOD vacuity separation]")
    print(f"  {'OOD':<28} {'ID vacuity':>12} {'OOD vacuity':>12} {'Separation':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")

    ood_codes = [c for c in _ALL_DATASETS if c != dataset_code]
    for ood_code in ood_codes:
        try:
            ood_name, ood_ch, _, _, _, _, _, _, _, _ = get_dataset_config(ood_code)
            if ood_ch != id_channels:
                continue
            X_ood, y_ood = prepare_ood_data(
                ood_name, data['global_means'], data['global_stds'],
                target_len=target_len, local_path=LOCAL_UCR_PATH,
            )
            X_ood_t = torch.from_numpy(X_ood).to(device)
            with torch.no_grad():
                _, logits_ood = model(X_ood_t)
                evidence_ood = F.relu(logits_ood)
                alpha_ood = evidence_ood + 2.0 / num_classes
                S_ood = alpha_ood.sum(dim=1)
                vacuity_ood = (num_classes / (S_ood + 1e-12)).cpu().numpy()
            sep = vacuity_ood.mean() - vacuity_logits.mean()
            print(f"  {ood_name:<28} {vacuity_logits.mean():>12.4f} {vacuity_ood.mean():>12.4f} {sep:>+12.4f}")
        except Exception as e:
            print(f"  {ood_code:<28} FAILED: {e}")

    # ---- Signal analysis: raw time series complexity ----
    print(f"\n  [Raw signal analysis (test set)]")
    X_test_raw = data['X_test']
    # Variance across time
    var_per_sample = X_test_raw.var(axis=2).mean(axis=1)
    print(f"    per-sample time variance: mean={var_per_sample.mean():.4f}, std={var_per_sample.std():.4f}")

    # Feature collapse: cosine similarity between same-class features
    feats_norm = expander_np / (np.linalg.norm(expander_np, axis=1, keepdims=True) + 1e-12)
    sim_matrix = feats_norm @ feats_norm.T
    same_class_mask = y_test[:, None] == y_test[None, :]
    diff_class_mask = ~same_class_mask
    np.fill_diagonal(same_class_mask, False)
    intra_sim = sim_matrix[same_class_mask].mean() if same_class_mask.sum() > 0 else 0
    inter_sim = sim_matrix[diff_class_mask].mean() if diff_class_mask.sum() > 0 else 0
    print(f"    feature cosine similarity: intra-class={intra_sim:.4f}, inter-class={inter_sim:.4f}")
    print(f"    intra/inter ratio: {intra_sim/max(inter_sim, 1e-8):.2f} (higher=better separation)")

    return {
        'code': dataset_code,
        'name': name,
        'backbone_neg_frac': (backbone_np < 0).mean(),
        'expander_neg_frac': (expander_np < 0).mean(),
        'fc_pos_frac': frac_pos,
        'intra_sim': intra_sim,
        'inter_sim': inter_sim,
        'vacuity_mean': vacuity_logits.mean(),
        'vacuity_std': vacuity_logits.std(),
        'logits_neg_frac': frac_neg,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(__file__).resolve().parent / "saved_models"
    print(f"Device: {device}\n")

    results = {}
    for code in ["D1", "D5"]:
        results[code] = diagnose(code, device, save_dir)

    # Side-by-side comparison
    print(f"\n\n{'='*60}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<35} {'D1 (GOOD)':<20} {'D5 (BAD)':<20}")
    print(f"  {'-'*35} {'-'*20} {'-'*20}")

    keys = [
        ("backbone_neg_frac", "Backbone frac negative"),
        ("expander_neg_frac", "Expander frac negative"),
        ("fc_pos_frac", "FC frac weights > 0"),
        ("intra_sim", "Intra-class cosine sim"),
        ("inter_sim", "Inter-class cosine sim"),
        ("vacuity_mean", "Vacuity mean"),
        ("vacuity_std", "Vacuity std"),
        ("logits_neg_frac", "Logits frac negative"),
    ]

    for k, label in keys:
        v1 = results["D1"][k]
        v5 = results["D5"][k]
        diff = "—"
        if isinstance(v1, float) and isinstance(v5, float):
            diff = f"Δ={abs(v1 - v5):.4f}"
            if abs(v1 - v5) > 0.05:
                diff = f">>> DIFF: {abs(v1 - v5):.4f} <<<"
        print(f"  {label:<35} {v1:<20.4f} {v5:<20.4f} {diff}")
