"""
FQN_HEDL_v2 两阶段训练流水线

Stage 1: CrossEntropy (GAP 池化, 与原始 FQN 完全一致)
Stage 2: EDL loss (集合核池化 + HEDL 投影)
  - Warmup: 冻结 backbone, 训练 self.output + τ, lr=1e-3
  - Joint:   解冻全部, lr=1e-4
"""
import sys
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from model_v2 import FQN_HEDL_v2
from HEDL_core.losses import edl_loss
from data_utils import load_and_split_dataset, compute_class_weights, LOCAL_UCR_PATH


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_stage1(
    dataset_code: str,
    device: torch.device,
    epochs: int = 200,
    save_dir: str = None,
) -> FQN_HEDL_v2:
    """Stage 1: CrossEntropy 分类预训练 (与原始 FQN 一致)."""
    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    num_classes = data['num_classes']
    batch_size = data['batch_size']
    lr_decay = data['lr_decay']

    train_loader = make_loader(data['X_train'], data['y_train'], batch_size, shuffle=True)
    val_loader = make_loader(data['X_val'], data['y_val'], batch_size, shuffle=False)

    print(f"[Stage 1 v2] {name} ({dataset_code}): classes={num_classes}, "
          f"dim={data['dim']}, depth={data['depth']}, ks={data['ks']}")

    model = FQN_HEDL_v2(
        num_channels=data['in_channels'], num_classes=num_classes,
        device=device, dim=data['dim'], depth=data['depth'],
        input_window=data['ks'], input_scale=data['input_scale'],
        hidden_window=data['kprop'],
    ).to(device)

    class_weights = compute_class_weights(data['y_train'], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr = 0.001 if dataset_code in ('D10', 'D12') else 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay, patience=10, threshold=0.001
    )

    best_val_loss = float('inf')
    best_state = None
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model.forward_gap(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model.forward_gap(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_acc = val_acc

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_state)
    ckpt_path = save_dir / f"FQN_HEDL_v2_stage1_{dataset_code}.pth"
    torch.save(best_state, ckpt_path)
    print(f"  Stage 1 saved: {ckpt_path} (best val_acc={best_acc:.4f})")
    return model


def train_stage2_v2(
    model: FQN_HEDL_v2,
    dataset_code: str,
    device: torch.device,
    epochs: int = 50,
    warmup_epochs: int = 15,
    W: float = 2.0,
    save_dir: str = None,
    loss_type: str = 'digamma',
) -> FQN_HEDL_v2:
    """Stage 2: 集合核 + EDL loss 微调.

    Warmup:  冻结 backbone (input + hidden_layers), lr=1e-3, 训练 output + τ
    Joint:   解冻全部, lr=1e-4
    """
    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    num_classes = data['num_classes']
    batch_size = data['batch_size']

    train_loader = make_loader(data['X_train'], data['y_train'], batch_size, shuffle=True)
    val_loader = make_loader(data['X_val'], data['y_val'], batch_size, shuffle=False)

    print(f"[Stage 2 v2] {name} ({dataset_code}): "
          f"warmup={warmup_epochs}ep, joint={epochs-warmup_epochs}ep, "
          f"loss={loss_type}, W={W}, N_sets={model.N_sets}, K={num_classes}")

    # ---- 方案E: 计算类别原型 (硬标签) + L2距离校准 tau ----
    model.eval()
    bb_by_class = {k: [] for k in range(num_classes)}
    all_frames_list = []
    all_labels_list = []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            bb = model.forward_backbone(X_batch.to(device))  # (N, K, T)
            for i in range(len(y_batch)):
                lbl = y_batch[i].item()
                bb_by_class[lbl].append(bb[i].cpu())  # (K, T)
                # Collect frames with their true labels for tau calibration
                frames = bb[i].permute(1, 0).cpu()  # (T, K)
                all_frames_list.append(frames)
                all_labels_list.append(lbl)
    prototypes = torch.zeros(num_classes, num_classes)
    for k in range(num_classes):
        if bb_by_class[k]:
            all_k = torch.cat(bb_by_class[k], dim=1)  # (K, total_T)
            prototypes[k] = all_k.mean(dim=1)  # (K,)
    model.set_pooling.prototypes.copy_(prototypes.to(device))

    # 打印原型及类间距离
    print(f"  [Proto] hard prototypes (per-class frame mean):")
    for k in range(num_classes):
        print(f"    class {k}: {prototypes[k].tolist()}")
    if num_classes > 1:
        pairwise_dists = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                d = (prototypes[i] - prototypes[j]).norm().item()
                pairwise_dists.append(d)
                print(f"    dist(c{i}, c{j}) = {d:.4f}")
        print(f"    mean inter-class dist = {sum(pairwise_dists)/len(pairwise_dists):.4f}")

    # τ 校准: RMS L2 距离 from frame to its class prototype
    all_frames = torch.cat(all_frames_list, dim=0)  # (total_T, K)
    all_frame_labels = torch.cat([torch.full((all_frames_list[i].shape[0],), all_labels_list[i], dtype=torch.long)
                                   for i in range(len(all_labels_list))], dim=0)  # (total_T,)
    proto_lookup = prototypes[all_frame_labels]  # (total_T, K)
    dist_sq = ((all_frames - proto_lookup) ** 2).sum(dim=1)  # (total_T,)
    tau_data = torch.sqrt(dist_sq.mean()).item()
    tau_data = max(tau_data, 0.05)
    model.set_pooling.tau.data.fill_(tau_data)
    model.set_pooling.tau_ref.fill_(tau_data)
    print(f"  [Tau] calibrated tau_data={tau_data:.4f} (RMS L2 dist to class prototype)")

    # ---- Loss function ----
    if loss_type == 'mse':
        def alpha_loss(alpha, y_onehot, ep):
            S = alpha.sum(dim=1, keepdim=True)
            pred = alpha / S
            err = (y_onehot - pred) ** 2
            var = alpha * (S - alpha) / (S * S * (S + 1))
            mse_loss = (err + var).sum(dim=1).mean()
            kl_alpha = (alpha - 1) * (1 - y_onehot) + 1
            anneal = min(1.0, ep / 10)
            kl = anneal * _kl_divergence(kl_alpha, num_classes, device)
            return mse_loss + 0.01 * kl
    else:  # digamma
        lambda_tau = 1e-3
        def alpha_loss(alpha, y_onehot, ep):
            edl = edl_loss(
                torch.digamma, y_onehot, alpha, ep, num_classes, 10, device
            ).mean()
            tau_reg = lambda_tau * (model.set_pooling.tau.abs() - model.set_pooling.tau_ref) ** 2
            return edl + tau_reg

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    # ---- Phase 1: Warmup ----
    if warmup_epochs > 0:
        # 冻结 backbone
        for name, param in model.named_parameters():
            if name.startswith('input.') or name.startswith('hidden_layers.'):
                param.requires_grad = False
            else:
                param.requires_grad = True

        warmup_params = [p for p in model.parameters() if p.requires_grad]
        warmup_opt = torch.optim.AdamW(warmup_params, lr=1e-3, weight_decay=1e-4)
        print(f"  [Warmup] frozen: input + hidden_layers, "
              f"trainable params: {sum(p.numel() for p in warmup_params)}")

        for epoch in range(1, warmup_epochs + 1):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
                y_onehot.scatter_(1, y_batch.unsqueeze(1).to(device), 1)

                warmup_opt.zero_grad()
                alpha, _ = model(X_batch)
                loss = alpha_loss(alpha, y_onehot, epoch)
                loss.backward()
                warmup_opt.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
                    y_onehot.scatter_(1, y_batch.unsqueeze(1).to(device), 1)
                    alpha, _ = model(X_batch)
                    loss = alpha_loss(alpha, y_onehot, epoch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                tau_val = model.set_pooling.tau.item()
                print(f"  [Warmup {epoch:3d}/{warmup_epochs}]  "
                      f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  τ={tau_val:.4f}")

    # ---- Phase 2: Joint fine-tuning ----
    for param in model.parameters():
        param.requires_grad = True
    joint_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"  [Joint] all params trainable, lr=1e-4")
    patience_counter = 0  # reset

    for epoch in range(warmup_epochs + 1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
            y_onehot.scatter_(1, y_batch.unsqueeze(1).to(device), 1)

            joint_opt.zero_grad()
            alpha, _ = model(X_batch)
            loss = alpha_loss(alpha, y_onehot, epoch)
            loss.backward()
            joint_opt.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
                y_onehot.scatter_(1, y_batch.unsqueeze(1).to(device), 1)
                alpha, _ = model(X_batch)
                loss = alpha_loss(alpha, y_onehot, epoch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == warmup_epochs + 1:
            tau_val = model.set_pooling.tau.item()
            print(f"  [Joint {epoch:3d}/{epochs}]  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  τ={tau_val:.4f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    ckpt_path = save_dir / f"FQN_HEDL_v2_stage2_{loss_type}_{dataset_code}.pth"
    torch.save(best_state, ckpt_path)
    print(f"  Stage 2 saved: {ckpt_path}")
    print(f"  Final τ={model.set_pooling.tau.item():.4f}")
    return model


def _kl_divergence(alpha, num_classes, device):
    """KL 散度正则项 (MSE loss 用)."""
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S = alpha.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    dalpha = (alpha - 1) * (1 - (alpha - 1))
    kl += (dalpha * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1, keepdim=True)
    return kl.mean()


def train_full_pipeline_v2(
    dataset_code: str,
    device: torch.device = None,
    stage1_epochs: int = 200,
    stage2_epochs: int = 50,
    warmup_epochs: int = 15,
    W: float = 2.0,
    tau_init: float = 0.5,
    save_dir: str = None,
    skip_stage1_if_exists: bool = True,
    loss_type: str = 'digamma',
) -> FQN_HEDL_v2:
    """完整 Stage 1 + Stage 2 训练流水线."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt2_path = save_dir / f"FQN_HEDL_v2_stage2_{loss_type}_{dataset_code}.pth"

    if skip_stage1_if_exists and ckpt2_path.exists():
        print(f"[Skip] Stage 2 checkpoint exists: {ckpt2_path}")
        data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
        model = FQN_HEDL_v2(
            num_channels=data['in_channels'], num_classes=data['num_classes'],
            device=device, dim=data['dim'], depth=data['depth'],
            input_window=data['ks'], input_scale=data['input_scale'],
            hidden_window=data['kprop'], W=W, tau_init=tau_init,
        ).to(device)
        state = torch.load(ckpt2_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [Info] Missing keys: {missing}")
        return model

    ckpt1_path = save_dir / f"FQN_HEDL_v2_stage1_{dataset_code}.pth"
    if skip_stage1_if_exists and ckpt1_path.exists():
        print(f"[Skip] Stage 1 checkpoint exists: {ckpt1_path}")
        data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
        model = FQN_HEDL_v2(
            num_channels=data['in_channels'], num_classes=data['num_classes'],
            device=device, dim=data['dim'], depth=data['depth'],
            input_window=data['ks'], input_scale=data['input_scale'],
            hidden_window=data['kprop'], W=W, tau_init=tau_init,
        ).to(device)
        state = torch.load(ckpt1_path, map_location=device)
        # 过滤 set_pooling/hedL_proj keys: Stage 2 会重建, 避免结构变化导致 shape mismatch
        state = {k: v for k, v in state.items()
                 if not k.startswith('set_pooling.') and not k.startswith('hedL_proj.')}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [Info] Missing keys (will use init defaults): {missing}")
        if unexpected:
            print(f"  [Warn] Unexpected keys (ignored): {unexpected}")
        # Reset tau: old checkpoints may have ReLU-era tau (~3.0), RBF needs ~0.5
        model.set_pooling.tau.data.fill_(tau_init)
        model.set_pooling.tau_ref.fill_(tau_init)
        print(f"  [Tau] reset to {tau_init}")
    else:
        model = train_stage1(
            dataset_code=dataset_code, device=device,
            epochs=stage1_epochs, save_dir=str(save_dir),
        )

    model = train_stage2_v2(
        model=model, dataset_code=dataset_code, device=device,
        epochs=stage2_epochs, warmup_epochs=warmup_epochs,
        W=W, save_dir=str(save_dir), loss_type=loss_type,
    )
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_full_pipeline_v2("D1", device=device, stage1_epochs=200, stage2_epochs=50)
    print("Done.")
