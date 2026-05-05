"""
FQN+HEDL 两阶段训练流水线

Stage 1: CrossEntropyLoss 分类预训练（标准 FQN 训练方式）
Stage 2: HENN loss 证据深度学习微调（HEDL 的不确定性量化）

数据加载使用 data_utils.py（60/20/20 split, random_state=42）
"""
import os
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

from model import FQN_HEDL
from HEDL_core.losses import edl_HENN, edl_digamma_loss, edl_mse_loss, edl_log_loss
from data_utils import load_and_split_dataset, compute_class_weights, LOCAL_UCR_PATH


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = True, drop_last: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_stage1(
    dataset_code: str,
    device: torch.device,
    epochs: int = 200,
    save_dir: str = None,
) -> FQN_HEDL:
    """
    Stage 1: CrossEntropy 分类预训练。
    使用 FQN 原始训练配置（Adam + ReduceLROnPlateau + 类别权重）。
    """
    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
    name = data['name']
    num_classes = data['num_classes']
    batch_size = data['batch_size']
    lr_decay = data['lr_decay']
    dim = data['dim']
    depth = data['depth']
    input_window = data['ks']
    input_scale = data['input_scale']
    hidden_window = data['kprop']

    train_loader = make_loader(data['X_train'], data['y_train'], batch_size, shuffle=True)
    val_loader = make_loader(data['X_val'], data['y_val'], batch_size, shuffle=False)

    print(f"[Stage 1] {name} ({dataset_code}): classes={num_classes}, "
          f"dim={dim}, depth={depth}, ks={input_window}")

    model = FQN_HEDL(
        num_channels=data['in_channels'], num_classes=num_classes,
        device=device, dim=dim, depth=depth,
        input_window=input_window, input_scale=input_scale,
        hidden_window=hidden_window,
    ).to(device)

    class_weights = compute_class_weights(data['y_train'], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 原始 FQN 学习率：D10/D12 使用 0.001，其他 0.01
    lr = 0.001 if dataset_code in ('D10', 'D12') else 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_decay, patience=10, threshold=0.001
    )

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            _, logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_state)
    ckpt_path = save_dir / f"FQN_HEDL_stage1_{dataset_code}.pth"
    torch.save(best_state, ckpt_path)
    print(f"  Stage 1 saved: {ckpt_path}")
    return model


def train_stage2(
    model: FQN_HEDL,
    dataset_code: str,
    device: torch.device,
    epochs: int = 50,
    W: float = 2.0,
    save_dir: str = None,
    from_scratch: bool = False,
    loss_type: str = 'digamma',
) -> FQN_HEDL:
    """
    Stage 2: 证据学习微调。

    loss_type:
      - 'digamma': 标准 EDL digamma loss, 用 logits 计算 vacuity
      - 'mse':     标准 EDL MSE loss + KL annealing
      - 'log':     标准 EDL log-likelihood loss
      - 'henn':    HENN 正权重 masking loss (FC 保留 Stage 1 权重继续训练)

    如果 from_scratch=True，从随机初始化开始训练（跳过 Stage 1）。
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

    use_henn = (loss_type == 'henn')

    if from_scratch:
        lr = 0.01
        epochs = 100
        print(f"[Stage 2-{loss_type}] {name} ({dataset_code}): training from scratch, W={W}, lr={lr}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif use_henn:
        print(f"[Stage 2] {name} ({dataset_code}): HENN fine-tuning, W={W} (FC kept from Stage 1)")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        print(f"[Stage 2-{loss_type}] {name} ({dataset_code}): EDL fine-tuning, W={W}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    # ---- loss function ----
    if use_henn:
        def compute_loss(features, logits, y_onehot, ep):
            return edl_HENN(W, features, model.get_weight(), y_onehot, ep, num_classes, 10, device, logits)
    elif loss_type == 'mse':
        def compute_loss(features, logits, y_onehot, ep):
            return edl_mse_loss(logits, y_onehot, ep, num_classes, 10, device)
    elif loss_type == 'log':
        def compute_loss(features, logits, y_onehot, ep):
            return edl_log_loss(logits, y_onehot, ep, num_classes, 10, device)
    else:  # digamma
        def compute_loss(features, logits, y_onehot, ep):
            return edl_digamma_loss(logits, y_onehot, ep, num_classes, 10, device)

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
            y_batch_onehot.scatter_(1, y_batch.unsqueeze(1).to(device), 1)

            optimizer.zero_grad()
            features, logits = model(X_batch)
            loss = compute_loss(features, logits, y_batch_onehot, epoch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch_onehot = torch.zeros(y_batch.size(0), num_classes, device=device)
                y_batch_onehot.scatter_(1, y_batch.unsqueeze(1).to(device), 1)
                features, logits = model(X_batch)
                loss = compute_loss(features, logits, y_batch_onehot, epoch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    ckpt_path = save_dir / f"FQN_HEDL_stage2_{loss_type}_{dataset_code}.pth"
    torch.save(best_state, ckpt_path)
    print(f"  Stage 2 saved: {ckpt_path}")
    return model


def train_full_pipeline(
    dataset_code: str,
    device: torch.device = None,
    stage1_epochs: int = 200,
    stage2_epochs: int = 50,
    W: float = 2.0,
    save_dir: str = None,
    skip_stage1_if_exists: bool = True,
    no_stage1: bool = False,
    loss_type: str = 'digamma',
) -> FQN_HEDL:
    """运行完整的 Stage 1 + Stage 2 训练流水线。no_stage1=True 跳过 Stage 1 直接用 EDL 训练
    loss_type: 'digamma' | 'mse' | 'log' | 'henn'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_dir is None:
        save_dir = _current_dir / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt2_path = save_dir / f"FQN_HEDL_stage2_{loss_type}_{dataset_code}.pth"

    if skip_stage1_if_exists and ckpt2_path.exists():
        print(f"[Skip] Stage 2 checkpoint exists: {ckpt2_path}")
        data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
        model = FQN_HEDL(
            num_channels=data['in_channels'], num_classes=data['num_classes'],
            device=device, dim=data['dim'], depth=data['depth'],
            input_window=data['ks'], input_scale=data['input_scale'],
            hidden_window=data['kprop'], W=W,
        ).to(device)
        model.load_state_dict(torch.load(ckpt2_path, map_location=device))
        return model

    if no_stage1:
        data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
        model = FQN_HEDL(
            num_channels=data['in_channels'], num_classes=data['num_classes'],
            device=device, dim=data['dim'], depth=data['depth'],
            input_window=data['ks'], input_scale=data['input_scale'],
            hidden_window=data['kprop'], W=W,
        ).to(device)
        model = train_stage2(
            model=model, dataset_code=dataset_code, device=device,
            epochs=stage2_epochs, W=W, save_dir=str(save_dir),
            from_scratch=True, loss_type=loss_type,
        )
        return model

    ckpt1_path = save_dir / f"FQN_HEDL_stage1_{dataset_code}.pth"
    if skip_stage1_if_exists and ckpt1_path.exists():
        print(f"[Skip] Stage 1 checkpoint exists: {ckpt1_path}")
        data = load_and_split_dataset(dataset_code, local_path=LOCAL_UCR_PATH)
        model = FQN_HEDL(
            num_channels=data['in_channels'], num_classes=data['num_classes'],
            device=device, dim=data['dim'], depth=data['depth'],
            input_window=data['ks'], input_scale=data['input_scale'],
            hidden_window=data['kprop'], W=W,
        ).to(device)
        model.load_state_dict(torch.load(ckpt1_path, map_location=device))
    else:
        model = train_stage1(
            dataset_code=dataset_code, device=device,
            epochs=stage1_epochs, save_dir=str(save_dir),
        )

    model = train_stage2(
        model=model, dataset_code=dataset_code, device=device,
        epochs=stage2_epochs, W=W, save_dir=str(save_dir),
        loss_type=loss_type,
    )
    return model
