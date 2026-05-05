"""
数据集加载与预处理工具
基于 FQN 论文中的 UEA/UCR 时间序列数据加载流程
修改：统一标准化参数，支持返回全局均值和标准差
"""

import os
from pathlib import Path
import numpy as np
import torch
import re
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict, Any

# load_from_tsfile 优先（新版 aeon），不可用则 fallback 到 load_classification
try:
    from aeon.datasets import load_from_tsfile
    _use_tsfile = True
except ImportError:
    try:
        from aeon.datasets import load_classification
        _use_tsfile = False
    except ImportError:
        load_classification = None
        _use_tsfile = False

# ===================== 1. 数据集配置 =====================
DATASET_CONFIG = {
    'D1': ('Chinatown', 1, 2, 64, 0.85, 9, 16, 3, 4, 1),
    'D2': ('SharePriceIncrease', 1, 2, 256, 0.85, 15, 16, 3, 3, 3),
    'D3': ('SyntheticControl', 1, 6, 128, 0.85, 15, 48, 3, 4, 2),
    'D4': ('PhalangesOutlinesCorrect', 1, 2, 256, 0.75, 21, 32, 3, 4, 2),
    'D5': ('ECG200', 1, 2, 32, 0.75, 31, 16, 3, 4, 4),
    'D6': ('PowerCons', 1, 2, 64, 0.75, 33, 32, 7, 4, 2),
    'D7': ('ToeSegmentation2', 1, 2, 32, 0.75, 31, 24, 5, 5, 5),
    'D8': ('DiatomSizeReduction', 1, 4, 32, 0.8, 45, 32, 9, 4, 4),
    'D9': ('Earthquakes', 1, 2, 64, 0.9, 21, 32, 5, 5, 5),
    'D10': ('InsectEPGRegularTrain', 1, 3, 32, 0.75, 33, 16, 9, 5, 5),
    'D11': ('StarLightCurves', 1, 3, 128, 0.8, 21, 32, 5, 5, 8),
    'D12': ('NerveDamage', 1, 3, 32, 0.85, 41, 16, 7, 6, 10),
    'D13': ('BinaryHeartbeat', 1, 2, 32, 0.9, 15, 8, 7, 5, 20),
    'D14': ('Epilepsy', 3, 4, 64, 0.85, 25, 64, 3, 5, 3),
    'D15': ('EthanolConcentration', 3, 4, 128, 0.85, 45, 24, 7, 7, 10),
    'D16': ('Blink', 4, 2, 128, 0.85, 41, 24, 5, 5, 5),
    'D17': ('SelfRegulationSCP1', 7, 2, 64, 0.85, 41, 32, 5, 5, 9),
    'D18': ('HandMovementDirection', 10, 4, 128, 0.75, 9, 8, 5, 9, 2),
    'D19': ('FingerMovements', 28, 2, 64, 0.75, 5, 16, 3, 4, 2),
    'D20': ('MotorImagery', 64, 2, 16, 0.9, 3, 16, 3, 3, 25),
}

LOCAL_UCR_PATH = r"F:\QML\Papers\CODES\quantumbanu-Quanv1D\UCRArchive_2018\UCRArchive_2018"

def get_dataset_config(dataset_code: str) -> Tuple[str, int, int, int, float, int, int, int, int, int]:
    return DATASET_CONFIG[dataset_code]

# ===================== 2. 手动 .ts 文件解析器（aeon 不可用时的回退） =====================
def _parse_ts_metadata(filepath: str):
    """解析 UCR .ts 文件元数据, 返回 (n_channels, has_timestamps)"""
    n_channels = 1
    has_timestamps = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '@data':
                break
            if line.lower().startswith('@univariate') and 'false' in line.lower():
                # 多变量但通道数未指定，设为 None 表示需要外部传入
                n_channels = None
            if line.lower().startswith('@timestamps') and 'true' in line.lower():
                has_timestamps = True
    return n_channels, has_timestamps


def _parse_ts_data(filepath: str):
    """手动解析 UCR .ts 数据行, 返回 (X_flat, y, n_channels_from_data)"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '@data':
            data_start = i + 1
            break

    data_lines = [l.strip() for l in lines[data_start:] if l.strip()]

    samples = []
    labels = []
    for line in data_lines:
        if ':' not in line:
            continue
        vals, label = line.rsplit(':', 1)
        dim_vals = [float(x) for x in vals.split(',') if x]
        samples.append(dim_vals)
        labels.append(label.strip())

    max_len = max(len(s) for s in samples)
    n_samples = len(samples)

    X_flat = np.zeros((n_samples, max_len), dtype=np.float32)
    for i, s in enumerate(samples):
        X_flat[i, :len(s)] = s

    try:
        y = np.array([int(l) if (l.isdigit() or (l.startswith('-') and l[1:].isdigit())) else l for l in labels])
    except (ValueError, TypeError):
        y = np.array(labels)

    if y.dtype == object:
        y = np.array(labels, dtype=object)  # keep as object for LabelEncoder

    return X_flat, y


def load_uea_ucr_data(dataset_name: str, local_path: Optional[str] = None,
                      n_channels: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载 UEA/UCR 数据集. 自动选择最佳加载方式."""
    if local_path is None:
        local_path = LOCAL_UCR_PATH

    # 优先使用 load_from_tsfile（新版 aeon）
    if _use_tsfile:
        data_dir = os.path.join(local_path, dataset_name)
        train_file = os.path.join(data_dir, f"{dataset_name}_TRAIN.ts")
        test_file = os.path.join(data_dir, f"{dataset_name}_TEST.ts")
        X_train, y_train = load_from_tsfile(train_file)
        X_test, y_test = load_from_tsfile(test_file)

        if isinstance(X_train, list):
            X = X_train + X_test
        else:
            X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        if isinstance(X, list):
            shapes = [x.shape for x in X]
            n_ch = shapes[0][0]
            max_len = max(s[1] for s in shapes)
            n_samples = len(X)
            X_fixed = np.zeros((n_samples, n_ch, max_len), dtype=np.float32)
            for i, x in enumerate(X):
                cur_len = x.shape[1]
                X_fixed[i, :, :cur_len] = x
            X = X_fixed
        elif X.ndim == 2:
            X = X[:, np.newaxis, :]

    # 其次使用 load_classification（旧版 aeon）
    elif load_classification is not None:
        X, y = load_classification(dataset_name, extract_path=local_path)
        if isinstance(X, list):
            shapes = [x.shape for x in X]
            n_ch = shapes[0][0]
            max_len = max(s[1] for s in shapes)
            n_samples = len(X)
            X_fixed = np.zeros((n_samples, n_ch, max_len), dtype=np.float32)
            for i, x in enumerate(X):
                cur_len = x.shape[1]
                X_fixed[i, :, :cur_len] = x
            X = X_fixed
        elif X.ndim == 2:
            X = X[:, np.newaxis, :]

    # 最后回退到内置解析器
    else:
        if n_channels is None:
            n_channels = 1
        data_dir = os.path.join(local_path, dataset_name)
        train_file = os.path.join(data_dir, f"{dataset_name}_TRAIN.ts")
        test_file = os.path.join(data_dir, f"{dataset_name}_TEST.ts")

        X_train_flat, y_train = _parse_ts_data(train_file)
        X_test_flat, y_test = _parse_ts_data(test_file)
        max_len = max(X_train_flat.shape[1], X_test_flat.shape[1])

        X_train_padded = np.zeros((X_train_flat.shape[0], max_len), dtype=np.float32)
        X_train_padded[:, :X_train_flat.shape[1]] = X_train_flat
        X_test_padded = np.zeros((X_test_flat.shape[0], max_len), dtype=np.float32)
        X_test_padded[:, :X_test_flat.shape[1]] = X_test_flat

        X_flat = np.concatenate([X_train_padded, X_test_padded], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        T = X_flat.shape[1] // n_channels
        X = X_flat.reshape(X_flat.shape[0], n_channels, T)

    if X is None or y is None:
        print(f"数据集 {dataset_name} 加载后为空")
        return None, None

    if not np.issubdtype(y.dtype, np.integer):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    return X.astype(np.float32), y.astype(np.int64)

# ===================== 3. 标准化与划分（per-time-step 标准化） =====================
def normalize_and_split_with_global_stats(X: np.ndarray, y: np.ndarray,
                                           test_size: float = 0.2, val_size: float = 0.2,
                                           random_state: int = 42):
    """
    Per-time-step StandardScaler 标准化，保存每个时间步的 mean/std 用于 OOD。
    返回：X_train, y_train, X_val, y_val, X_test, y_test, global_means, global_stds
    """
    n_samples, n_channels, n_timesteps = X.shape
    global_means = np.zeros((n_channels, n_timesteps))
    global_stds = np.zeros((n_channels, n_timesteps))
    X_norm = np.zeros_like(X)
    for c in range(n_channels):
        X_c = X[:, c, :]  # (n_samples, n_timesteps)
        scaler = StandardScaler()
        X_c_norm = scaler.fit_transform(X_c)  # 每个时间步跨样本标准化
        X_norm[:, c, :] = X_c_norm
        global_means[c] = scaler.mean_       # (n_timesteps,)
        global_stds[c] = scaler.scale_ + 1e-8

    # 划分训练+验证 vs 测试
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss.split(X_norm, y))
    X_train_val, y_train_val = X_norm[train_val_idx], y[train_val_idx]
    X_test, y_test = X_norm[test_idx], y[test_idx]

    # 划分训练 vs 验证
    relative_val_size = val_size / (1 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))
    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test, global_means, global_stds

# ===================== 4. 便捷函数：一键加载并划分数据集 =====================
def load_and_split_dataset(dataset_code: str, test_size: float = 0.2, val_size: float = 0.2,
                           random_state: int = 42, local_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载、标准化、划分，并返回数据字典，同时包含全局均值和标准差。
    """
    name, in_channels, num_classes, batch_size, lr_decay, ks, dim, kprop, depth, input_scale = get_dataset_config(dataset_code)
    X, y = load_uea_ucr_data(name, local_path=local_path, n_channels=in_channels)
    if X is None:
        raise ValueError(f"无法加载数据集 {name}")

    # 标签转换为连续的 0-based (不能用 y - y.min(), 某些数据集标签非连续如 {-1,1})
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, y_train, X_val, y_val, X_test, y_test, global_means, global_stds = normalize_and_split_with_global_stats(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )

    return {
        'name': name,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'lr_decay': lr_decay,
        'ks': ks,
        'dim': dim,
        'kprop': kprop,
        'depth': depth,
        'input_scale': input_scale,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'global_means': global_means,
        'global_stds': global_stds,
    }

# ===================== 5. OOD 数据预处理（使用相同的全局统计量） =====================
def prepare_ood_data(ood_name: str, global_means: np.ndarray, global_stds: np.ndarray,
                     target_len: int, local_path: Optional[str] = None):
    """
    加载 OOD 原始数据，使用 ID 数据集的全局均值和标准差进行标准化，并统一长度。
    """
    X_ood, y_ood = load_uea_ucr_data(ood_name, local_path=local_path, n_channels=global_means.shape[0])
    if X_ood is None:
        raise ValueError(f"无法加载 OOD 数据集 {ood_name}")
    le = LabelEncoder()
    y_ood = le.fit_transform(y_ood)

    # 长度调整到 target_len
    if X_ood.shape[2] > target_len:
        X_ood = X_ood[:, :, :target_len]
    elif X_ood.shape[2] < target_len:
        pad = target_len - X_ood.shape[2]
        X_ood = np.pad(X_ood, ((0,0), (0,0), (0,pad)), constant_values=0)

    # 使用 ID 数据集的 per-time-step 均值和标准差进行标准化
    n_channels_ood = X_ood.shape[1]
    n_id_channels = global_means.shape[0]
    X_ood_norm = np.zeros_like(X_ood)
    for c in range(min(n_channels_ood, n_id_channels)):
        cur_len = X_ood.shape[2]
        if cur_len > target_len:
            gm = global_means[c][:target_len]
            gs = global_stds[c][:target_len]
        elif cur_len < target_len:
            gm = np.pad(global_means[c], (0, target_len - cur_len), constant_values=0)
            gs = np.pad(global_stds[c], (0, target_len - cur_len), constant_values=1)
        else:
            gm = global_means[c]
            gs = global_stds[c]
        X_ood_norm[:, c, :] = (X_ood[:, c, :] - gm) / gs
    return X_ood_norm, y_ood

# ===================== 6. 计算类别权重 =====================
def compute_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * len(classes)
    return class_weights