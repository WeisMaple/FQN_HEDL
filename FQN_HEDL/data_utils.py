"""
数据集加载与预处理工具
基于 FQN 论文中的 UEA/UCR 时间序列数据加载流程
修改：统一标准化参数，支持返回全局均值和标准差
"""

import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from aeon.datasets import load_classification
from typing import Tuple, Optional, Dict, Any

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

# ===================== 2. 数据集加载 =====================
def load_uea_ucr_data(dataset_name: str, local_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载原始数据，返回 (X, y)"""
    try:
        if local_path is not None:
            X, y = load_classification(dataset_name, extract_path=local_path)
        else:
            X, y = load_classification(dataset_name)
    except Exception as e:
        print(f"加载数据集 {dataset_name} 失败: {e}")
        return None, None

    if X is None or y is None:
        print(f"数据集 {dataset_name} 加载后为空")
        return None, None

    # 处理 X：如果是列表，填充到最大长度
    if isinstance(X, list):
        shapes = [x.shape for x in X]
        n_channels = shapes[0][0]
        max_len = max(s[1] for s in shapes)
        n_samples = len(X)
        X_fixed = np.zeros((n_samples, n_channels, max_len), dtype=np.float32)
        for i, x in enumerate(X):
            cur_len = x.shape[1]
            X_fixed[i, :, :cur_len] = x
        X = X_fixed
    else:
        if X.ndim == 2:
            X = X[:, np.newaxis, :]

    # 处理标签：如果是字符串，转换为整数编码
    if y.dtype == np.object_ or (len(y) > 0 and isinstance(y[0], str)):
        le = LabelEncoder()
        y = le.fit_transform(y)
    return X.astype(np.float32), y.astype(np.int64)

# ===================== 3. 标准化与划分（统一使用全局统计量） =====================
def normalize_and_split_with_global_stats(X: np.ndarray, y: np.ndarray,
                                           test_size: float = 0.2, val_size: float = 0.2,
                                           random_state: int = 42):
    """
    先计算全数据集的通道均值和标准差，标准化后再划分。
    返回：X_train, y_train, X_val, y_val, X_test, y_test, global_means, global_stds
    """
    n_samples, n_channels, n_timesteps = X.shape
    # 计算每个通道的全局均值和标准差
    global_means = np.zeros(n_channels)
    global_stds = np.zeros(n_channels)
    for c in range(n_channels):
        X_c = X[:, c, :]  # (n_samples, n_timesteps)
        global_means[c] = X_c.mean()
        global_stds[c] = X_c.std() + 1e-8

    # 标准化整个数据集
    X_norm = np.zeros_like(X)
    for c in range(n_channels):
        X_norm[:, c, :] = (X[:, c, :] - global_means[c]) / global_stds[c]

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
    X, y = load_uea_ucr_data(name, local_path=local_path)
    if X is None:
        raise ValueError(f"无法加载数据集 {name}")

    # 标签转换为 0-based
    y = y - y.min()

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
    X_ood, y_ood = load_uea_ucr_data(ood_name, local_path=local_path)
    if X_ood is None:
        raise ValueError(f"无法加载 OOD 数据集 {ood_name}")
    y_ood = y_ood - y_ood.min()

    # 长度调整到 target_len
    if X_ood.shape[2] > target_len:
        X_ood = X_ood[:, :, :target_len]
    elif X_ood.shape[2] < target_len:
        pad = target_len - X_ood.shape[2]
        X_ood = np.pad(X_ood, ((0,0), (0,0), (0,pad)), constant_values=0)

    # 使用传入的全局均值和标准差进行标准化
    n_channels = X_ood.shape[1]
    X_ood_norm = np.zeros_like(X_ood)
    for c in range(n_channels):
        X_ood_norm[:, c, :] = (X_ood[:, c, :] - global_means[c]) / global_stds[c]
    return X_ood_norm, y_ood

# ===================== 6. 计算类别权重 =====================
def compute_class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    class_weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * len(classes)
    return class_weights