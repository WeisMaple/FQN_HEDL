"""
轻量 Dataset 包装器。
数据已由 data_utils.load_and_split_dataset 完成标准化和切分，
本类只负责包装成 PyTorch Dataset。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Args:
        X           : np.ndarray, shape (N, C, L), float32，已标准化
        y           : np.ndarray, shape (N,),      int64，0-based 标签
        num_classes : 类别总数（用于 one-hot 编码）
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, num_classes: int):
        self.X           = torch.from_numpy(X.astype(np.float32))   # (N, C, L)
        self.y           = y.astype(np.int64)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x      = self.X[idx]                                          # (C, L)
        label  = torch.tensor(self.y[idx], dtype=torch.int64)
        target = F.one_hot(label, num_classes=self.num_classes)       # (num_classes,)
        return x, target