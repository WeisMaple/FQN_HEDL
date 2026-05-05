"""
FQN + HEDL 组合模型

FQN (Fully Quanvolutional Networks) 作为特征提取 backbone，
HEDL (Hyper-opinion Evidential Deep Learning) 作为不确定性量化 head。

输入: (N, C, T) 原始时间序列
输出: features, logits, vacuity (OOD score)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从本地 FQN_core 导入 Quanv1d 层
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from FQN_core import Quanv1d
from HEDL_core.losses import get_henn_fc


class FQNFeatureExtractor(nn.Module):
    """
    FQN 特征提取器：从原始时间序列 (N,C,T) 中提取 (N,dim) 特征向量。
    不含分类头，仅 Quanv1d 层 + BN + ReLU + 池化。
    """

    def __init__(
        self,
        num_channels: int,
        device: torch.device,
        dim: int = 16,
        depth: int = 3,
        input_window: int = 15,
        input_scale: int = 2,
        hidden_window: int = 5,
    ):
        super().__init__()
        self.feature_dim = dim

        self.input = Quanv1d(
            in_channels=num_channels,
            out_channels=dim,
            kernel_size=input_window,
            padding=(input_window - 1) // 2,
            stride=input_scale,
            dilation=1,
            device=device,
        )

        def quanv_bn_relu(in_ch, out_ch, k, d):
            return nn.Sequential(
                Quanv1d(
                    in_channels=in_ch, out_channels=out_ch,
                    kernel_size=k, padding=0, stride=1, dilation=d,
                    device=device,
                ),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            )

        self.hidden_layers = nn.Sequential(
            *[quanv_bn_relu(dim, dim, hidden_window, i + 1) for i in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)            # (N, dim, T1)
        x = self.hidden_layers(x)    # (N, dim, T2)
        x = F.adaptive_avg_pool1d(x, 1)  # (N, dim, 1)
        return x.view(x.size(0), -1)      # (N, dim)


class FQN_HEDL(nn.Module):
    """
    FQN backbone + Feature Expander + HEDL evidential head.

    在 FQN 低维特征和 HENN 证据头之间插入 MLP 扩展器，
    将特征维度从 ~8-64 扩展到 ≥128，确保 HENN 的投票机制有足够维度。

    forward() 返回 (features, logits)，匹配 HEDL 训练接口。
    compute_vacuity() 返回 vacuity = W / Σα，作为 OOD score。
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        device: torch.device,
        dim: int = 16,
        depth: int = 3,
        input_window: int = 15,
        input_scale: int = 2,
        hidden_window: int = 5,
        W: float = 2.0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.W = W
        self.device = device

        self.backbone = FQNFeatureExtractor(
            num_channels=num_channels,
            device=device,
            dim=dim,
            depth=depth,
            input_window=input_window,
            input_scale=input_scale,
            hidden_window=hidden_window,
        )

        hidden_dim = max(dim * 4, 128)
        self.expander = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        fqn_feats = self.backbone(x)              # (N, dim)
        features = self.expander(fqn_feats)       # (N, hidden_dim), ≥0
        logits = self.fc(features)                # (N, K)
        return features, logits

    def compute_vacuity(self, x: torch.Tensor, method: str = 'logits') -> torch.Tensor:
        """计算 vacuity 作为 OOD score（越高越可能是 OOD）

        method='logits': 标准 EDL, evidence=relu(logits), alpha=evidence+W/K, vacuity=K/S
        method='henn':   HENN 正权重 masking, vacuity=W/S (与原版 HEDL 一致)
        """
        features, logits = self.forward(x)
        if method == 'henn':
            alpha = get_henn_fc(
                features, self.fc.weight, self.W, self.num_classes, self.device
            )
            S = alpha.sum(dim=1)
            vacuity = self.W / (S + 1e-12)
        else:
            evidence = F.relu(logits)
            alpha = evidence + self.W / self.num_classes
            S = alpha.sum(dim=1)
            vacuity = self.num_classes / (S + 1e-12)
        return vacuity

    def get_weight(self):
        """返回 FC 权重矩阵（HENN loss 需要）"""
        return self.fc.weight
