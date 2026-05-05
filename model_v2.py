"""
FQN_HEDL_v2: 方案二模型

使用原始 FQN 完整结构（Quanv1d(dim→K, ks=1) 逐帧投影）+ 集合核池化 + HEDL 投影。

Stage 1: forward_gap() → logits (GAP 池化, CrossEntropy 预训练)
Stage 2: forward() → alpha (集合核池化 + HEDL 投影, EDL loss 微调)
"""
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from FQN_core import Quanv1d
from set_kernel import SetKernelPooling, HEDLProjection, generate_hyper_domain_sets


class FQN_HEDL_v2(nn.Module):
    """
    FQN + HEDL 方案二：集合核池化版本。

    骨干结构完全复现原始 FQN (core.py):
      input Quanv1d(C→dim, ks, stride)
      → depth × (Quanv1d(dim→dim) + BN + ReLU)
      → output Quanv1d(dim→K, ks=1)  [逐帧类别投影]
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
        tau_init: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.K = num_classes
        self.W = W
        self.device = device

        # ---- FQN backbone (与 core.py FQN 完全一致) ----
        self.input = Quanv1d(
            in_channels=num_channels, out_channels=dim,
            kernel_size=input_window,
            padding=(input_window - 1) // 2,
            stride=input_scale, dilation=1,
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

        # 逐帧类别投影层 (原始 FQN 的 self.output)
        self.output = Quanv1d(
            in_channels=dim, out_channels=num_classes,
            kernel_size=1, padding=0, stride=1, dilation=1,
            device=device,
        )

        # ---- 方案二专属模块 ----
        # 只用单例集合: N_sets=K, 避免多类集合的累积放大问题
        max_set_size = 1
        self.set_pooling = SetKernelPooling(
            K=num_classes, max_set_size=max_set_size,
            tau_init=tau_init,
        )
        masks = self.set_pooling.masks  # (N_sets, K)
        self.hedL_proj = HEDLProjection(K=num_classes, masks=masks, W=W)

        self._n_sets = masks.shape[0]

    @property
    def N_sets(self):
        return self._n_sets

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """FQN 骨干前向, 输出 (N, K, T2) 逐帧类别激活 (无 GAP)."""
        x = self.input(x)            # (N, dim, T1)
        x = self.hidden_layers(x)    # (N, dim, T2)
        x = self.output(x)           # (N, K, T2)
        return x

    def forward_gap(self, x: torch.Tensor) -> torch.Tensor:
        """Stage 1 用：GAP 池化 → logits (N, K). 与原始 FQN 完全一致."""
        x = self.forward_backbone(x)   # (N, K, T2)
        x = F.adaptive_avg_pool1d(x, 1)  # (N, K, 1)
        return x.view(x.size(0), -1)   # (N, K)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 2 用：集合核池化 + HEDL 投影 → (alpha, evidence)."""
        backbone_out = self.forward_backbone(x)  # (N, K, T2)
        evidence = self.set_pooling(backbone_out)  # (N, N_sets)
        alpha = self.hedL_proj(evidence)           # (N, K)
        return alpha, evidence

    def compute_vacuity(self, x: torch.Tensor) -> torch.Tensor:
        """计算 vacuity = K / Σα 作为 OOD score."""
        with torch.no_grad():
            alpha, _ = self.forward(x)
            S = alpha.sum(dim=1)
            vacuity = self.K / (S + 1e-12)
        return vacuity

    def forward_backbone_with_gap(self, x: torch.Tensor) -> torch.Tensor:
        """兼容旧接口：同 forward_gap."""
        return self.forward_gap(x)
