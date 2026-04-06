import torch
import torch.nn as nn
import torch.nn.functional as F

from FQN_core import Quanv1d   # 直接复用你提供的 FQN_core.py


# ══════════════════════════════════════════════════════════════
#  Conv1D Backbone（保留原有，略）
# ══════════════════════════════════════════════════════════════
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_k3  = nn.Conv1d(in_channels, out_channels // 4, 3,  padding=1,  bias=False)
        self.conv_k9  = nn.Conv1d(in_channels, out_channels // 4, 9,  padding=4,  bias=False)
        self.conv_k19 = nn.Conv1d(in_channels, out_channels // 4, 19, padding=9,  bias=False)
        self.maxpool  = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_mp  = nn.Conv1d(in_channels, out_channels // 4, 1,  bias=False)
        self.bn       = nn.BatchNorm1d(out_channels)
        self.act      = nn.GELU()
        self.residual = (nn.Sequential(
                            nn.Conv1d(in_channels, out_channels, 1, bias=False),
                            nn.BatchNorm1d(out_channels))
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        out = torch.cat([self.conv_k3(x), self.conv_k9(x),
                         self.conv_k19(x), self.conv_mp(self.maxpool(x))], dim=1)
        return self.act(self.bn(out)) + self.residual(x)


class Conv1DBackbone(nn.Module):
    def __init__(self, in_channels, num_classes, feat_dim=256, loss='Softmax'):
        super().__init__()
        self.loss = loss
        self.feat_dim = feat_dim
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.GELU())
        self.stage1 = nn.Sequential(InceptionBlock1D(64,  128), nn.MaxPool1d(2))
        self.stage2 = nn.Sequential(InceptionBlock1D(128, 256), nn.MaxPool1d(2))
        self.stage3 = InceptionBlock1D(256, feat_dim)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3 if loss == 'dropout' else 0.2)
        self.fc      = nn.Linear(feat_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        x = self.gap(x).squeeze(-1)
        if self.loss == 'edl_HENN':
            x = F.relu(x)
        features = self.dropout(x)
        return features, self.fc(features)

    def get_weight(self):
        return self.fc.weight


# ══════════════════════════════════════════════════════════════
#  FQN Backbone — 将 FQN 改造为 (features, logits) 接口
# ══════════════════════════════════════════════════════════════
class FQNBackbone(nn.Module):
    """
    用 FQN_core 的 Quanv1d 组件构建 backbone，
    最终分类层改为普通 Linear，以暴露 features 和 fc.weight，
    满足 HEDL 框架对 get_weight() 的要求。

    参数含义与 DATASET_CONFIG 的对应关系：
        input_window  ← ks
        dim           ← dim
        depth         ← depth
        input_scale   ← input_scale  (输入层 stride)
        hidden_window ← kprop        (隐藏层 kernel size)
    """
    def __init__(self, in_channels: int, num_classes: int, device: str = 'cpu',
                 dim: int = 16, depth: int = 3, input_window: int = 15,
                 input_scale: int = 2, hidden_window: int = 5,
                 loss: str = 'Softmax'):
        super().__init__()
        self.loss     = loss
        self.feat_dim = dim

        # ── 输入量子卷积层 ────────────────────────────────────────────────
        self.input_layer = Quanv1d(
            in_channels  = in_channels,
            out_channels = dim,
            kernel_size  = input_window,
            padding      = (input_window - 1) // 2,   # same padding
            stride       = input_scale,
            dilation     = 1,
            device       = device,
        )

        # ── 堆叠隐藏量子卷积层（与原 FQN 一致，使用膨胀卷积增大感受野）────
        hidden = []
        for i in range(depth):
            hidden.append(Quanv1d(
                in_channels  = dim,
                out_channels = dim,
                kernel_size  = hidden_window,
                padding      = 0,
                stride       = 1,
                dilation     = i + 1,
                device       = device,
            ))
            hidden.append(nn.BatchNorm1d(dim))
            hidden.append(nn.ReLU(inplace=True))
        self.hidden_layers = nn.Sequential(*hidden)

        # ── 全局平均池化 + 分类头 ─────────────────────────────────────────
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc      = nn.Linear(dim, num_classes)   # ← HEDL 框架通过此层取权重

        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        x       : (B, in_channels, seq_len)
        returns : features (B, dim),  logits (B, num_classes)
        """
        x = self.input_layer(x)          # (B, dim, L1)
        x = self.hidden_layers(x)        # (B, dim, L2)
        x = self.gap(x).squeeze(-1)      # (B, dim)

        # edl_HENN 要求特征非负
        if self.loss == 'edl_HENN':
            x = F.relu(x)

        features = self.dropout(x)
        out      = self.fc(features)     # (B, num_classes)
        return features, out

    def get_weight(self) -> torch.Tensor:
        """与 HEDL 框架接口一致，返回分类头权重。"""
        return self.fc.weight