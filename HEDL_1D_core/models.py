import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Conv1D Residual Block
# ─────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity)


# ─────────────────────────────────────────────
# Conv1D Backbone  (drop-in 替换 ResNet)
# ─────────────────────────────────────────────
class Conv1DBackbone(nn.Module):
    """
    轻量级 1-D 残差网络，用于时序分类。

    Args:
        in_channels  : 输入通道数（多变量序列的变量数）
        feat_dim     : 全局池化后的特征维度（默认 256）
        num_classes  : 分类数
        loss         : 损失函数名称（与原框架保持一致）
    """
    def __init__(self, in_channels: int, num_classes: int,
                 feat_dim: int = 256, loss: str = 'Softmax'):
        super().__init__()
        self.loss     = loss
        self.feat_dim = feat_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual stages
        self.layer1 = self._make_layer(64,  64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, feat_dim, blocks=2, stride=2)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dropout（仅 dropout 模式启用）
        self.dropout = nn.Dropout(p=0.2) if loss == 'dropout' else nn.Identity()

        # Classifier head
        self.fc = nn.Linear(feat_dim, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride):
        layers = [ResBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x : (batch, in_channels, seq_len)
        returns: features (batch, feat_dim),  logits (batch, num_classes)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)                      # (B, feat_dim, 1)
        features = x.view(x.size(0), -1)     # (B, feat_dim)

        if self.loss == 'edl_HENN':
            features = F.relu(features)

        features = self.dropout(features)
        out = self.fc(features)
        return features, out

    def get_weight(self):
        """与原框架接口一致"""
        return self.fc.weight