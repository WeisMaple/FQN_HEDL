import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock1D(nn.Module):
    """
    多尺度感受野模块，同时捕捉短程、中程、长程时序模式。
    类比 Inception 结构，适合时序长度变化大的UCR数据集。
    """

    def __init__(self, in_channels, out_channels, bottleneck=32):
        super().__init__()
        # 三个不同 kernel size 的并行卷积
        self.conv_k3 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=3,
                                 padding=1, bias=False)
        self.conv_k9 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=9,
                                 padding=4, bias=False)
        self.conv_k19 = nn.Conv1d(in_channels, out_channels // 4, kernel_size=19,
                                  padding=9, bias=False)
        # MaxPool 分支
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(in_channels, out_channels // 4, kernel_size=1,
                                 bias=False)

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        # 残差连接（维度不同时用 1x1 conv 对齐）
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = torch.cat([
            self.conv_k3(x),
            self.conv_k9(x),
            self.conv_k19(x),
            self.conv_mp(self.maxpool(x)),
        ], dim=1)
        out = self.act(self.bn(out))
        return out + self.residual(x)


class Conv1DBackbone(nn.Module):
    """
    基于 InceptionBlock1D 的时序分类 backbone。

    Args:
        in_channels : 输入通道数（多变量序列的变量数）
        num_classes : 分类数
        feat_dim    : 全局池化后的特征维度
        loss        : 损失函数名称（与原框架保持一致）
    """

    def __init__(self, in_channels: int, num_classes: int,
                 feat_dim: int = 256, loss: str = 'Softmax'):
        super().__init__()
        self.loss = loss
        self.feat_dim = feat_dim

        # Stem：初步提取局部特征
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # 三个 Inception 阶段，逐步增加通道和下采样
        self.stage1 = nn.Sequential(
            InceptionBlock1D(64, 128),
            nn.MaxPool1d(2),
        )
        self.stage2 = nn.Sequential(
            InceptionBlock1D(128, 256),
            nn.MaxPool1d(2),
        )
        self.stage3 = InceptionBlock1D(256, feat_dim)

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dropout（dropout 模式和正常训练均使用，抑制过拟合）
        drop_rate = 0.3 if loss == 'dropout' else 0.2
        self.dropout = nn.Dropout(p=drop_rate)

        # 分类头
        self.fc = nn.Linear(feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x        : (B, in_channels, seq_len)
        returns  : features (B, feat_dim),  logits (B, num_classes)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).squeeze(-1)  # (B, feat_dim)

        if self.loss == 'edl_HENN':
            x = F.relu(x)  # 保证 evidence >= 0

        features = self.dropout(x)
        out = self.fc(features)
        return features, out

    def get_weight(self):
        return self.fc.weight