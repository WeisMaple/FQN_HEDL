"""
Fully Quanvolutional Network (FQN) 模块化组件
基于论文 "Fully Quanvolutional Networks for Time Series Classification" (KDD'25)
原始代码作者: Nabil Anan Orka 等
模块化重构: 便于与其他模型组合
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


# ===================== 1. 量子态归一化与编码模块 =====================
class QubitNormalization(nn.Module):
    """
    将经典特征编码为量子态振幅。
    输入: (batch, in_features)
    输出: (batch, 2^num_wires) 归一化后的振幅向量
    """
    def __init__(self, num_wires: int, temp: float,
                 noise_strength: float = 0.001, dropout_prob: float = 0.01,
                 apply_noise: bool = False):
        super().__init__()
        self.num_wires = num_wires
        self.temp = temp
        self.noise_strength = noise_strength
        self.dropout_prob = dropout_prob
        self.apply_noise = apply_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, in_size = x.shape
        temp = self.temp if isinstance(self.temp, torch.Tensor) else torch.tensor(self.temp, device=x.device)
        # softmax + sqrt 保证概率幅归一化
        x = F.softmax(x / temp, dim=1).sqrt()
        # 补零到 2^num_wires
        pad_size = (2 ** self.num_wires) - in_size
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))
        # 可选噪声（模拟硬件不完美）
        if self.apply_noise:
            noise = torch.randn_like(x) * self.noise_strength
            x = x + noise
            keep_prob = 1 - self.dropout_prob
            dropout_mask = torch.bernoulli(torch.full_like(x, keep_prob)) / keep_prob
            x = x * dropout_mask
            x = x / torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=-1, keepdim=True))
        return x


# ===================== 2. 量子酉变换模块（可训练 ansatz） =====================
class Unitary(nn.Module):
    """
    构建多层、多 qubit 的酉变换矩阵。
    输入: 无（内部使用参数 weight, phi）
    输出: (num_filters, 2^num_wires, 2^num_wires) 酉矩阵
    """
    def __init__(self, num_filters: int, num_layers: int, num_wires: int,
                 device: torch.device, apply_noise: bool = False):
        super().__init__()
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.device = device
        self.pi_half = math.pi / 2
        self.apply_noise = apply_noise

    def forward(self, weight: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # weight: (num_filters, num_layers, num_wires, 2)  [theta, lambda]
        # phi:    (num_filters, num_layers, num_wires) 固定随机相位
        if self.apply_noise:
            weight = weight + torch.randn_like(weight) * 0.001

        final_product = torch.eye(2**self.num_wires, dtype=torch.complex64, device=self.device)
        final_product = final_product.unsqueeze(0).repeat(self.num_filters, 1, 1)

        for layer in range(self.num_layers):
            current_weight = weight[:, layer, :, :]   # (F, W, 2)
            angle = current_weight[..., 0] * self.pi_half
            cos_term = torch.cos(angle)
            sin_term = torch.sin(angle)
            exp_term1 = torch.exp(1j * phi[:, layer, :])
            exp_term2 = torch.exp(1j * current_weight[..., 1])

            # 每个 qubit 的 2x2 酉矩阵
            U = torch.zeros(self.num_filters, self.num_wires, 2, 2, dtype=torch.complex64, device=self.device)
            U[..., 0, 0] = cos_term
            U[..., 0, 1] = -sin_term * exp_term2
            U[..., 1, 0] = sin_term * exp_term1
            U[..., 1, 1] = cos_term * exp_term1 * exp_term2

            # 张量积得到该层的总酉矩阵
            kron_product = U[:, 0]
            for i in range(1, self.num_wires):
                kron_product = torch.einsum('bij,bkl->bikjl', kron_product, U[:, i])
                kron_product = kron_product.view(self.num_filters, 2**(i+1), 2**(i+1))
            final_product = torch.bmm(final_product, kron_product)
        return final_product


# ===================== 3. 测量模块（期望值） =====================
class Measurement(nn.Module):
    """
    生成每个 qubit 的 Pauli-Z 测量算符。
    返回: (num_wires, 2^num_wires, 2^num_wires) 观测矩阵
    """
    def __init__(self, num_wires: int, device: torch.device):
        super().__init__()
        self.num_wires = num_wires
        self.register_buffer('dummy', torch.zeros(1, device=device))

    def forward(self) -> torch.Tensor:
        device = self.dummy.device
        I = torch.eye(2, dtype=torch.complex64, device=device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
        results = []
        for i in range(self.num_wires):
            op = torch.tensor(1, dtype=torch.complex64, device=device)
            for j in range(self.num_wires):
                op = torch.kron(op, Z if i == j else I)
            results.append(op)
        return torch.stack(results)


# ===================== 4. 完整的 Quanv1D 层 =====================
class Quanv1d(nn.Module):
    """
    1D 量子卷积层，模仿 Conv1d 的行为。
    支持多通道输入、可变 kernel size、可定制输出通道数。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1,
                 bias: bool = False, device: str = "cpu", temp: float = 1.0,
                 shots: int = 0, noise: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_wires = math.ceil(math.log2(max(in_channels * kernel_size, 2)))
        self.num_filters = (out_channels + self.num_wires - 1) // self.num_wires
        self.num_layers = kernel_size
        self.out_channels = out_channels
        self.device = torch.device(device)
        self.temp = torch.tensor(temp, dtype=torch.float32)
        self.shots = shots
        self.noise = noise

        # 1D 滑动窗口提取器
        self.patcher = nn.Unfold(kernel_size=(kernel_size, 1), stride=(stride, 1),
                                 padding=(padding, 0), dilation=(dilation, 1))
        # 量子子模块
        self.qubit_norm = QubitNormalization(self.num_wires, self.temp, apply_noise=self.noise)
        self.unitary_op = Unitary(self.num_filters, self.num_layers, self.num_wires,
                                  self.device, apply_noise=self.noise)
        self.measurement_op = Measurement(self.num_wires, self.device)

        # 可训练参数
        self.weight = nn.Parameter(
            nn.init.kaiming_uniform_(torch.randn((self.num_filters, self.num_layers,
                                                  self.num_wires, 2), device=self.device)),
            requires_grad=True
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, device=self.device),
                                 requires_grad=True) if bias else None
        # 固定的随机相位
        self.register_buffer('phi', torch.rand((self.num_filters, self.num_layers, self.num_wires),
                                               device=self.device))

    def apply_depolarizing_noise(self, x: torch.Tensor) -> torch.Tensor:
        """模拟 depolarizing 噪声（1% 概率随机施加 Pauli 门）"""
        if torch.rand(1).item() >= 0.01:
            return x
        I = torch.eye(2, dtype=torch.complex64, device=x.device)
        X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=x.device)
        Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=x.device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=x.device)
        gate = random.choice([X, Y, Z])
        target_wire = random.randint(0, self.num_wires - 1)
        full = torch.eye(1, dtype=torch.complex64, device=x.device)
        for i in range(self.num_wires):
            full = torch.kron(full, gate if i == target_wire else I)
        return torch.einsum('bfij,jk->bfik', x, full)

    def adjust_probabilities_with_shots(self, x: torch.Tensor) -> torch.Tensor:
        """模拟有限 shot 测量引入的统计噪声"""
        p = torch.abs(x) ** 2
        noise_scale = torch.sqrt(p * (1 - p) / self.shots)
        x = x + torch.complex(
            torch.randn_like(x.real) * noise_scale,
            torch.randn_like(x.imag) * noise_scale
        )
        x = x / torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=-1, keepdim=True))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, length)
        x = torch.unsqueeze(x, -1)                # (B, C, L, 1)
        x = self.patcher(x)                       # (B, C*k, num_patches)
        bs, ch_ker, l = x.shape
        x = x.permute(0, 2, 1).contiguous().view(bs * l, ch_ker)  # (B*L, C*k)

        # 编码 -> 量子态
        x = self.qubit_norm(x).to(torch.cfloat)   # (B*L, 2^W)
        unitaries = self.unitary_op(self.weight, self.phi)  # (F, 2^W, 2^W)

        x = x.unsqueeze(dim=1)                    # (B*L, 1, 2^W)
        x = torch.einsum('bik,fjk->bfij', x, unitaries)  # (B*L, F, 2^W, 2^W)

        if self.noise:
            x = self.apply_depolarizing_noise(x)
        if self.shots > 0:
            x = self.adjust_probabilities_with_shots(x)

        obs = self.measurement_op()               # (W, 2^W, 2^W)
        intermediate = torch.einsum('bfij,wjk->bfwik', x.conj(), obs)  # (B*L, F, W, 2^W)
        x = torch.einsum('bfwik,bfik->bfw', intermediate, x)           # (B*L, F, W)
        x = x.real.contiguous().view(bs * l, -1)   # (B*L, F*W)

        if self.bias is not None:
            x = x[:, :self.out_channels] + self.bias
        else:
            x = x[:, :self.out_channels]

        x = x.contiguous().view(bs, l, self.out_channels).permute(0, 2, 1)  # (B, C_out, L_out)
        return x


# ===================== 5. 完整的 FQN 模型 =====================
class FQN(nn.Module):
    """
    完全量子卷积网络，仅由 Quanv1D 层 + BN + ReLU 构成。
    用于时间序列分类。
    """
    def __init__(self, num_channels: int, num_classes: int, device: torch.device,
                 dim: int = 16, depth: int = 3, input_window: int = 15,
                 input_scale: int = 2, hidden_window: int = 5):
        super().__init__()
        self.input = Quanv1d(
            in_channels=num_channels, out_channels=dim,
            kernel_size=input_window, padding=(input_window - 1) // 2,
            stride=input_scale, dilation=1, device=device
        )

        def quanv_bn_relu(in_ch, out_ch, k, d):
            return nn.Sequential(
                Quanv1d(in_channels=in_ch, out_channels=out_ch,
                        kernel_size=k, padding=0, stride=1, dilation=d, device=device),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            )

        self.hidden_layers = nn.Sequential(
            *[quanv_bn_relu(dim, dim, hidden_window, i+1) for i in range(depth)]
        )

        self.output = Quanv1d(
            in_channels=dim, out_channels=num_classes,
            kernel_size=1, padding=0, stride=1, dilation=1, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        x = self.hidden_layers(x)
        x = self.output(x)
        x = F.adaptive_avg_pool1d(x, 1)
        return x.view(x.size(0), -1)


# ===================== 6. 辅助工具函数 =====================
def set_quanv_noise(model: nn.Module, shots: int, noise: bool) -> None:
    """递归设置模型中所有 Quanv1D 层的 shots 和 noise 属性"""
    for module in model.modules():
        if hasattr(module, 'shots'):
            module.shots = shots
        if hasattr(module, 'noise'):
            module.noise = noise