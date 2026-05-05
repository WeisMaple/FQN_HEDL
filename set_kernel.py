"""
集合核池化 + HEDL 投影

方案二核心模块：
- SetKernelPooling: 逐帧扫描 Quanv1d(dim→K) 输出, 显式计算每个超域集合的证据
- HEDLProjection: 固定 W^P 矩阵, 将集合证据投影为 Dirichlet alpha
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations, chain


def generate_hyper_domain_sets(K: int, max_set_size: int = None):
    """生成所有非空非全集的子集二进制掩码。

    Args:
        K: 类别数
        max_set_size: 限制最大集合大小。None 表示不限制 (K≤10),
                      K>10 时建议设为 2。

    Returns:
        masks: (N_sets, K) bool tensor
    """
    all_classes = list(range(K))
    masks = []

    if max_set_size is None:
        max_size = K - 1  # exclude full set
    else:
        max_size = min(max_set_size, K - 1)

    for size in range(1, max_size + 1):
        for subset in combinations(all_classes, size):
            mask = torch.zeros(K, dtype=torch.bool)
            mask[list(subset)] = True
            masks.append(mask)

    return torch.stack(masks, dim=0)  # (N_sets, K)


class SetKernelPooling(nn.Module):
    """集合核池化: (N, K, T) per-frame class probabilities → (N, N_sets) evidence.

    evidence_s = Σ_t (Π_{k∈s} p_{t,k})^(1/|s|)

    RBF Prototype Activation (方案E):
        p_{t,k} = exp( -||x_t - c_k||^2 / (2 τ^2) )
        c_k = 第 k 类在训练集上的 backbone 输出均值向量 (K维)
        对 ID: 帧向量接近正确类原型 → p_correct≈1, p_other≈0 → 证据集中
        对 OOD: 帧向量远离所有原型 → 所有 p≈0 → 证据趋零 → 高 vacuity

    对数空间计算防止下溢:
        log_geom = mean_{k∈s}(log(p_{t,k}))   # 几何平均的 log
        e_s = Σ_t exp(log_geom)               # 求和回到线性空间
    """

    def __init__(self, K: int, max_set_size: int = None,
                 eps: float = 1e-8, tau_init: float = 0.5):
        super().__init__()
        self.K = K
        self.eps = eps

        masks = generate_hyper_domain_sets(K, max_set_size)
        self.register_buffer('masks', masks)  # (N_sets, K)
        set_sizes = masks.sum(dim=1).float()  # (N_sets,)
        self.register_buffer('set_sizes', set_sizes)

        self.tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))

        # 类别原型: prototype[k, j] = channel j 的期望输出 given class k
        self.register_buffer('prototypes', torch.zeros(K, K))

        # τ 参考值: 从训练数据校准得到, 用于弱正则防止 EDL 推偏 τ
        self.register_buffer('tau_ref', torch.tensor(tau_init, dtype=torch.float32))

    @property
    def N_sets(self):
        return len(self.masks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, K, T) — Quanv1d(dim→K) 输出, 未经 softmax.

        RBF per-class activation: p_{t,k} = exp(-||x_t - c_k||^2 / (2 τ^2))
        - 每帧输出 K 维向量, 与 K 个类别原型比较 L2 距离
        - ID: 帧向量接近正确类别原型 → p_correct 高, 其他低
        - OOD: 帧向量远离所有原型 → 所有 p_k 低

        集合证据: evidence_s = Σ_t min_{k∈s} p_{t,k}
        - |s|=1: 退化为单类证据, D1 不受影响
        - |s|>1: min 抑制 ID 的高-低组合, 消除几何平均对均匀 OOD 的奖励
        """
        tau = self.tau.abs().clamp(min=0.05)
        # x_flat: (N, T, K) — each frame is a K-dim vector
        x_flat = x.permute(0, 2, 1)  # (N, T, K)
        # prototypes: (K, K) → (1, 1, K, K)
        proto = self.prototypes.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
        # dist_sq[n, t, k] = Σ_j (x[n,j,t] - proto[k,j])^2
        dist_sq = ((x_flat.unsqueeze(2) - proto) ** 2).sum(dim=3)  # (N, T, K)
        p_per_class = torch.exp(-dist_sq / (2 * tau ** 2))  # (N, T, K)
        p = p_per_class.permute(0, 2, 1).clamp(min=self.eps)  # (N, K, T)

        evidence_list = []
        for i in range(self.N_sets):
            mask = self.masks[i]       # (K,)
            p_masked = p[:, mask, :]   # (N, |s|, T)
            # min 聚合: |s|=1 退化为恒等, |s|>1 抑制高-低组合
            e_s = p_masked.min(dim=1).values.sum(dim=1)  # (N,)
            evidence_list.append(e_s)

        return torch.stack(evidence_list, dim=1)  # (N, N_sets)

    def extra_repr(self):
        return f"K={self.K}, N_sets={self.N_sets}, tau={self.tau.item():.3f}"


class HEDLProjection(nn.Module):
    """HEDL 投影: (N, N_sets) evidence → (N, K) alpha.

    W^P_{k,s} = 1/|s| if k ∈ s else 0  (原论文公式(6) 无先验信息版)
    α = e^H @ W^P^T + W/K
    """

    def __init__(self, K: int, masks: torch.Tensor, W: float = 2.0):
        super().__init__()
        self.K = K
        self.W = W

        # masks: (N_sets, K) → W_P: (K, N_sets)
        W_P = masks.float() / masks.sum(dim=1, keepdim=True).float().clamp(min=1)
        self.register_buffer('W_P', W_P.T)  # (K, N_sets)

    @property
    def N_sets(self):
        return self.W_P.shape[1]

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """evidence: (N, N_sets) → alpha: (N, K)"""
        alpha = evidence @ self.W_P.T + self.W / self.K
        return alpha

    def extra_repr(self):
        return f"K={self.K}, N_sets={self.N_sets}, W={self.W}"
