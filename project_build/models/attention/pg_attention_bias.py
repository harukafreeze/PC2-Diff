from typing import Optional
import torch
import torch.nn as nn
from project_build.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("LennardJonesCoulombBias")
class LennardJonesCoulombBias(nn.Module):
    """
    物理引导注意力偏置计算引擎 (PG-Attention Kernel)。
    
    将三维微观统计力学势能映射为 Transformer 结构的先验注意力 Logits 偏置:
        Γ_ij = - 1 / (k_B T) * [ V_vdw(d_ij) + V_elec(d_ij) ]
        
    数学特性:
        1. 严格满足 SO(3) 旋转不变性与 T(3) 平移不变性；
        2. 当 d_ij -> 0 时，范德华排斥项急剧发散，Γ_ij -> -∞，通过 Softmax 机制在底层硬剪枝原子碰撞；
        3. 引入平滑软核正则项 (Soft-core δ)，保证势能 Hessian 矩阵全局有界，杜绝数值下溢与奇异点。
    """
    def __init__(
        self,
        kb_t: float = 0.6,
        sigma_default: float = 1.5,
        epsilon_default: float = 0.1,
        delta_vdw: float = 0.5,
        delta_elec: float = 0.5,
        min_clamp: float = -50.0,
        max_clamp: float = 10.0
    ) -> None:
        super().__init__()
        if kb_t <= 0.0:
            raise ValueError(f"热力学标量尺度 kb_t 必须严格大于 0，当前为: {kb_t}")
            
        self.kb_t: float = kb_t
        self.sigma_default: float = sigma_default
        self.epsilon_default: float = epsilon_default
        self.min_clamp: float = min_clamp
        self.max_clamp: float = max_clamp
        
        # 软核平滑截断参数 (设为可训练参数以支持数据驱动弹性微调)
        self.delta_vdw = nn.Parameter(torch.tensor(float(delta_vdw), dtype=torch.float32))
        self.delta_elec = nn.Parameter(torch.tensor(float(delta_elec), dtype=torch.float32))

    def forward(
        self,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        charges_i: Optional[torch.Tensor] = None,
        charges_j: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算坐标集合 i 与坐标集合 j 之间的物理注意力偏置矩阵。
        
        参数:
            pos_i: [B, N_i, 3] Query 空间点坐标 (如配体分子原子坐标)
            pos_j: [B, N_j, 3] Key 空间点坐标 (如配体自身或蛋白质口袋残基坐标)
            charges_i: [B, N_i] Query 点的形式电荷 (可选)
            charges_j: [B, N_j] Key 点的形式电荷 (可选)
            mask: [B, N_i, N_j] 填充掩码
            
        返回:
            gamma_bias: [B, N_i, N_j] 物理注意力偏置矩阵
        """
        if pos_i.ndim != 3 or pos_j.ndim != 3:
            raise ValueError(f"pos_i 与 pos_j 必须为 3 维张量 [B, N, 3]，当前分别为 {pos_i.shape} 与 {pos_j.shape}")
            
        B, N_i, _ = pos_i.shape
        _, N_j, _ = pos_j.shape
        
        # 1. 计算相对位移向量场并提取欧氏距离平方 (严格保持 SO(3) 不变性)
        diff = pos_i.unsqueeze(2) - pos_j.unsqueeze(1) # [B, N_i, N_j, 3]
        dist_sq = torch.sum(diff ** 2, dim=-1)         # [B, N_i, N_j]
        
        # 2. 计算软核平滑范德华势 (Soft-core 12-6 Lennard-Jones)
        delta_vdw_sq = torch.clamp(self.delta_vdw ** 2, min=1e-4)
        r_soft_sq = dist_sq + delta_vdw_sq
        
        ratio = (self.sigma_default ** 2) / r_soft_sq
        lj_att = ratio ** 3
        lj_rep = lj_att ** 2
        v_vdw = 4.0 * self.epsilon_default * (lj_rep - lj_att)
        
        # 3. 计算屏蔽静电库仑势 (Screened Coulomb)
        if charges_i is not None and charges_j is not None:
            delta_elec_safe = torch.clamp(torch.abs(self.delta_elec), min=1e-4)
            dist_safe = torch.sqrt(dist_sq + 1e-8)
            charge_prod = charges_i.unsqueeze(2) * charges_j.unsqueeze(1)
            v_elec = charge_prod / (dist_safe + delta_elec_safe)
        else:
            v_elec = torch.zeros_like(v_vdw)
            
        # 4. 映射为对数几率偏置: Γ = - (V_vdw + V_elec) / (k_B * T)
        total_potential = v_vdw + v_elec
        gamma = - total_potential / self.kb_t
        
        # 5. 数值安全截断，防止梯度爆炸或下溢
        gamma_clamped = torch.clamp(gamma, min=self.min_clamp, max=self.max_clamp)
        
        # 6. 掩码处理
        if mask is not None:
            gamma_clamped = gamma_clamped.masked_fill(~mask, float("-inf"))
            
        return gamma_clamped

    def extra_repr(self) -> str:
        return (
            f"kb_t={self.kb_t}, sigma={self.sigma_default}, "
            f"epsilon={self.epsilon_default}, min_clamp={self.min_clamp}, max_clamp={self.max_clamp}"
        )
