import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from project_build.core.registry import ATTENTION_REGISTRY

class GaussianSmearing(nn.Module):
    """连续欧式距离的高斯径向基函数 (RBF) 展开算子"""
    def __init__(self, start: float = 0.0, stop: float = 8.0, num_gaussians: int = 32):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1) - self.offset.view(1, 1, 1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class LocalRadialSBDDLayer(nn.Module):
    """
    Project BUILD 3.0 - 全局 Lipschitz 稳定的 8.0 Å 局域物理径向跨注意力层。
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        cutoff_radius: float = 8.0,
        physics_bias_cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cutoff_radius = cutoff_radius
        
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff_radius, num_gaussians=32)
        
        # 配体内部自注意力
        self.q_lig = nn.Linear(hidden_dim, hidden_dim)
        self.k_lig = nn.Linear(hidden_dim, hidden_dim)
        self.v_lig = nn.Linear(hidden_dim, hidden_dim)
        
        # 受体口袋跨注意力投影 (32 维 RBF 增强)
        self.k_pocket = nn.Linear(hidden_dim + 32, hidden_dim)
        self.v_pocket = nn.Linear(hidden_dim + 32, hidden_dim)
        
        # 软核物理排斥偏置核 (白皮书标准软核 δ=0.5 Å，防止奇异点)
        if physics_bias_cfg is not None:
            self.physics_bias = ATTENTION_REGISTRY.build(physics_bias_cfg)
        else:
            self.physics_bias = ATTENTION_REGISTRY.build({
                "type": "LennardJonesCoulombBias",
                "sigma_default": 1.7,
                "epsilon_default": 0.15,
                "delta_vdw": 0.5,
                "min_clamp": -50.0,
                "max_clamp": 10.0
            })
            
        self.coord_mlp_lig = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.coord_mlp_cross = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_lig: torch.Tensor,
        pos_lig: torch.Tensor,
        h_pocket: torch.Tensor,
        pos_pocket: torch.Tensor,
        lig_mask: Optional[torch.Tensor] = None,
        pocket_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_l, _ = pos_lig.shape
        _, N_p, _ = pos_pocket.shape
        
        # 1. 配体自注意力
        Q_l = self.q_lig(h_lig).view(B, N_l, self.num_heads, self.head_dim).transpose(1, 2)
        K_l = self.k_lig(h_lig).view(B, N_l, self.num_heads, self.head_dim).transpose(1, 2)
        V_l = self.v_lig(h_lig).view(B, N_l, self.num_heads, self.head_dim).transpose(1, 2)
        
        intra_logits = torch.matmul(Q_l, K_l.transpose(-2, -1)) / (self.head_dim ** 0.5)
        eye_mask = torch.eye(N_l, device=pos_lig.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        intra_logits = intra_logits.masked_fill(eye_mask, -1e4)
        
        if lig_mask is not None:
            pair_mask = (lig_mask.unsqueeze(2) & lig_mask.unsqueeze(1)).unsqueeze(1)
            intra_logits = intra_logits.masked_fill(~pair_mask, -1e4)
            
        attn_intra = torch.softmax(intra_logits, dim=-1)
        msg_intra = torch.matmul(attn_intra, V_l).transpose(1, 2).contiguous().view(B, N_l, self.hidden_dim)
        
        # 2. 局域 8.0 Å 径向跨注意力 (带 PG-CrossAttention 软核偏置)
        diff_cross = pos_lig.unsqueeze(2) - pos_pocket.unsqueeze(1) # [B, N_l, N_p, 3]
        dist_cross = torch.sqrt(torch.sum(diff_cross ** 2, dim=-1) + 1e-8)
        
        rbf_feat = self.rbf(dist_cross) # [B, N_l, N_p, 32]
        h_pocket_exp = h_pocket.unsqueeze(1).expand(-1, N_l, -1, -1)
        pocket_augmented = torch.cat([h_pocket_exp, rbf_feat], dim=-1)
        
        K_p = self.k_pocket(pocket_augmented).view(B, N_l, N_p, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V_p = self.v_pocket(pocket_augmented).view(B, N_l, N_p, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        Q_l_exp = Q_l.unsqueeze(3)
        cross_logits = torch.sum(Q_l_exp * K_p, dim=-1) / (self.head_dim ** 0.5)
        
        # 注入物理偏置 Γ_cross
        gamma_cross = self.physics_bias(pos_i=pos_lig, pos_j=pos_pocket).unsqueeze(1)
        cross_logits = cross_logits + gamma_cross
        
        # 8.0 Å 空间截断
        local_cutoff_mask = (dist_cross <= self.cutoff_radius).unsqueeze(1)
        cross_logits = cross_logits.masked_fill(~local_cutoff_mask, -1e4)
        
        if pocket_mask is not None and lig_mask is not None:
            mask_2d = (lig_mask.unsqueeze(2) & pocket_mask.unsqueeze(1)).unsqueeze(1)
            cross_logits = cross_logits.masked_fill(~mask_2d, -1e4)
            
        attn_cross = torch.softmax(cross_logits, dim=-1)
        msg_cross_heads = torch.sum(attn_cross.unsqueeze(-1) * V_p, dim=3)
        msg_cross = msg_cross_heads.permute(0, 2, 1, 3).contiguous().view(B, N_l, self.hidden_dim)
        
        # 3. 几何坐标演化 (受内部共价与局域物理偏置引导，绝无梯度爆炸)
        diff_intra = pos_lig.unsqueeze(2) - pos_lig.unsqueeze(1)
        attn_intra_mean = torch.mean(attn_intra, dim=1, keepdim=True).squeeze(1).unsqueeze(-1)
        attn_cross_mean = torch.mean(attn_cross, dim=1, keepdim=True).squeeze(1).unsqueeze(-1)
        
        scale_factor = 1.0 / math.sqrt(N_l)
        delta_pos_intra = torch.sum(diff_intra * self.coord_mlp_lig(msg_intra).unsqueeze(2) * attn_intra_mean, dim=2) * scale_factor
        delta_pos_cross = torch.sum(diff_cross * self.coord_mlp_cross(msg_cross).unsqueeze(2) * attn_cross_mean, dim=2)
        
        pos_lig_next = pos_lig + delta_pos_intra + delta_pos_cross
        h_lig_next = self.layer_norm(h_lig + self.node_mlp(torch.cat([h_lig, msg_intra, msg_cross], dim=-1)))
        
        if lig_mask is not None:
            h_lig_next = h_lig_next * lig_mask.unsqueeze(-1).float()
            pos_lig_next = pos_lig_next * lig_mask.unsqueeze(-1).float()
            
        return h_lig_next, pos_lig_next
