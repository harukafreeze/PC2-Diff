import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from project_build.core.registry import BACKBONE_REGISTRY
from project_build.models.backbones.egnn_backbone import SinusoidalTimeEmbedding
from project_build.models.backbones.target_sbdd_layer import LocalRadialSBDDLayer

@BACKBONE_REGISTRY.register("TargetSBDDBackbone")
class TargetSBDDBackbone(nn.Module):
    """
    Project BUILD 3.0 - 局域 RBF 径向跨注意力靶点 SBDD 主干网络。
    """
    def __init__(
        self,
        num_lig_types: int = 9,
        num_pocket_types: int = 6,
        max_lig_atoms: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        cutoff_radius: float = 8.0,
        physics_bias_cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.num_lig_types = num_lig_types
        self.num_atom_types = num_lig_types
        self.num_pocket_types = num_pocket_types
        self.hidden_dim = hidden_dim
        
        # 标量嵌入层
        self.lig_embed = nn.Linear(num_lig_types, hidden_dim)
        self.lig_pos_embed = nn.Embedding(max_lig_atoms, hidden_dim)
        self.pocket_embed = nn.Linear(num_pocket_types, hidden_dim)
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 堆叠 L 层局域 RBF 径向跨注意力层
        self.layers = nn.ModuleList([
            LocalRadialSBDDLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                cutoff_radius=cutoff_radius,
                physics_bias_cfg=physics_bias_cfg
            )
            for _ in range(num_layers)
        ])
        
        # 1. 离散原子分类预测头
        self.head_h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_lig_types)
        )
        
        # 2. 显式化学键连通概率预测头
        self.head_bond = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        t: torch.Tensor,
        pocket_pos: Optional[torch.Tensor] = None,
        pocket_h: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None,
        pocket_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N_l, _ = x_t.shape
        device = x_t.device
        
        if lig_mask is None and mask is not None:
            lig_mask = mask
            
        if pocket_pos is None:
            pocket_pos = torch.zeros(B, 1, 3, device=device)
            pocket_h = torch.zeros(B, 1, 6, device=device)
            
        node_indices = torch.arange(N_l, device=device).unsqueeze(0).expand(B, -1)
        pos_id_feat = self.lig_pos_embed(node_indices)
        
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, N_l, -1)
        h_lig_curr = self.lig_embed(h_t) + pos_id_feat + t_emb
        h_pocket_curr = self.pocket_embed(pocket_h)
        pos_lig_curr = x_t
        
        # 逐层局域 RBF 径向等变消息传递
        for layer in self.layers:
            h_lig_curr, pos_lig_curr = layer(
                h_lig=h_lig_curr,
                pos_lig=pos_lig_curr,
                h_pocket=h_pocket_curr,
                pos_pocket=pocket_pos,
                lig_mask=lig_mask,
                pocket_mask=pocket_mask
            )
            
        # 预测标准的绝对去噪位移场 eps_x (绝对不执行中心剥离)
        eps_x = pos_lig_curr - x_t
        eps_h = self.head_h(h_lig_curr)
        
        # 显式计算配体内部成对化学键 Logits
        h_i = h_lig_curr.unsqueeze(2).expand(-1, -1, N_l, -1)
        h_j = h_lig_curr.unsqueeze(1).expand(-1, N_l, -1, -1)
        diff = pos_lig_curr.unsqueeze(2) - pos_lig_curr.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1, keepdim=True) + 1e-8)
        
        bond_feat = torch.cat([h_i, h_j, dist], dim=-1)
        pred_bond_logits = self.head_bond(bond_feat).squeeze(-1)
        pred_bond_logits = 0.5 * (pred_bond_logits + pred_bond_logits.transpose(1, 2))
        
        return eps_x, eps_h, pred_bond_logits
