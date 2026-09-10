from typing import Optional, Tuple
import torch
import torch.nn as nn
from project_build.core.registry import ATTENTION_REGISTRY

class EquivariantPhysicsGraphLayer(nn.Module):
    """
    高响应度 3D 等变物理图变换器层 (High-Response EGNN Layer)。
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 0,
        num_heads: int = 4,
        act_fn: str = "silu",
        coord_clamp: float = 100.0,
        physics_bias_cfg: Optional[dict] = None
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        act = nn.SiLU() if act_fn == "silu" else nn.GELU()
        
        in_edge_feat = hidden_dim * 2 + 1 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_feat, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act
        )
        
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        
        if physics_bias_cfg is not None:
            self.physics_bias = ATTENTION_REGISTRY.build(physics_bias_cfg)
        else:
            self.physics_bias = ATTENTION_REGISTRY.build({"type": "LennardJonesCoulombBias"})
            
        coord_linear = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(coord_linear.weight)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            coord_linear
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        charges: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = pos.shape
        
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)
        dist_feat = torch.log(1.0 + dist_sq)
        
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        
        edge_inputs = [h_i, h_j, dist_feat]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
            
        edge_feat = torch.cat(edge_inputs, dim=-1)
        m_ij = self.edge_mlp(edge_feat)
        
        Q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        data_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        gamma = self.physics_bias(pos_i=pos, pos_j=pos, charges_i=charges, charges_j=charges)
        gamma_expanded = gamma.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        total_logits = data_logits + gamma_expanded
        
        eye_mask = torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        total_logits = total_logits.masked_fill(eye_mask, -1e4)
        
        if mask is not None:
            pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(1)
            total_logits = total_logits.masked_fill(~pair_mask, -1e4)
            
        attn_weights = torch.softmax(total_logits, dim=-1)
        attn_weights_mean = torch.mean(attn_weights, dim=1, keepdim=True)
        
        # 释放坐标位移梯度
        coord_scale = self.coord_mlp(m_ij)
        modulated_diff = diff * coord_scale * attn_weights_mean.squeeze(1).unsqueeze(-1)
        
        if mask is not None:
            pair_mask_3d = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1)
            modulated_diff = modulated_diff * pair_mask_3d.float()
            
        pos_update = torch.sum(modulated_diff, dim=2)
        pos_next = pos + pos_update
        
        agg_message = torch.sum(m_ij * attn_weights_mean.squeeze(1).unsqueeze(-1), dim=2)
        node_update = self.node_mlp(torch.cat([h, agg_message], dim=-1))
        h_next = self.layer_norm(h + node_update)
        
        if mask is not None:
            h_next = h_next * mask.unsqueeze(-1).float()
            pos_next = pos_next * mask.unsqueeze(-1).float()
            
        return h_next, pos_next
