import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from project_build.core.registry import BACKBONE_REGISTRY
from project_build.models.backbones.egnn_layer import EquivariantPhysicsGraphLayer

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000.0) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

@BACKBONE_REGISTRY.register("DualSpaceEGNNBackbone")
class DualSpaceEGNNBackbone(nn.Module):
    """
    集成拓扑位置编码的等变图扩散主干 (Topological Identity EGNN)。
    """
    def __init__(
        self,
        num_atom_types: int = 5,
        max_atoms: int = 100, # 支持最大原子数量的位置编码
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        act_fn: str = "silu",
        coord_clamp: float = 10.0,
        physics_bias_cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. 离散原子类型嵌入 + 拓扑位置编码 (破除置换二义性)
        self.atom_embed = nn.Linear(num_atom_types, hidden_dim)
        self.node_pos_embed = nn.Embedding(max_atoms, hidden_dim)
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            EquivariantPhysicsGraphLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                act_fn=act_fn,
                coord_clamp=coord_clamp,
                physics_bias_cfg=physics_bias_cfg
            )
            for _ in range(num_layers)
        ])
        
        self.head_h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_t: torch.Tensor,
        t: torch.Tensor,
        charges: Optional[torch.Tensor] = None,
        node_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x_t.shape
        device = x_t.device
        
        # 1. 构造拓扑身份特征 (破除相同元素的同构对称性)
        if node_indices is None:
            node_indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            
        pos_id_feat = self.node_pos_embed(node_indices) # [B, N, hidden_dim]
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)
        
        h_curr = self.atom_embed(h_t) + pos_id_feat + t_emb
        pos_curr = x_t
        
        # 2. 等变消息传递
        for layer in self.layers:
            h_curr, pos_curr = layer(
                h=h_curr,
                pos=pos_curr,
                charges=charges,
                mask=mask
            )
            
        x0_hat = pos_curr
        h0_hat = self.head_h(h_curr)
        
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            x0_hat = x0_hat * mask_float
            num_nodes = torch.clamp(torch.sum(mask_float, dim=1, keepdim=True), min=1.0)
            mean_x0 = torch.sum(x0_hat, dim=1, keepdim=True) / num_nodes
            x0_hat = (x0_hat - mean_x0) * mask_float
            h0_hat = h0_hat * mask.unsqueeze(-1).float()
        else:
            mean_x0 = torch.mean(x0_hat, dim=1, keepdim=True)
            x0_hat = x0_hat - mean_x0
            
        return x0_hat, h0_hat
