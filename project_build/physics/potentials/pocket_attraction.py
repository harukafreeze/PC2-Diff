from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from project_build.core.registry import POTENTIAL_REGISTRY
from project_build.physics.potentials.base_potential import BasePotential

@POTENTIAL_REGISTRY.register("PocketSurfaceAttractionPotential")
class PocketSurfaceAttractionPotential(BasePotential):
    """
    三维蛋白质口袋药效团引导势能场 (修复维度对齐版)。
    """
    def __init__(
        self,
        scale: float = 1.0,
        hbond_ideal: float = 2.8,
        hydro_ideal: float = 3.5,
        clash_cutoff: float = 1.5
    ) -> None:
        super().__init__(name="PocketSurfaceAttractionPotential")
        self.scale: float = scale
        self.hbond_ideal: float = hbond_ideal
        self.hydro_ideal: float = hydro_ideal
        self.clash_cutoff: float = clash_cutoff

    def compute_energy(
        self,
        pos: torch.Tensor,               # 配体坐标 [B, N_l, 3]
        atom_types: torch.Tensor,        # 配体原子类型 [B, N_l, 9] 或 [B, N_l]
        pocket_pos: Optional[torch.Tensor] = None, # [B, N_p, 3]
        pocket_h: Optional[torch.Tensor] = None,   # [B, N_p, 6]
        lig_mask: Optional[torch.Tensor] = None,
        pocket_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        if pocket_pos is None or pocket_h is None:
            return torch.zeros(pos.shape[0], device=pos.device)
            
        B, N_l, _ = pos.shape
        _, N_p, _ = pocket_pos.shape
        
        diff = pos.unsqueeze(2) - pocket_pos.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        
        # 强制将 One-Hot/Logits 降维为纯标量索引 [B, N]
        if pocket_h.ndim == 3:
            pocket_z = torch.argmax(pocket_h, dim=-1)
        else:
            pocket_z = pocket_h
            
        if atom_types.ndim == 3:
            lig_z = torch.argmax(atom_types, dim=-1)
        else:
            lig_z = atom_types
            
        # 极性 (N=1, O=2, F=3 / N=2, O=3) 与疏水掩码
        pocket_polar_mask = (pocket_z == 2) | (pocket_z == 3)
        pocket_hydro_mask = (pocket_z == 1) | (pocket_z == 4)
        
        lig_polar_mask = (lig_z == 1) | (lig_z == 2) | (lig_z == 3)
        lig_hydro_mask = (lig_z == 0) | (lig_z == 5)
        
        hbond_pair_mask = (lig_polar_mask.unsqueeze(2) & pocket_polar_mask.unsqueeze(1)).float()
        hbond_dev = (dist - self.hbond_ideal) ** 2
        e_hbond = torch.sum(hbond_dev * hbond_pair_mask, dim=[-1, -2])
        
        hydro_pair_mask = (lig_hydro_mask.unsqueeze(2) & pocket_hydro_mask.unsqueeze(1)).float()
        hydro_dev = (dist - self.hydro_ideal) ** 2
        e_hydro = torch.sum(hydro_dev * hydro_pair_mask, dim=[-1, -2])
        
        clash_mask = (dist < self.clash_cutoff).float()
        e_clash = torch.sum(((self.clash_cutoff - dist) ** 2) * clash_mask, dim=[-1, -2])
        
        total_pharm_energy = self.scale * (0.5 * e_hbond + 0.2 * e_hydro + 10.0 * e_clash) / (N_l * 1.0)
        return total_pharm_energy
