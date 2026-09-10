from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from project_build.core.registry import POTENTIAL_REGISTRY
from project_build.physics.potentials.base_potential import BasePotential

DEFAULT_COVALENT_RADII = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    53: 1.39,  # I
}

DEFAULT_IDEAL_VALENCE = {
    1: 1.0,   # H
    6: 4.0,   # C
    7: 3.0,   # N
    8: 2.0,   # O
    9: 1.0,   # F
    15: 3.0,  # P
    16: 2.0,  # S
    17: 1.0,  # Cl
    35: 1.0,  # Br
    53: 1.0,  # I
}

@POTENTIAL_REGISTRY.register("ContinuousValencyPotential")
class ContinuousValencyPotential(BasePotential):
    """
    连续可微长程量子价态势能场 (自动设备自适应对齐版)。
    """
    def __init__(
        self,
        scale: float = 1.0,
        bond_sigma: float = 0.40,
        custom_radii: Optional[Dict[int, float]] = None,
        custom_valences: Optional[Dict[int, float]] = None
    ) -> None:
        super().__init__(name="ContinuousValencyPotential")
        self.scale: float = scale
        self.bond_sigma: float = bond_sigma
        
        radii_dict = DEFAULT_COVALENT_RADII.copy()
        if custom_radii is not None:
            radii_dict.update(custom_radii)
            
        valences_dict = DEFAULT_IDEAL_VALENCE.copy()
        if custom_valences is not None:
            valences_dict.update(custom_valences)
            
        max_z = 100
        radii_tensor = torch.zeros(max_z + 1, dtype=torch.float32)
        valence_tensor = torch.zeros(max_z + 1, dtype=torch.float32)
        
        for z, r in radii_dict.items():
            if z <= max_z:
                radii_tensor[z] = float(r)
                
        for z, v in valences_dict.items():
            if z <= max_z:
                valence_tensor[z] = float(v)
                
        self.register_buffer("covalent_radii", radii_tensor)
        self.register_buffer("ideal_valences", valence_tensor)

    def compute_energy(
        self,
        pos: torch.Tensor,
        atom_types: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        B, N, _ = pos.shape
        device = pos.device
        
        # 自动设备对齐保护: 确保查找表与输入坐标在同一 GPU/CPU 设备上
        radii_table = self.covalent_radii.to(device)
        valences_table = self.ideal_valences.to(device)
        
        atom_types_clamped = torch.clamp(atom_types.long(), min=0, max=100)
        r_cov = radii_table[atom_types_clamped]       # [B, N]
        v_ideal = valences_table[atom_types_clamped]   # [B, N]
        
        r_ideal_matrix = r_cov.unsqueeze(2) + r_cov.unsqueeze(1) # [B, N, N]
        
        diff = pos.unsqueeze(2) - pos.unsqueeze(1) # [B, N, N, 3]
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8) # [B, N, N]
        
        delta_r = (dist - r_ideal_matrix) / self.bond_sigma
        bond_order_matrix = 2.0 * torch.sigmoid(-delta_r) # [B, N, N]
        
        eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(B, -1, -1)
        bond_order_matrix = bond_order_matrix.masked_fill(eye_mask, 0.0)
        
        if mask is not None:
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            bond_order_matrix = bond_order_matrix * pair_mask.float()
            
        v_apparent = torch.sum(bond_order_matrix, dim=-1) # [B, N]
        valency_error_sq = (v_apparent - v_ideal) ** 2 # [B, N]
        
        if mask is not None:
            valency_error_sq = valency_error_sq * mask.float()
            
        molecule_valency_energy = self.scale * torch.sum(valency_error_sq, dim=-1) # [B]
        return molecule_valency_energy
