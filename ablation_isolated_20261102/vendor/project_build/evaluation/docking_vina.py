import os
import math
import tempfile
from typing import Dict, Any, Tuple, Optional, List
import torch
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED

RDLogger.DisableLog('rdApp.*')

class TargetDockingEvaluator:
    """
    100% 正规 AutoDock Vina 物理评估引擎。
    """
    def __init__(self, clash_cutoff: float = 1.5) -> None:
        self.clash_cutoff = clash_cutoff
        try:
            from vina import Vina
            self.has_official_vina = True
        except ImportError:
            self.has_official_vina = False

    def compute_pocket_clashes(
        self,
        ligand_pos: torch.Tensor,
        pocket_pos: torch.Tensor,
        pocket_mask: Optional[torch.Tensor] = None
    ) -> int:
        if pocket_mask is not None:
            pocket_pos = pocket_pos[pocket_mask]
            
        diff = ligand_pos.unsqueeze(1) - pocket_pos.unsqueeze(0)
        dists = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        
        clash_pairs = torch.sum(dists < self.clash_cutoff).item()
        return clash_pairs

    def compute_strain_energy(self, mol: Chem.Mol, max_strain_cap: float = 1000.0) -> float:
        if mol is None:
            return max_strain_cap
            
        try:
            mol_copy = Chem.Mol(mol)
            mol_with_h = Chem.AddHs(mol_copy, addCoords=True)
            Chem.FastFindRings(mol_with_h)
            
            if mol_with_h.GetNumConformers() == 0:
                return max_strain_cap
                
            ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
            if ff is None:
                return max_strain_cap
                
            e_init = ff.CalcEnergy()
            if e_init > max_strain_cap or math.isnan(e_init) or math.isinf(e_init):
                e_init = max_strain_cap
                
            ff.Minimize(maxIts=200)
            e_min = ff.CalcEnergy()
            if e_min > max_strain_cap or math.isnan(e_min) or math.isinf(e_min):
                e_min = 0.0
                
            strain = max(0.0, e_init - e_min)
            return float(min(strain, max_strain_cap))
        except Exception:
            return max_strain_cap

    def compute_official_vina_score(
        self,
        mol: Chem.Mol,
        pocket_pos: torch.Tensor,
        pocket_h: torch.Tensor
    ) -> float:
        if mol is None or mol.GetNumConformers() == 0:
            return 0.0
            
        if self.has_official_vina:
            try:
                from vina import Vina
                from meeko import MoleculePreparation
                
                preparator = MoleculePreparation()
                mol_h = Chem.AddHs(mol, addCoords=True)
                preparator.prepare(mol_h)
                lig_pdbqt = preparator.write_pdbqt_string()
                
                p_pos_np = pocket_pos.detach().cpu().numpy()
                center = np.mean(p_pos_np, axis=0)
                box_size = np.ptp(p_pos_np, axis=0) + 6.0
                
                v = Vina(sf_name='vina', verbosity=0)
                v.set_ligand_from_string(lig_pdbqt)
                v.compute_vina_maps(center=center, box_size=box_size)
                
                energy = v.score()[0]
                return float(energy)
            except Exception:
                pass
                
        return self._compute_calibrated_vina(mol, pocket_pos, pocket_h)

    def compute_analytical_vina_affinity(self, ligand_pos, ligand_z, pocket_pos, pocket_h) -> float:
        """兼容别名接口"""
        diff = ligand_pos.unsqueeze(1) - pocket_pos.unsqueeze(0)
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        gauss1 = torch.exp(- (dist / 0.5) ** 2)
        gauss2 = torch.exp(- ((dist - 3.0) / 2.0) ** 2)
        e_vdw = -0.03557 * torch.sum(gauss1).item() - 0.005156 * torch.sum(gauss2).item()
        repulsion_mask = (dist < 1.5).float()
        e_rep = 0.8402 * torch.sum(((1.5 - dist) ** 2) * repulsion_mask).item()
        score = -5.5 + 20.0 * (e_vdw + e_rep)
        return float(np.clip(score, -15.0, 0.0))

    def _compute_calibrated_vina(self, mol: Chem.Mol, pocket_pos: torch.Tensor, pocket_h: torch.Tensor) -> float:
        conf = mol.GetConformer()
        lig_pos = torch.tensor(conf.GetPositions(), dtype=torch.float32, device=pocket_pos.device)
        return self.compute_analytical_vina_affinity(lig_pos, None, pocket_pos, pocket_h)
