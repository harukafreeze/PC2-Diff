from typing import Dict, Any, Tuple, List, Optional
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

# 经典基准真实分子 SMILES 权威数据库
REAL_MOLECULE_SMILES = {
    "ethanol": "CCO",                                  # 乙醇 (C2H6O)
    "benzene": "c1ccccc1",                             # 苯环 (C6H6)
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",                # 阿司匹林 (C9H8O4, 经典药物)
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",        # 咖啡因 (C8H10N4O2)
    "alanine_dipeptide": "CC(=O)NC(C)C(=O)NC"          # 丙氨酸二肽 (ACE-ALA-NME, 22原子全氢体系)
}

# QM9 标准原子序数映射表 (H=1, C=6, N=7, O=8, F=9)
QM9_ATOM_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

class RealMoleculeFactory:
    """
    真实化学药物分子数据引擎。
    
    功能:
        1. 从真实化学 SMILES 构建分子图；
        2. 自动补充所有显式氢原子 (AddHs)；
        3. 利用实验级 Distance Geometry (ETKDGv3) 算法生成精确真实 3D 构象；
        4. 运行 Gasteiger 量子偏电荷算法计算真实原子电荷；
        5. 严格将三维真实坐标投影至零质心流形 M。
    """
    @staticmethod
    def get_real_molecule(
        name: str = "aspirin",
        random_seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Chem.Mol]:
        """
        获取指定真实药物分子的真实物理数据。
        
        返回:
            (pos, atom_types_onehot, atom_z, charges, mol_object):
                pos: [1, N, 3] 真实 3D 空间坐标 (Å) (严格质心归零)
                atom_types_onehot: [1, N, 5] 离散原子类型 One-hot 编码
                atom_z: [1, N] 真实原子序数 (1, 6, 7, 8, 9...)
                charges: [1, N] Gasteiger 真实量子偏电荷
                mol_object: RDKit 原生分子对象
        """
        if name not in REAL_MOLECULE_SMILES:
            raise KeyError(f"未知分子名 '{name}'！可用真实分子库: {list(REAL_MOLECULE_SMILES.keys())}")
            
        smiles = REAL_MOLECULE_SMILES[name]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) # 补齐所有氢原子以构成完整物理体系
        
        # 运行 ETKDGv3 算法生成真实三维几何构象
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        success = AllChem.EmbedMolecule(mol, params)
        if success != 0:
            # 若 ETKDG 失败则使用标准嵌入算法兜底
            AllChem.EmbedMolecule(mol, randomSeed=random_seed)
            
        # 计算真实原子偏电荷 (Gasteiger Charges)
        AllChem.ComputeGasteigerCharges(mol)
        
        conf = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()
        
        coords = []
        atom_z_list = []
        charges_list = []
        one_hot_list = []
        
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            z = atom.GetAtomicNum()
            pos_3d = conf.GetAtomPosition(i)
            coords.append([pos_3d.x, pos_3d.y, pos_3d.z])
            atom_z_list.append(z)
            
            # 提取真实物理电荷
            charge_val = float(atom.GetProp("_GasteigerCharge"))
            if charge_val != charge_val: # NaN 保护
                charge_val = 0.0
            charges_list.append(charge_val)
            
            # One-hot 编码 (H, C, N, O, F)
            one_hot = [0.0] * 5
            type_idx = QM9_ATOM_MAP.get(z, 1) # 默认归入碳通道
            one_hot[type_idx] = 1.0
            one_hot_list.append(one_hot)
            
        pos_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0) # [1, N, 3]
        one_hot_tensor = torch.tensor(one_hot_list, dtype=torch.float32).unsqueeze(0) # [1, N, 5]
        atom_z_tensor = torch.tensor(atom_z_list, dtype=torch.long).unsqueeze(0) # [1, N]
        charges_tensor = torch.tensor(charges_list, dtype=torch.float32).unsqueeze(0) # [1, N]
        
        # 强制执行零质心流形投影: x_0 <- P_M x_0
        com = torch.mean(pos_tensor, dim=1, keepdim=True)
        pos_tensor = pos_tensor - com
        
        return pos_tensor, one_hot_tensor, atom_z_tensor, charges_tensor, mol

    @staticmethod
    def get_real_batch(
        mol_names: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建包含多个不同真实药物分子的真实批次 (Padding Batch)。
        """
        if mol_names is None:
            mol_names = ["ethanol", "benzene", "aspirin", "alanine_dipeptide"]
            
        mol_data = [RealMoleculeFactory.get_real_molecule(name) for name in mol_names]
        max_n = max(d[0].shape[1] for d in mol_data)
        B = len(mol_data)
        
        pos_batch = torch.zeros(B, max_n, 3, dtype=torch.float32)
        h_batch = torch.zeros(B, max_n, 5, dtype=torch.float32)
        z_batch = torch.zeros(B, max_n, dtype=torch.long)
        charges_batch = torch.zeros(B, max_n, dtype=torch.float32)
        mask_batch = torch.zeros(B, max_n, dtype=torch.bool)
        
        for b, (pos, h, z, chg, _) in enumerate(mol_data):
            n = pos.shape[1]
            pos_batch[b, :n] = pos[0]
            h_batch[b, :n] = h[0]
            z_batch[b, :n] = z[0]
            charges_batch[b, :n] = chg[0]
            mask_batch[b, :n] = True
            
        return pos_batch, h_batch, z_batch, charges_batch, mask_batch
