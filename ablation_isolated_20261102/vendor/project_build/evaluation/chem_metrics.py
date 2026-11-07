from typing import List, Dict, Any, Tuple, Optional
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from project_build.postprocess.bond_perception import BondPerceptionEngine

class MolecularMetricsEvaluator:
    """
    全量分子生成学术标准评测引擎。
    
    输出指标:
        1. Validity (%): RDKit 化学合法率 (无价态错误)
        2. Connectivity (%): 完整单一连通大分子比例 (无碎片)
        3. Uniqueness (%): 生成有效分子中的非重复分子比例
        4. Novelty (%): 生成分子在训练集中未曾出现的新颖性比例
        5. Mean QED: 平均类药性评分 [0, 1]
        6. Clash Rate (%): 配体原子间距 < 1.0 Å 的异常碰撞率
    """
    def __init__(self, train_smiles_set: Optional[set] = None) -> None:
        self.train_smiles_set = train_smiles_set if train_smiles_set is not None else set()

    def evaluate_batch(
        self,
        pos_list: torch.Tensor,
        atom_types_list: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        参数:
            pos_list: [B, N, 3] 批量生成的三维坐标
            atom_types_list: [B, N, 5] 批量生成的原子类型
            mask: [B, N] 有效原子掩码
            
        返回:
            评测指标报表字典
        """
        B = pos_list.shape[0]
        valid_mols = []
        valid_smiles = []
        num_connected = 0
        clash_count = 0
        total_atom_pairs = 0
        
        pos_cpu = pos_list.detach().cpu()
        h_cpu = atom_types_list.detach().cpu()
        mask_cpu = mask.detach().cpu() if mask is not None else None
        
        # 1. 逐个分子进行三维建键与化学体检
        for b in range(B):
            if mask_cpu is not None:
                n = torch.sum(mask_cpu[b]).item()
                pos_b = pos_cpu[b, :n]
                h_b = h_cpu[b, :n]
            else:
                pos_b = pos_cpu[b]
                h_b = h_cpu[b]
                n = pos_b.shape[0]
                
            # 计算分子内部原子对碰撞率 (< 1.0 Å)
            diff = pos_b.unsqueeze(1) - pos_b.unsqueeze(0)
            dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
            eye_mask = torch.eye(n, dtype=torch.bool)
            pair_dists = dist.masked_select(~eye_mask)
            
            clashes = torch.sum(pair_dists < 1.0).item() // 2
            clash_count += clashes
            total_atom_pairs += len(pair_dists) // 2
            
            # RDKit 重构
            mol, is_valid, num_frags = BondPerceptionEngine.reconstruct_mol(pos_b, h_b)
            
            if is_valid and mol is not None:
                valid_mols.append(mol)
                smiles = Chem.MolToSmiles(mol)
                valid_smiles.append(smiles)
                if num_frags == 1:
                    num_connected += 1

        num_valid = len(valid_mols)
        validity_rate = (num_valid / B) * 100.0
        connectivity_rate = (num_connected / B) * 100.0
        
        # 2. 唯一性 (Uniqueness) 与 新颖性 (Novelty)
        unique_smiles_set = set(valid_smiles)
        uniqueness_rate = (len(unique_smiles_set) / num_valid * 100.0) if num_valid > 0 else 0.0
        
        if self.train_smiles_set and len(unique_smiles_set) > 0:
            novel_count = sum(1 for smi in unique_smiles_set if smi not in self.train_smiles_set)
            novelty_rate = (novel_count / len(unique_smiles_set)) * 100.0
        else:
            novelty_rate = 100.0
            
        # 3. 类药性 QED 计算
        qed_scores = [QED.qed(m) for m in valid_mols]
        mean_qed = (sum(qed_scores) / len(qed_scores)) if qed_scores else 0.0
        
        # 4. 碰撞率
        clash_rate = (clash_count / total_atom_pairs * 100.0) if total_atom_pairs > 0 else 0.0
        
        return {
            "total_sampled": B,
            "valid_molecules": num_valid,
            "validity_rate (%)": validity_rate,
            "connectivity_rate (%)": connectivity_rate,
            "uniqueness_rate (%)": uniqueness_rate,
            "novelty_rate (%)": novelty_rate,
            "mean_qed": mean_qed,
            "clash_rate (%)": clash_rate
        }
