# scripts/supplementary/extract_true_allosteric_5d41.py
# ==============================================================================
# 精准锁定 5D41 真正的变构配体 57N (EAI001)，提取真正的 αC-helix 变构裂谷
# ==============================================================================

import os
import sys
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, QED

RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

PROTEIN_ATOM_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 34: 5}

# 57N (EAI001) 的真实 SMILES
EAI001_SMILES = "O=C1Nc2c(C1c1cc(F)c(O)c(F)c1)cc(F)c(O)c2"

def main():
    print("=" * 105)
    print(">>> 🎯 [精准提取] 锁定 5D41 真正变构抑制剂 57N (EAI001)，切分 αC-helix 变构裂谷")
    print("=" * 105)

    data_dir = "real_world_study/data/allosteric_targets"
    pdb_path = os.path.join(data_dir, "5D41.pdb")
    
    if not os.path.exists(pdb_path):
        print(f"❌ 找不到文件: {pdb_path}")
        return

    with open(pdb_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    prot_atoms = []
    lig_atoms_pos = []

    # 显式只抓 57N (真正的变构配体)！绝不碰 31 原子的正构 ANP！
    for line in lines:
        if line.startswith("ATOM"):
            elem = line[76:78].strip() or line[12:14].strip()[0]
            prot_atoms.append({
                "elem": elem,
                "pos": [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            })
        elif line.startswith("HETATM") and "57N" in line:
            lig_atoms_pos.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    lig_pos = np.array(lig_atoms_pos)
    print(f"  ✅ 成功锁定真正变构配体 57N: 提取到 {len(lig_pos)} 个配体原子坐标！")

    # 以 57N 为中心，切分 10.0 Å 变构结合腔
    elem_to_z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "SE": 34}
    pocket_pos_list = []
    pocket_elem_list = []

    for atom in prot_atoms:
        p_coord = np.array(atom["pos"])
        min_dist = np.min(np.linalg.norm(lig_pos - p_coord, axis=-1))
        if min_dist <= 10.0:
            pocket_pos_list.append(atom["pos"])
            pocket_elem_list.append(elem_to_z.get(atom["elem"].upper(), 6))

    p_tensor = torch.tensor(pocket_pos_list, dtype=torch.float32)
    e_tensor = torch.tensor(pocket_elem_list, dtype=torch.long)

    if p_tensor.shape[0] > 256:
        dists = torch.cdist(p_tensor, torch.tensor(lig_pos, dtype=torch.float32)).min(dim=1)[0]
        topk_idx = torch.topk(dists, k=256, largest=False)[1]
        p_tensor = p_tensor[topk_idx]
        e_tensor = e_tensor[topk_idx]

    N_p = p_tensor.shape[0]
    h_tensor = torch.zeros(N_p, 6, dtype=torch.float32)
    for i, z in enumerate(e_tensor.tolist()):
        h_tensor[i, PROTEIN_ATOM_MAP.get(z, 1)] = 1.0

    # 保存真正的变构口袋张量
    pt_path = os.path.join(data_dir, "pocket_EGFR_Allosteric.pt")
    center = np.mean(p_tensor.numpy(), axis=0).tolist()
    torch.save({
        "pocket_pos": p_tensor,
        "pocket_h": h_tensor,
        "pocket_elem": e_tensor,
        "center": center
    }, pt_path)

    # 保存受体 PDBQT
    rec_pdbqt = os.path.join(data_dir, "pocket_EGFR_Allosteric.pdbqt")
    with open(rec_pdbqt, "w", encoding="utf-8") as f:
        f.write("REMARK    TRUE EGFR 5D41 ALLOSTERIC RECEPTOR PDBQT (57N SITE)\n")
        elem_map = {1: 'C', 2: 'N', 3: 'O', 4: 'S', 5: 'Se'}
        types = np.argmax(h_tensor.numpy(), axis=-1)
        pos_np = p_tensor.numpy()
        for idx in range(N_p):
            t = types[idx]
            if t == 0: continue
            elem = elem_map.get(t, 'C')
            x, y, z = pos_np[idx]
            atom_t = 'OA' if elem == 'O' else ('NA' if elem == 'N' else ('SA' if elem == 'S' else 'C'))
            f.write(f"ATOM  {idx+1:5d}  {elem:<3s} PRO A{idx//5+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00     0.000 {atom_t:<2s}\n")
        f.write("TER\nEND\n")

    print(f"  ✅ 真正变构口袋切分完毕: {N_p} 个重原子")
    print(f"  • 真正变构口袋几何中心: {np.round(center, 2)}")
    print(f"  • 已覆盖保存至: {pt_path} 与 {rec_pdbqt}")

    # 测定 57N 在它自己口袋里的官方 Vina 对接基准
    mol_ref = Chem.MolFromSmiles(EAI001_SMILES)
    mol_h = Chem.AddHs(mol_ref)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    
    prep = MoleculePreparation()
    setups = prep.prepare(mol_h)
    ref_pdbqt = PDBQTWriterLegacy.write_string(setups[0])
    if isinstance(ref_pdbqt, tuple): ref_pdbqt = ref_pdbqt[0]

    v = Vina(sf_name='vina', cpu=4, verbosity=0)
    v.set_receptor(rec_pdbqt)
    v.set_ligand_from_string(ref_pdbqt)
    v.compute_vina_maps(center=center, box_size=[15.0, 15.0, 15.0]) # 限制在 15Å 避免漂到 ATP 位点
    v.dock(exhaustiveness=8, n_poses=1)
    ref_vina = float(v.energies(n_poses=1)[0][0])

    print(f"  🏆 原研变构药 (57N / EAI001) 在真变构位点的实测 Vina: {ref_vina:.2f} kcal/mol | QED: {QED.qed(mol_ref):.4f}\n")

if __name__ == "__main__":
    main()
