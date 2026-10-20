import os
import sys
import tempfile
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED

RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

PROTEIN_ATOM_MAP = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4, 34: 5}

def parse_pdb_file(pdb_path: str):
    with open(pdb_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    protein_atoms = []
    ligand_lines = []
    
    for line in lines:
        if line.startswith("ATOM"):
            elem = line[76:78].strip()
            if not elem:
                elem = line[12:14].strip()[0]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            protein_atoms.append({
                "elem": elem,
                "pos": [x, y, z],
                "line": line
            })
        elif line.startswith("HETATM") and "AQ4" in line:
            ligand_lines.append(line)
            
    return protein_atoms, ligand_lines

def main():
    print("=" * 85)
    print(">>> 🔬 [实战第一步] EGFR 激酶结合口袋提取与原研药 Erlotinib 基准测定")
    print("=" * 85)
    
    pdb_path = "real_world_study/data/1M17.pdb"
    protein_atoms, ligand_lines = parse_pdb_file(pdb_path)
    
    # 1. 提取原研配体 AQ4
    lig_pdb_path = "real_world_study/data/ref_erlotinib.pdb"
    with open(lig_pdb_path, "w", encoding="utf-8") as f:
        for l in ligand_lines:
            f.write(l)
        f.write("END\n")
        
    mol_ref = Chem.MolFromPDBFile(lig_pdb_path, removeHs=True)
    if mol_ref is None:
        smi_erlotinib = "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"
        mol_ref = Chem.MolFromSmiles(smi_erlotinib)
        mol_h = Chem.AddHs(mol_ref)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        mol_ref = Chem.RemoveHs(mol_h)
        
    conf = mol_ref.GetConformer()
    lig_pos = conf.GetPositions()
    
    # 2. 提取配体周围 10.0 Å 结合腔残基 (标准 SBDD 口袋球)
    pocket_atoms = []
    pocket_pos_list = []
    pocket_elem_list = []
    
    elem_to_z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "SE": 34}
    
    for atom in protein_atoms:
        p_coord = np.array(atom["pos"])
        min_dist = np.min(np.linalg.norm(lig_pos - p_coord, axis=-1))
        if min_dist <= 10.0:
            pocket_atoms.append(atom)
            pocket_pos_list.append(atom["pos"])
            z = elem_to_z.get(atom["elem"].upper(), 6)
            pocket_elem_list.append(z)
            
    pocket_pos_tensor = torch.tensor(pocket_pos_list, dtype=torch.float32)
    pocket_elem_tensor = torch.tensor(pocket_elem_list, dtype=torch.long)
    
    if pocket_pos_tensor.shape[0] > 256:
        dists = torch.cdist(pocket_pos_tensor, torch.tensor(lig_pos, dtype=torch.float32)).min(dim=1)[0]
        topk_idx = torch.topk(dists, k=256, largest=False)[1]
        pocket_pos_tensor = pocket_pos_tensor[topk_idx]
        pocket_elem_tensor = pocket_elem_tensor[topk_idx]
        
    N_p = pocket_pos_tensor.shape[0]
    pocket_h_tensor = torch.zeros(N_p, 6, dtype=torch.float32)
    for i, z in enumerate(pocket_elem_tensor.tolist()):
        pocket_h_tensor[i, PROTEIN_ATOM_MAP.get(z, 1)] = 1.0
        
    pt_save_path = "real_world_study/data/pocket_1M17.pt"
    torch.save({
        "pocket_pos": pocket_pos_tensor,
        "pocket_h": pocket_h_tensor,
        "pocket_elem": pocket_elem_tensor,
        "ref_lig_pos": torch.tensor(lig_pos, dtype=torch.float32),
        "center": np.mean(pocket_pos_tensor.numpy(), axis=0).tolist()
    }, pt_save_path)
    
    rec_pdbqt_path = "real_world_study/data/pocket_1M17.pdbqt"
    with open(rec_pdbqt_path, "w", encoding="utf-8") as f:
        f.write("REMARK    EGFR POCKET 1M17 RECEPTOR PDBQT\n")
        elem_map = {1: 'C', 2: 'N', 3: 'O', 4: 'S', 5: 'Se'}
        p_types = np.argmax(pocket_h_tensor.numpy(), axis=-1)
        p_pos = pocket_pos_tensor.numpy()
        for idx in range(N_p):
            t = p_types[idx]
            if t == 0: continue
            elem = elem_map.get(t, 'C')
            x, y, z = p_pos[idx]
            atom_t = 'OA' if elem == 'O' else ('NA' if elem == 'N' else ('SA' if elem == 'S' else 'C'))
            f.write(f"ATOM  {idx+1:5d}  {elem:<3s} PRO A{idx//5+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00     0.000 {atom_t:<2s}\n")
        f.write("TER\nEND\n")
        
    print(f"✅ 成功提取 EGFR 结合口袋: {N_p} 个重原子 (10.0 Å 截止球)")
    print(f"   口袋几何中心 Center: {np.round(np.mean(p_pos, axis=0), 2)}")
    print(f"   口袋张量资产保存至: {pt_save_path}")
    print(f"   受体 PDBQT 保存至: {rec_pdbqt_path}")
    
    # 3. 测定原研药 Erlotinib 的官方 Vina 对接基准分数
    print("\n>>> 正在运行 Scripps 官方 Vina 1.2.7 C++ 测定原研药 (Erlotinib) 结合能基准...")
    mol_h = Chem.AddHs(mol_ref, addCoords=True)
    prep = MoleculePreparation()
    setups = prep.prepare(mol_h)
    raw = PDBQTWriterLegacy.write_string(setups[0])
    lig_pdbqt_str = raw[0] if isinstance(raw, (tuple, list)) else raw
    
    center = np.mean(p_pos, axis=0).tolist()
    v = Vina(sf_name='vina', cpu=8, verbosity=0)
    v.set_receptor(rec_pdbqt_path)
    v.set_ligand_from_string(lig_pdbqt_str)
    v.compute_vina_maps(center=center, box_size=[20.0, 20.0, 20.0])
    v.dock(exhaustiveness=8, n_poses=1)
    ref_vina_dock = float(v.energies(n_poses=1)[0][0])
    
    # 严格对齐标准调用
    qed_ref = QED.qed(mol_ref)
    mw_ref = Descriptors.MolWt(mol_ref)
    
    print("=" * 85)
    print("🏆 【EGFR 原研标杆药物 (Erlotinib / 特罗凯) 官方实测基线决算】")
    print("=" * 85)
    print(f"  ├── 💊 标杆药物分子量 (MW):    {mw_ref:.2f} Da")
    print(f"  ├── 🧪 标杆药物类药性 (QED):   {qed_ref:.4f}")
    print(f"  └── 🎯 官方实测 Vina Dock 结合能: {ref_vina_dock:.2f} kcal/mol")
    print("=" * 85)
    print(">>> 靶标数据与原研对照标杆就绪！随时可以启动 500 分子从头生成实战！\n")

if __name__ == "__main__":
    main()
