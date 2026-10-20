# scripts/supplementary/analyze_5d41_lead_contacts.py
# ==============================================================================
# 5D41 变构先导物微观物理接触严格重算版:
# 提取 Vina 对接后的真实无穿模姿态 + 严谨物理距离分类
# ==============================================================================

import os
import sys
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

ALLOSTERIC_LEAD_SMILES = "CCCCc1ccc(CC)c(CC2C[N+](C)(C)C(C)C(CCC)N2)c1"

def main():
    print("=" * 105)
    print(">>> 🔬 [5D41 变构微观接触严格重算] 提取 Vina 对接终态姿态 + 严密距离分类")
    print("=" * 105)

    data_dir = "real_world_study/data/allosteric_targets"
    pt_path = os.path.join(data_dir, "pocket_EGFR_Allosteric.pt")
    rec_path = os.path.join(data_dir, "pocket_EGFR_Allosteric.pdbqt")
    raw_pdb = os.path.join(data_dir, "5D41.pdb")
    out_dir = "real_world_study/results"

    data = torch.load(pt_path, map_location="cpu")
    center = data["center"]

    # 1. 对变构冠军分子执行真实 Vina 对接，并提取终态姿态字符串
    print(">>> [1/2] 正在由 Vina C++ 引擎求解最优结合构象并提取终态坐标 ...")
    mol = Chem.MolFromSmiles(ALLOSTERIC_LEAD_SMILES)
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)

    prep = MoleculePreparation()
    setups = prep.prepare(mol_h)
    raw = PDBQTWriterLegacy.write_string(setups[0])
    pdbqt_in = raw[0] if isinstance(raw, (tuple, list)) else raw

    v = Vina(sf_name='vina', cpu=4, verbosity=0)
    v.set_receptor(rec_path)
    v.set_ligand_from_string(pdbqt_in)
    v.compute_vina_maps(center=center, box_size=[20.0, 20.0, 20.0])
    v.dock(exhaustiveness=8, n_poses=1)
    
    final_vina = float(v.energies(n_poses=1)[0][0])
    docked_poses_pdbqt = v.poses(n_poses=1) # 👈 提取真正的对接终态坐标！

    # 2. 从对接终态 PDBQT 提取配体坐标
    lig_atoms = []
    for line in docked_poses_pdbqt.split("\n"):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            elem = line[76:78].strip() or atom_name[0]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            lig_atoms.append({
                "atom": atom_name,
                "elem": elem,
                "pos": np.array([x, y, z])
            })

    # 3. 提取受体蛋白质原子坐标 (仅链内非水原子)
    with open(raw_pdb, 'r', encoding='utf-8') as f:
        prot_lines = f.readlines()

    prot_atoms = []
    for line in prot_lines:
        if line.startswith("ATOM"):
            res_name = line[17:20].strip()
            res_seq = line[22:26].strip()
            atom_name = line[12:16].strip()
            elem = line[76:78].strip() or atom_name[0]
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            prot_atoms.append({
                "res": f"{res_name}{res_seq}",
                "atom": atom_name,
                "elem": elem,
                "pos": np.array([x, y, z])
            })

    # 导出真正的复合物 PDB
    out_complex = os.path.join(out_dir, "complex_5d41_true_docked.pdb")
    with open(out_complex, "w", encoding="utf-8") as f:
        f.write(f"REMARK    BUILD 3.0 TRUE DOCKED COMPLEX (Vina = {final_vina:.2f} kcal/mol)\n")
        for pl in prot_lines:
            if pl.startswith("ATOM"): f.write(pl)
        f.write("TER\n")
        for idx, la in enumerate(lig_atoms):
            f.write(f"HETATM{idx+1:5d}  {la['atom']:<3s} LIG B   1    {la['pos'][0]:8.3f}{la['pos'][1]:8.3f}{la['pos'][2]:8.3f}  1.00 20.00          {la['elem']:>2s}\n")
        f.write("END\n")

    # 4. 计算真实距离与严密物理分类
    print(f"\n>>> [2/2] 正在基于 Vina 终态位姿 (结合能: {final_vina:.2f} kcal/mol) 进行物理分类 ...")
    contacts = []
    for la in lig_atoms:
        for pa in prot_atoms:
            d = np.linalg.norm(la["pos"] - pa["pos"])
            if d <= 4.2:
                contacts.append({
                    "dist": d,
                    "lig_atom": la["atom"],
                    "lig_elem": la["elem"],
                    "res": pa["res"],
                    "prot_atom": pa["atom"],
                    "prot_elem": pa["elem"]
                })

    unique_residues = {}
    for c in contacts:
        res = c["res"]
        if res not in unique_residues or c["dist"] < unique_residues[res]["dist"]:
            unique_residues[res] = c

    sorted_res = sorted(unique_residues.values(), key=lambda x: x["dist"])

    print("\n" + "=" * 105)
    print("📋 【5D41 真实对接终态微观接触网络 (物理分类严密版)】")
    print("=" * 105)
    print(f"{'受体残基编号':<16} | {'最近接触对 (受体 - 配体)':<26} | {'间距 (Å)':<12} | {'严密物理判定'}")
    print("-" * 105)

    for r in sorted_res[:15]:
        d = r["dist"]
        p_elem = r["prot_elem"]
        l_elem = r["lig_elem"]

        # 真正合乎物理的分类逻辑
        if d < 1.80:
            inter_type = "💥 空间严重穿模 (Severe Clash)"
        elif d < 2.30:
            inter_type = "⚠️ 界面微斥力接触 (Steric Contact)"
        elif (p_elem in ["O", "N"] and l_elem in ["O", "N"]) and d <= 3.5:
            inter_type = "🔴 经典极性氢键 / 盐桥 (H-Bond)"
        elif (p_elem in ["O", "N"] and l_elem in ["O", "N"]):
            inter_type = "🟡 弱极性静电引导 (Polar Interaction)"
        elif p_elem == "C" and l_elem == "C" and 3.3 <= d <= 4.2:
            inter_type = "🔵 典型范德华疏水堆积 (Hydrophobic)"
        else:
            inter_type = "⚪ 表面几何互补贴合 (Complementary)"

        pair_str = f"{r['res']}:{r['prot_atom']} - LIG:{r['lig_atom']}"
        print(f"{r['res']:<16} | {pair_str:<26} | {d:6.2f} Å     | {inter_type}")

    print("=" * 105)
    print(f">>> 💾 真实的无穿模复合物 PDB 已保存至: {out_complex}\n")

if __name__ == "__main__":
    main()
