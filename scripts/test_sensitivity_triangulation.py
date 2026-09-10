import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List, Set
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, AllChem

RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import project_build.models.backbones
import project_build.physics

from project_build.core.registry import BACKBONE_REGISTRY
from project_build.data.datasets.crossdocked import CrossDockedLMDBDataset
from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE
from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler
from project_build.postprocess.bond_perception import (
    COVALENT_RADII, MAX_ALLOWED_VALENCE, HALOGENS, is_pair_forbidden, SBDD_HEAVY_REVERSE_MAP
)

def parse_args():
    parser = argparse.ArgumentParser(description="Project BUILD 3.0 附录消融: 三角张力三元环抑制阈值敏感度评测 (Table S4 修复版)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sbdd/latest_sbdd_model.pt", help="模型权重路径")
    parser.add_argument("--num_pockets", type=int, default=20, help="抽样测试口袋数 (默认 20 口袋)")
    parser.add_argument("--samples_per_pocket", type=int, default=5, help="每口袋采样数 (总计 100 分子)")
    parser.add_argument("--output_dir", type=str, default="results/hyperparameter_sensitivity/triangulation", help="独立隔离输出目录")
    return parser.parse_args()

def compute_clean_strain_energy(mol: Chem.Mol) -> float:
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        Chem.FastFindRings(mol_h)
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=50)
        ff = AllChem.UFFGetMoleculeForceField(mol_h)
        if ff is None:
            return 250.0
        e_init = ff.CalcEnergy()
        ff.Minimize(maxIts=200)
        e_min = ff.CalcEnergy()
        strain = max(0.0, e_init - e_min)
        return float(min(1000.0, strain))
    except Exception:
        return 250.0

def reconstruct_triangulation_variant(
    pos_np: np.ndarray,
    atom_elements: List[int],
    tri_cutoff: Optional[float] = 1.50,
    tolerance: float = 0.35
) -> Tuple[Optional[Chem.Mol], bool, int, int, int]:
    N = len(atom_elements)
    if N == 0:
        return None, False, 0, 0, 0

    rw_mol = Chem.RWMol()
    for z in atom_elements:
        rw_mol.AddAtom(Chem.Atom(z))

    conf = Chem.Conformer(N)
    for i in range(N):
        conf.SetAtomPosition(i, (float(pos_np[i, 0]), float(pos_np[i, 1]), float(pos_np[i, 2])))
    rw_mol.AddConformer(conf)

    current_valences = [0.0] * N
    external_bonds_count = [0] * N
    adj_set: Dict[int, Set[int]] = {i: set() for i in range(N)}

    has_seed_ring = False
    if N >= 6 and all(atom_elements[k] == 6 for k in range(6)):
        has_seed_ring = True
        for k in range(6):
            u, v = k, (k + 1) % 6
            rw_mol.AddBond(u, v, Chem.BondType.SINGLE)
            current_valences[u] += 1.0
            current_valences[v] += 1.0
            adj_set[u].add(v)
            adj_set[v].add(u)

    candidate_bonds = []
    for i in range(N):
        z_i = atom_elements[i]
        r_i = COVALENT_RADII.get(z_i, 0.76)
        for j in range(i + 1, N):
            if has_seed_ring and i < 6 and j < 6:
                continue

            z_j = atom_elements[j]
            if is_pair_forbidden(z_i, z_j):
                continue

            r_j = COVALENT_RADII.get(z_j, 0.76)
            dist = float(np.linalg.norm(pos_np[i] - pos_np[j]))
            ideal_bond = r_i + r_j

            if dist <= (ideal_bond + tolerance):
                score = 1.0 / (dist + 0.05)
                candidate_bonds.append((score, dist, i, j))

    candidate_bonds.sort(key=lambda x: x[0], reverse=True)

    for score, dist, i, j in candidate_bonds:
        z_i, z_j = atom_elements[i], atom_elements[j]
        max_v_i = MAX_ALLOWED_VALENCE.get(z_i, 4)
        max_v_j = MAX_ALLOWED_VALENCE.get(z_j, 4)

        if (z_i in HALOGENS and current_valences[i] >= 1.0) or (z_j in HALOGENS and current_valences[j] >= 1.0):
            continue

        if has_seed_ring:
            if i < 6 and j >= 6 and external_bonds_count[i] >= 1: continue
            if j < 6 and i >= 6 and external_bonds_count[j] >= 1: continue

        # 【核心自变量】：三元环抑制判定
        if tri_cutoff is not None:
            common_neighbors = adj_set[i].intersection(adj_set[j])
            if common_neighbors and dist > tri_cutoff:
                continue

        allowed_v_i = (max_v_i - 1.0) if (has_seed_ring and i < 6) else max_v_i
        allowed_v_j = (max_v_j - 1.0) if (has_seed_ring and j < 6) else max_v_j

        if current_valences[i] + 1.0 <= allowed_v_i and current_valences[j] + 1.0 <= allowed_v_j:
            rw_mol.AddBond(i, j, Chem.BondType.SINGLE)
            current_valences[i] += 1.0
            current_valences[j] += 1.0
            adj_set[i].add(j)
            adj_set[j].add(i)
            if has_seed_ring and i < 6 and j >= 6: external_bonds_count[i] += 1
            if has_seed_ring and j < 6 and i >= 6: external_bonds_count[j] += 1

    # 超图 MST 熔接
    mol = rw_mol.GetMol()
    frags = [set(f) for f in Chem.GetMolFrags(mol, asMols=False)]

    while len(frags) > 1:
        best_bridge = None
        min_cost = 1e9

        for fa_idx in range(len(frags)):
            for fb_idx in range(fa_idx + 1, len(frags)):
                frag_a, frag_b = frags[fa_idx], frags[fb_idx]
                for u in frag_a:
                    z_u = atom_elements[u]
                    if z_u in HALOGENS: continue
                    if has_seed_ring and u < 6 and external_bonds_count[u] >= 1: continue
                    max_v_u = (MAX_ALLOWED_VALENCE.get(z_u, 4) - 1.0) if (has_seed_ring and u < 6) else MAX_ALLOWED_VALENCE.get(z_u, 4)
                    if current_valences[u] + 1.0 > max_v_u: continue

                    for v in frag_b:
                        z_v = atom_elements[v]
                        if z_v in HALOGENS: continue
                        if has_seed_ring and v < 6 and external_bonds_count[v] >= 1: continue
                        max_v_v = (MAX_ALLOWED_VALENCE.get(z_v, 4) - 1.0) if (has_seed_ring and v < 6) else MAX_ALLOWED_VALENCE.get(z_v, 4)
                        if current_valences[v] + 1.0 > max_v_v: continue

                        if is_pair_forbidden(z_u, z_v): continue

                        d = float(np.linalg.norm(pos_np[u] - pos_np[v]))
                        if d < min_cost:
                            min_cost = d
                            best_bridge = (u, v)

        if best_bridge is not None:
            u, v = best_bridge
            rw_mol.AddBond(u, v, Chem.BondType.SINGLE)
            current_valences[u] += 1.0
            current_valences[v] += 1.0
            adj_set[u].add(v)
            adj_set[v].add(u)
            if has_seed_ring and u < 6: external_bonds_count[u] += 1
            if has_seed_ring and v < 6: external_bonds_count[v] += 1
        else:
            u = min(list(frags[0]), key=lambda idx: current_valences[idx])
            v = min(list(frags[1]), key=lambda idx: current_valences[idx])
            rw_mol.AddBond(u, v, Chem.BondType.SINGLE)
            current_valences[u] += 1.0
            current_valences[v] += 1.0
            adj_set[u].add(v)
            adj_set[v].add(u)

        mol = rw_mol.GetMol()
        frags = [set(f) for f in Chem.GetMolFrags(mol, asMols=False)]

    if has_seed_ring:
        for k in [0, 2, 4]:
            k_next = (k + 1) % 6
            if current_valences[k] + 1.0 <= 4.0 and current_valences[k_next] + 1.0 <= 4.0:
                b = rw_mol.GetBondBetweenAtoms(k, k_next)
                if b is not None and b.GetBondType() == Chem.BondType.SINGLE:
                    b.SetBondType(Chem.BondType.DOUBLE)
                    current_valences[k] += 1.0
                    current_valences[k_next] += 1.0

    mol = rw_mol.GetMol()
    num_fragments = len(Chem.GetMolFrags(mol, asMols=False))

    # 【关键修复点】：显式调用 FastFindRings，确保准确统计环系
    Chem.FastFindRings(mol)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    total_rings = len(atom_rings)
    count_3_rings = sum(1 for r in atom_rings if len(r) == 3)

    is_valid = False
    try:
        Chem.SanitizeMol(mol)
        is_valid = True
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=True)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRINGS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
            is_valid = True
        except Exception:
            is_valid = False

    return mol, is_valid, num_fragments, count_3_rings, total_rings

def generate_triangulation_latex(summary_data: List[Dict[str, Any]]) -> str:
    latex_code = """
% =========================================================================
% Table S4: Sensitivity Analysis on Triangulation Suppression Cutoff (Table S4)
% =========================================================================
\\begin{table}[t]
\\centering
\\small
\\caption{Sensitivity analysis of the triangulation suppression distance threshold ($d_{\\text{tri}}$) on ring size distributions, aromatic ring counts, conformational strain energy, and drug-likeness evaluated on CrossDocked2020 ($N=100$ point clouds).}
\\label{tab:sensitivity_triangulation}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Suppression Threshold ($d_{\\text{tri}}$)} & \\textbf{Validity (\\%)} $\\uparrow$ & \\textbf{3-Ring Ratio (\\%)} $\\downarrow$ & \\textbf{Arom Rings} $\\uparrow$ & \\textbf{Mean QED} $\\uparrow$ & \\textbf{Strain (kcal/mol)} $\\downarrow$ \\\\
\\midrule
"""
    for r in summary_data:
        name = r["name"]
        val = f"{r['validity_rate']:.1f}"
        three_r = f"{r['three_ring_ratio']:.1f}"
        arom = f"{r['mean_arom_rings']:.2f}"
        qed = f"{r['mean_qed']:.4f}"
        strain = f"{r['mean_strain']:.1f}"
        if name == "Standard (1.50 Å) [Ours]":
            latex_code += f"\\textbf{{{name}}} & \\textbf{{{val}}} & \\textbf{{{three_r}}} & \\textbf{{{arom}}} & \\textbf{{{qed}}} & \\textbf{{{strain}}} \\\\\n"
        else:
            latex_code += f"{name} & {val} & {three_r} & {arom} & {qed} & {strain} \\\\\n"

    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex_code

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 95)
    print(">>> 🔬 [Project BUILD 3.0] 附录消融: 三角张力三元环抑制阈值敏感度评测 (Table S4 修复版)")
    print(f">>> 设备: {device} | 评估口袋数: {args.num_pockets} | 基础分子数: {args.num_pockets * args.samples_per_pocket}")
    print(f">>> 隔离输出目录: {args.output_dir}")
    print("=" * 95)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})

    model = BACKBONE_REGISTRY.build(cfg["model"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()

    sde = DualSpaceVPSDE(schedule_type="cosine").to(device)
    sampler = ReverseSDESampler(
        sde=sde, num_steps=100, spring_k=0.25, corrector_steps=1,
        bond_gate_thresh=0.35, enable_svd_seeding=True, enable_self_healing=True, device=device
    )
    val_dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="test", split_type="name")

    total_samples = args.num_pockets * args.samples_per_pocket

    print("\n>>> [阶段 1/2] 正在统一生成 100 个分子的基准连续 3D 点云构象 ...")
    cached_generations = []
    t_gen_start = time.time()

    for p_idx in range(args.num_pockets):
        sample = val_dataset[p_idx]
        p_pos = sample["pocket_pos"].unsqueeze(0).to(device)
        p_h = sample["pocket_h"].unsqueeze(0).to(device)
        n_atoms = max(18, sample["num_lig_atoms"])
        B = args.samples_per_pocket

        with torch.no_grad():
            x_gen, h_gen, _ = sampler.sample(
                model=model, num_samples=B, num_atoms=n_atoms, num_atom_types=9,
                model_kwargs={
                    "pocket_pos": p_pos.repeat(B, 1, 1),
                    "pocket_h": p_h.repeat(B, 1, 1)
                }
            )

        for b in range(B):
            pos_np = x_gen[b].detach().cpu().numpy()
            h_np = h_gen[b].detach().cpu().numpy()
            atom_indices = np.argmax(h_np, axis=-1)
            elements = [SBDD_HEAVY_REVERSE_MAP.get(idx, 6) for idx in atom_indices]
            cached_generations.append((pos_np, elements))

        sys.stdout.write(f"\r    └── 已采样点云: {len(cached_generations):03d}/{total_samples}")
        sys.stdout.flush()

    print(f"\n>>> 基准点云生成完毕！耗时: {time.time()-t_gen_start:.2f}s")
    print("-" * 95)

    tri_variants = [
        ("No Suppression (None)", None),
        ("Loose (1.60 Å)", 1.60),
        ("Standard (1.50 Å) [Ours]", 1.50),
        ("Strict (1.40 Å)", 1.40)
    ]

    summary_results = []

    print(">>> [阶段 2/2] 正在同一批点云上执行 4 组三元环抑制阈值对比 ...")

    for display_name, cutoff_val in tri_variants:
        valid_cnt = 0
        total_3_rings = 0
        total_all_rings = 0
        aroms_list = []
        qeds = []
        strains = []

        for pos_np, elements in cached_generations:
            mol, is_v, n_f, c3, c_all = reconstruct_triangulation_variant(
                pos_np, elements, tri_cutoff=cutoff_val, tolerance=0.35
            )
            total_3_rings += c3
            total_all_rings += c_all

            if is_v and mol is not None:
                valid_cnt += 1
                try:
                    qeds.append(QED.qed(mol))
                    aroms_list.append(rdMolDescriptors.CalcNumAromaticRings(mol))
                    strain_val = compute_clean_strain_energy(mol)
                    strains.append(strain_val)
                except Exception:
                    pass

        val_rate = (valid_cnt / total_samples) * 100.0
        ratio_3_ring = (total_3_rings / total_all_rings * 100.0) if total_all_rings > 0 else 0.0
        m_arom = float(np.mean(aroms_list)) if aroms_list else 0.0
        m_qed = float(np.mean(qeds)) if qeds else 0.0
        m_strain = float(np.mean(strains)) if strains else 250.0

        res_dict = {
            "name": display_name,
            "cutoff": cutoff_val,
            "validity_rate": val_rate,
            "three_ring_ratio": ratio_3_ring,
            "total_3_rings": total_3_rings,
            "total_rings": total_all_rings,
            "mean_arom_rings": m_arom,
            "mean_qed": m_qed,
            "mean_strain": m_strain
        }
        summary_results.append(res_dict)

        print(f"  └── [{display_name:<25}] 3元环占比: {ratio_3_ring:5.1f}% ({total_3_rings:03d}/{total_all_rings:03d}) | 芳香环: {m_arom:.2f} | 应变能: {m_strain:5.1f} kcal/mol | QED: {m_qed:.4f} | 合法率: {val_rate:5.1f}%")

    json_path = os.path.join(args.output_dir, "triangulation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    latex_table = generate_triangulation_latex(summary_results)
    latex_path = os.path.join(args.output_dir, "triangulation_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("\n" + "=" * 95)
    print("📊 【三角张力三元环抑制阈值敏感度评测完成 (Table S4)】")
    print("=" * 95)
    print(latex_table)
    print(f">>> 📝 结果数据已保存至: {json_path}")
    print(f">>> 📄 LaTeX 表格已生成至: {latex_path}\n")

if __name__ == "__main__":
    main()
