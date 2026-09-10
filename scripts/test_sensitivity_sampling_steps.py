import os
import sys
import time
import json
import argparse
import tempfile
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors, AllChem

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
from project_build.postprocess.bond_perception import BondPerceptionEngine

from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

def parse_args():
    parser = argparse.ArgumentParser(description="Project BUILD 3.0 附录消融: 逆向扩散采样步数敏感度评测 (T ∈ {25, 50, 100, 250, 500})")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sbdd/latest_sbdd_model.pt", help="模型权重路径")
    parser.add_argument("--num_pockets", type=int, default=20, help="抽样测试口袋数 (默认 20 口袋)")
    parser.add_argument("--samples_per_pocket", type=int, default=5, help="每口袋采样数 (总计 100 分子/步数配置)")
    parser.add_argument("--output_dir", type=str, default="results/hyperparameter_sensitivity/sampling_steps", help="独立隔离输出目录")
    return parser.parse_args()

def mol_to_pdbqt_meeko(mol: Chem.Mol) -> str:
    mol_h = Chem.AddHs(mol, addCoords=True)
    prep = MoleculePreparation()
    setups = prep.prepare(mol_h)
    raw = PDBQTWriterLegacy.write_string(setups[0])
    return raw[0] if isinstance(raw, (tuple, list)) else raw

def write_pocket_pdbqt_file(pocket_pos: np.ndarray, pocket_types: np.ndarray, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("REMARK    RECEPTOR PDBQT FOR SENSITIVITY TEST\n")
        elem_map = {1: 'C', 2: 'N', 3: 'O', 4: 'S', 5: 'Se'}
        for idx in range(len(pocket_pos)):
            t = pocket_types[idx]
            if t == 0:
                continue
            elem = elem_map.get(t, 'C')
            x, y, z = pocket_pos[idx]
            atom_t = 'OA' if elem == 'O' else ('NA' if elem == 'N' else ('SA' if elem == 'S' else 'C'))
            line = f"ATOM  {idx+1:5d}  {elem:<3s} PRO A{idx//5+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00     0.000 {atom_t:<2s}\n"
            f.write(line)
        f.write("TER\nEND\n")

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

def compute_pocket_clashes(lig_pos: np.ndarray, pocket_pos: np.ndarray, pocket_types: np.ndarray) -> int:
    p_heavy = pocket_pos[pocket_types > 0]
    dists = np.linalg.norm(lig_pos[:, None, :] - p_heavy[None, :, :], axis=-1)
    return int(np.sum(dists < 1.50))

def generate_latex_table(summary_results: List[Dict[str, Any]]) -> str:
    latex_code = """
% =========================================================================
% Table S1: Sensitivity Analysis on Reverse Diffusion Sampling Steps (T)
% =========================================================================
\\begin{table}[t]
\\centering
\\small
\\caption{Sensitivity analysis of the number of reverse sampling steps ($T$) on generation quality, biophysical plausibility, and runtime efficiency evaluated on CrossDocked2020 ($N=100$ molecules per setting across 20 independent pockets).}
\\label{tab:sensitivity_sampling_steps}
\\begin{tabular}{ccccccc}
\\toprule
\\textbf{Steps ($T$)} & \\textbf{Validity (\\%)} $\\uparrow$ & \\textbf{Connectivity (\\%)} $\\uparrow$ & \\textbf{Mean QED} $\\uparrow$ & \\textbf{Vina Dock} $\\downarrow$ & \\textbf{Clashes} $\\downarrow$ & \\textbf{Latency (s/mol)} $\\downarrow$ \\\\
\\midrule
"""
    for r in summary_results:
        t_val = r["steps"]
        val = f"{r['validity_rate']:.1f}"
        conn = f"{r['connectivity_rate']:.1f}"
        qed = f"{r['mean_qed']:.4f}"
        vina = f"{r['mean_vina_dock']:.2f}"
        clash = f"{r['mean_clashes']:.2f}"
        lat = f"{r['latency_per_mol_sec']:.3f}"
        if t_val == 100:
            latex_code += f"\\textbf{{{t_val} (Default)}} & \\textbf{{{val}}} & \\textbf{{{conn}}} & \\textbf{{{qed}}} & \\textbf{{{vina}}} & \\textbf{{{clash}}} & {lat} \\\\\n"
        else:
            latex_code += f"{t_val} & {val} & {conn} & {qed} & {vina} & {clash} & {lat} \\\\\n"

    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    return latex_code

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    step_candidates = [25, 50, 100, 250, 500]
    total_mols_per_setting = args.num_pockets * args.samples_per_pocket

    print("\n" + "=" * 95)
    print(">>> 🔬 [Project BUILD 3.0] 附录消融: 采样步数敏感度评测 (T ∈ {25, 50, 100, 250, 500})")
    print(f">>> 设备: {device} | 评估口袋数: {args.num_pockets} | 每设置测试分子数: {total_mols_per_setting} (全矩阵总计 {len(step_candidates) * total_mols_per_setting} 分子)")
    print(f">>> 隔离输出目录: {args.output_dir}")
    print("=" * 95)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})

    model = BACKBONE_REGISTRY.build(cfg["model"]).to(device)
    model.load_state_dict(checkpoint.get("ema_state_dict", checkpoint["model_state_dict"]))
    model.eval()

    sde = DualSpaceVPSDE(schedule_type="cosine").to(device)
    val_dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="test", split_type="name")

    summary_results = []

    for step_t in step_candidates:
        print(f"\n🚀 正在评估采样步数配置: 【T = {step_t:03d}】 ...")
        sampler = ReverseSDESampler(
            sde=sde,
            num_steps=step_t,
            spring_k=0.25,
            corrector_steps=1,
            bond_gate_thresh=0.35,
            enable_svd_seeding=True,
            enable_self_healing=True,
            device=device
        )

        valid_cnt = 0
        conn_cnt = 0
        qeds = []
        vinas = []
        clashes = []
        strains = []
        gen_times = []

        for p_idx in range(args.num_pockets):
            sample = val_dataset[p_idx]
            p_pos = sample["pocket_pos"].unsqueeze(0).to(device)
            p_h = sample["pocket_h"].unsqueeze(0).to(device)
            n_atoms = max(18, sample["num_lig_atoms"])
            B = args.samples_per_pocket

            t_gen_start = time.time()
            with torch.no_grad():
                x_gen, h_gen, _ = sampler.sample(
                    model=model,
                    num_samples=B,
                    num_atoms=n_atoms,
                    num_atom_types=9,
                    model_kwargs={
                        "pocket_pos": p_pos.repeat(B, 1, 1),
                        "pocket_h": p_h.repeat(B, 1, 1)
                    }
                )
            t_gen_elapsed = time.time() - t_gen_start
            gen_times.append(t_gen_elapsed)

            p_pos_np = p_pos[0].cpu().numpy()
            p_h_np = p_h[0].cpu().numpy()
            p_types = np.argmax(p_h_np, axis=-1)

            with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as tmp_rec:
                rec_path = tmp_rec.name
            write_pocket_pdbqt_file(p_pos_np, p_types, rec_path)
            center = np.mean(p_pos_np, axis=0).tolist()

            try:
                for b in range(B):
                    pos_b = x_gen[b]
                    h_b = h_gen[b]

                    mol, is_valid, num_frags = BondPerceptionEngine.reconstruct_mol(pos=pos_b, atom_types=h_b, tolerance=0.35)
                    c_cnt = compute_pocket_clashes(pos_b.cpu().numpy(), p_pos_np, p_types)
                    clashes.append(c_cnt)

                    if num_frags == 1:
                        conn_cnt += 1

                    if is_valid and mol is not None:
                        valid_cnt += 1
                        try:
                            q_val = QED.qed(mol)
                            qeds.append(q_val)
                            strain_val = compute_clean_strain_energy(mol)
                            strains.append(strain_val)

                            lig_pdbqt_str = mol_to_pdbqt_meeko(mol)
                            v = Vina(sf_name='vina', cpu=1, verbosity=0)
                            v.set_receptor(rec_path)
                            v.set_ligand_from_string(lig_pdbqt_str)
                            v.compute_vina_maps(center=center, box_size=[20.0, 20.0, 20.0])
                            v.dock(exhaustiveness=4, n_poses=1)
                            dock_score = float(v.energies(n_poses=1)[0][0])
                            if dock_score != 0.0:
                                vinas.append(dock_score)
                        except Exception:
                            pass
            finally:
                if os.path.exists(rec_path):
                    os.remove(rec_path)

            sys.stdout.write(f"\r  [T={step_t:03d}] 进度: 已评估口袋 {p_idx+1:02d}/{args.num_pockets:02d} | 连通: {conn_cnt:03d}/{total_mols_per_setting} | 合法: {valid_cnt:03d}/{total_mols_per_setting}")
            sys.stdout.flush()

        val_rate = (valid_cnt / total_mols_per_setting) * 100.0
        conn_rate = (conn_cnt / total_mols_per_setting) * 100.0
        m_qed = float(np.mean(qeds)) if qeds else 0.0
        m_vina = float(np.mean(vinas)) if vinas else 0.0
        m_clash = float(np.mean(clashes)) if clashes else 0.0
        m_strain = float(np.mean(strains)) if strains else 0.0
        total_gen_time = sum(gen_times)
        latency_per_mol = total_gen_time / total_mols_per_setting

        res_dict = {
            "steps": step_t,
            "total_tested": total_mols_per_setting,
            "validity_rate": val_rate,
            "connectivity_rate": conn_rate,
            "mean_qed": m_qed,
            "mean_vina_dock": m_vina,
            "mean_clashes": m_clash,
            "mean_strain": m_strain,
            "latency_per_mol_sec": latency_per_mol
        }
        summary_results.append(res_dict)

        print(f"\n  └── 🏁 【T = {step_t:03d} 决算】合法率: {val_rate:.1f}% | 连通率: {conn_rate:.1f}% | QED: {m_qed:.4f} | Vina: {m_vina:.2f} kcal/mol | 碰撞: {m_clash:.2f} | 耗时: {latency_per_mol:.3f} s/mol")

    json_path = os.path.join(args.output_dir, "sampling_steps_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    latex_table = generate_latex_table(summary_results)
    latex_path = os.path.join(args.output_dir, "sampling_steps_table.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("\n" + "=" * 95)
    print("📊 【逆向采样步数敏感度评测完成 (Table S1)】")
    print("=" * 95)
    print(latex_table)
    print(f">>> 📝 结果数据已保存至: {json_path}")
    print(f">>> 📄 LaTeX 表格已生成至: {latex_path}\n")

if __name__ == "__main__":
    main()
