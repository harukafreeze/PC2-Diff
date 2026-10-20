# -*- coding: utf-8 -*-
"""
PG-CrossAttention physics-bias ablation, v2 (paired, multi-condition).

What this fixes relative to v1
------------------------------
1. Reconstruction now uses project_build.postprocess.bond_perception.
   BondPerceptionEngine -- the SAME engine the main benchmark uses, including
   the quaternary-ammonium formal-charge balance and the C=O / S=O double-bond
   upgrade. v1 used scripts.run_ablation_matrix.reconstruct_ablation, which
   lacks the charge balance, so its "physics on" row could never reproduce the
   stored ablation table (validity 90.0% here vs 100.0% stored, with RDKit
   "Explicit valence for atom #N, 4" errors). Pass --reconstruction legacy to
   reproduce the old behaviour.
2. The bias kernel is now selectable, so the distance-scale question can be
   settled in one run:
       off      : bias bypassed (use_physics_bias = False)
       shipped  : LennardJonesCoulombBias, constant sigma = 1.70 A
       pairwise : PairwiseSoftCoreLJBias, sigma_ij = r_i^vdW + r_j^vdW
                  (C...C -> 3.40 A), i.e. the physical contact distance
   All three load the SAME checkpoint with strict=True (parameter names are
   identical), and all three see the same pockets and the same initial noise.

Why the kernel matters
----------------------
In the 12-6 form used here sigma is the zero-crossing (contact) distance, not a
single-atom radius. With the shipped constant sigma = 1.70 A the zero crossing
sits at 1.63 A and the -50 clamp at ~1.10 A, both far inside the 1.5 A clash
definition, and the bias is slightly ATTRACTIVE over 1.7-3.5 A -- the band
where real contacts form. The script prints this profile for both kernels
before sampling, so the numbers can be read against it.

Location
--------
scripts/run_physics_bias_ablation.py

Usage
-----
python scripts/run_physics_bias_ablation.py \
    --checkpoint checkpoints/sbdd/latest_sbdd_model.pt \
    --num_pockets 20 --samples_per_pocket 5 \
    --output_dir results/ablation_physics --seed 42

Outputs
-------
<output_dir>/physics_bias_ablation_summary.json
<output_dir>/physics_bias_ablation_table.tex
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from rdkit import RDLogger
from rdkit.Chem import QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (ROOT_DIR, SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import project_build.models.backbones  # noqa: E402, F401
import project_build.physics  # noqa: E402, F401
from project_build.core.registry import BACKBONE_REGISTRY  # noqa: E402
from project_build.data.datasets.crossdocked import CrossDockedLMDBDataset  # noqa: E402
from project_build.diffusion.sde.vp_sde import DualSpaceVPSDE  # noqa: E402
from project_build.diffusion.solvers.reverse_sde_sampler import ReverseSDESampler  # noqa: E402
from project_build.evaluation.docking_vina import TargetDockingEvaluator  # noqa: E402
from project_build.models.attention.pg_attention_bias import (  # noqa: E402
    LIGAND_VDW_RADII,
)
from project_build.models.backbones.target_sbdd_layer import LocalRadialSBDDLayer  # noqa: E402
from project_build.postprocess.bond_perception import BondPerceptionEngine  # noqa: E402

METRIC_ORDER = [
    "validity",
    "connectivity",
    "mean_qed",
    "mean_arom",
    "mean_vina",
    "mean_clashes",
    "mean_strain",
]

METRIC_DECIMALS = {
    "validity": 1,
    "connectivity": 1,
    "mean_qed": 4,
    "mean_arom": 2,
    "mean_vina": 2,
    "mean_clashes": 2,
    "mean_strain": 1,
}

# name -> (use_physics_bias, attention-bias type or None to keep the checkpoint's)
CONDITIONS: Dict[str, Dict[str, Any]] = {
    "off": {"use_physics_bias": False, "bias_type": None,
            "label": "Full BUILD 3.0 (physics off)"},
    "shipped": {"use_physics_bias": True, "bias_type": "LennardJonesCoulombBias",
                "label": "Full BUILD 3.0 (shipped kernel, sigma=1.70 A)"},
    "pairwise": {"use_physics_bias": True, "bias_type": "PairwiseSoftCoreLJBias",
                 "label": "Full BUILD 3.0 (element-pairwise kernel)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired multi-condition ablation of the PG-CrossAttention physics bias."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sbdd/latest_sbdd_model.pt")
    parser.add_argument("--num_pockets", type=int, default=20)
    parser.add_argument("--samples_per_pocket", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/ablation_physics")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reset before every condition so all share identical initial noise.")
    parser.add_argument("--conditions", type=str, default="off,shipped,pairwise",
                        help="Comma-separated subset of: off, shipped, pairwise.")
    parser.add_argument("--sigma_scale", type=float, default=1.0,
                        help="Multiplier on sigma for the pairwise kernel (sweep knob).")
    parser.add_argument("--reconstruction", type=str, default="engine", choices=["engine", "legacy"],
                        help="engine = BondPerceptionEngine (main benchmark); legacy = ablation harness.")
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--deterministic", action="store_true",
                        help="cudnn.deterministic=True (bitwise reproducibility; slower).")
    return parser.parse_args()


def analytic_gamma(sigma: float, d: float, kb_t: float, epsilon: float,
                   delta: float, lo: float = -50.0, hi: float = 10.0) -> float:
    """gamma(d) for a 12-6 soft-core kernel, matching pg_attention_bias exactly."""
    r_soft_sq = d * d + delta * delta
    ratio = (sigma * sigma) / r_soft_sq
    lj_att = ratio ** 3
    lj_rep = lj_att ** 2
    v_vdw = 4.0 * epsilon * (lj_rep - lj_att)
    return max(lo, min(hi, -v_vdw / kb_t))


def print_gamma_profiles(kb_t: float, epsilon: float, delta: float) -> None:
    c_c_pair = LIGAND_VDW_RADII[0] + LIGAND_VDW_RADII[0]   # C...C
    print("Analytic bias profile  Gamma(d)  [clamp -50 .. +10]")
    print("    d(A)   sigma=1.70   sigma=%.2f (C...C pairwise)" % c_c_pair)
    for d in (1.10, 1.20, 1.30, 1.50, 1.80, 2.00, 2.50, 3.00, 3.40, 4.00):
        a = analytic_gamma(1.70, d, kb_t, epsilon, delta)
        b = analytic_gamma(c_c_pair, d, kb_t, epsilon, delta)
        print("   %5.2f   %9.3f   %9.3f" % (d, a, b))
    print("  (clash criterion = heavy-atom contact below 1.50 A)\n")


def build_model(model_cfg: Dict[str, Any], cond: Dict[str, Any], state: Dict[str, Any],
                device: torch.device, sigma_scale: float) -> torch.nn.Module:
    mcfg = copy.deepcopy(model_cfg)
    mcfg["use_physics_bias"] = bool(cond["use_physics_bias"])

    bias_type = cond["bias_type"]
    if bias_type is not None:
        bias_cfg = dict(mcfg.get("physics_bias_cfg") or {})
        bias_cfg["type"] = bias_type
        if bias_type == "PairwiseSoftCoreLJBias":
            # The constant kernel's sigma is an atom radius; the pairwise kernel
            # derives sigma from the radii tables, so this is only a fallback.
            bias_cfg["sigma_default"] = LIGAND_VDW_RADII[0] + LIGAND_VDW_RADII[0]
            bias_cfg["sigma_scale"] = float(sigma_scale)
        else:
            bias_cfg.pop("sigma_scale", None)
        mcfg["physics_bias_cfg"] = bias_cfg

    model = BACKBONE_REGISTRY.build(mcfg).to(device)
    model.load_state_dict(state)   # param names identical across kernels -> strict load
    model.eval()
    return model


def count_switches(model: torch.nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, LocalRadialSBDDLayer))


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate(
    model: torch.nn.Module,
    sde: DualSpaceVPSDE,
    dataset: CrossDockedLMDBDataset,
    num_pockets: int,
    samples_per_pocket: int,
    seed: int,
    evaluator: TargetDockingEvaluator,
    device: torch.device,
    reconstruction: str,
    tolerance: float,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sampler = ReverseSDESampler(
        sde=sde, num_steps=100, spring_k=0.25, corrector_steps=1,
        bond_gate_thresh=0.35, enable_svd_seeding=True,
        enable_self_healing=True, device=device,
    )

    total = num_pockets * samples_per_pocket
    valid_cnt = conn_cnt = 0
    qeds: List[float] = []
    aroms: List[float] = []
    vinas: List[float] = []
    clashes: List[float] = []
    strains: List[float] = []

    for p_idx in range(num_pockets):
        sample = dataset[p_idx]
        p_pos = sample["pocket_pos"].unsqueeze(0).to(device)
        p_h = sample["pocket_h"].unsqueeze(0).to(device)
        n_atoms = max(18, int(sample["num_lig_atoms"]))

        with torch.no_grad():
            x_gen, h_gen, _ = sampler.sample(
                model=model,
                num_samples=samples_per_pocket,
                num_atoms=n_atoms,
                num_atom_types=9,
                model_kwargs={
                    "pocket_pos": p_pos.repeat(samples_per_pocket, 1, 1),
                    "pocket_h": p_h.repeat(samples_per_pocket, 1, 1),
                },
            )

        for b in range(samples_per_pocket):
            if reconstruction == "engine":
                mol, is_valid, n_frag = BondPerceptionEngine.reconstruct_mol(
                    pos=x_gen[b], atom_types=h_gen[b], tolerance=tolerance,
                )
            else:
                from run_ablation_matrix import reconstruct_ablation
                mol, is_valid, n_frag = reconstruct_ablation(
                    pos=x_gen[b], atom_types=h_gen[b],
                    use_mst=True, use_sp2_budget=True, use_kekule=True,
                    use_forbidden_filter=True, use_triangulation_suppression=True,
                    tolerance=tolerance,
                )

            clashes.append(float(evaluator.compute_pocket_clashes(x_gen[b], p_pos[0])))
            if n_frag == 1:
                conn_cnt += 1
            if is_valid and mol is not None:
                valid_cnt += 1
                try:
                    qeds.append(float(QED.qed(mol)))
                    aroms.append(float(rdMolDescriptors.CalcNumAromaticRings(mol)))
                    vinas.append(float(evaluator.compute_official_vina_score(mol, p_pos[0], p_h[0])))
                    strains.append(float(evaluator.compute_strain_energy(mol)))
                except Exception:
                    pass

        print(f"    pocket {p_idx + 1:3d}/{num_pockets} done")

    return {
        "validity": (valid_cnt / total) * 100.0,
        "connectivity": (conn_cnt / total) * 100.0,
        "mean_qed": _mean(qeds),
        "mean_arom": _mean(aroms),
        "mean_vina": _mean(vinas),
        "mean_clashes": _mean(clashes),
        "mean_strain": _mean(strains),
        "n_total": total,
        "n_valid": valid_cnt,
        "n_connected": conn_cnt,
        "n_qed": len(qeds),
        "n_vina": len(vinas),
        "n_strain": len(strains),
    }


def _cell(value: float, decimals: int) -> str:
    if isinstance(value, float) and value != value:   # NaN
        return "--"
    return f"{value:.{decimals}f}"


def generate_latex_table(results: Dict[str, Dict[str, Any]], labels: Dict[str, str]) -> str:
    lines = [
        "% ==========================================================================",
        "% PG-CrossAttention bias ablation (auto-generated, paired conditions)",
        "% ==========================================================================",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Paired ablation of the PG-CrossAttention bias on CrossDocked2020. Every "
        "condition sees the same pockets and the same initial noise and uses the main-benchmark "
        "graph-reconstruction engine; only the soft-core Lennard-Jones cross-attention bias "
        "changes. \\textbf{Clashes} are the primary readout of the bias.}",
        "\\label{tab:physics_bias_ablation}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Model Variant} & \\textbf{Valid (\\%)} $\\uparrow$ & \\textbf{Conn (\\%)} $\\uparrow$ & "
        "\\textbf{QED} $\\uparrow$ & \\textbf{Arom} $\\uparrow$ & \\textbf{Vina} $\\downarrow$ & "
        "\\textbf{Clashes} $\\downarrow$ & \\textbf{Strain} $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for key, metrics in results.items():
        cells = [labels[key]] + [_cell(metrics[m], METRIC_DECIMALS[m]) for m in METRIC_ORDER]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown condition(s): {unknown}; valid: {sorted(CONDITIONS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("PG-CrossAttention physics-bias ablation v2")
    print(f"checkpoint={args.checkpoint}  device={device}  seed={args.seed}")
    print(f"conditions={requested}  reconstruction={args.reconstruction}")
    print("=" * 90)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("checkpoint has no 'config.model' dict; cannot rebuild the backbone")
    state = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("checkpoint has neither 'ema_state_dict' nor 'model_state_dict'")

    base_bias_cfg = dict(model_cfg.get("physics_bias_cfg") or {})
    kb_t = float(base_bias_cfg.get("kb_t", 0.6))
    epsilon = float(base_bias_cfg.get("epsilon_default", 0.15))
    delta = float(base_bias_cfg.get("delta_vdw", 0.5))
    print_gamma_profiles(kb_t, epsilon, delta)

    sde = DualSpaceVPSDE(schedule_type=cfg.get("diffusion", {}).get("schedule_type", "cosine")).to(device)
    dataset = CrossDockedLMDBDataset(lmdb_dir=cfg["dataset"]["lmdb_dir"], split="test", split_type="name")
    evaluator = TargetDockingEvaluator(clash_cutoff=1.5)

    num_pockets = min(args.num_pockets, len(dataset))
    if num_pockets <= 0:
        raise RuntimeError("test set is empty; cannot run the ablation")
    print(f"pockets={num_pockets}  samples/pocket={args.samples_per_pocket}  "
          f"total/condition={num_pockets * args.samples_per_pocket}\n")

    results: Dict[str, Dict[str, Any]] = {}
    labels: Dict[str, str] = {}

    for i, key in enumerate(requested, 1):
        cond = CONDITIONS[key]
        print(f"[{i}/{len(requested)}] {key}")
        model = build_model(model_cfg, cond, state, device, args.sigma_scale)
        n_switch = count_switches(model)
        print(f"    bias layers: {n_switch}  use_physics_bias={cond['use_physics_bias']}")

        t0 = time.time()
        summary = evaluate(
            model, sde, dataset, num_pockets, args.samples_per_pocket,
            args.seed, evaluator, device, args.reconstruction, args.tolerance,
        )
        summary["condition"] = key
        summary["use_physics_bias"] = bool(cond["use_physics_bias"])
        summary["bias_type"] = cond["bias_type"]
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        results[key] = summary
        labels[key] = cond["label"]
        print(f"    -> clashes={summary['mean_clashes']:.2f}  valid={summary['validity']:.1f}%  "
              f"qed={summary['mean_qed']:.4f}  vina={summary['mean_vina']:.2f}  "
              f"strain={summary['mean_strain']:.1f}\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "meta": {
            "checkpoint": os.path.abspath(args.checkpoint),
            "device": str(device),
            "seed": args.seed,
            "num_pockets": num_pockets,
            "samples_per_pocket": args.samples_per_pocket,
            "total_per_condition": num_pockets * args.samples_per_pocket,
            "reconstruction": args.reconstruction,
            "conditions": requested,
            "sigma_scale": args.sigma_scale,
        },
        "results": results,
    }

    json_path = os.path.join(args.output_dir, "physics_bias_ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    latex_table = generate_latex_table(results, labels)
    tex_path = os.path.join(args.output_dir, "physics_bias_ablation_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("=" * 90)
    print("RESULT")
    print("=" * 90)
    header = "  %-12s" % "condition" + "".join("%12s" % m.replace("mean_", "") for m in METRIC_ORDER)
    print(header)
    for key in requested:
        r = results[key]
        row = "  %-12s" % key + "".join("%12s" % _cell(r[m], METRIC_DECIMALS[m]) for m in METRIC_ORDER)
        print(row)

    if "shipped" in results and "off" in results:
        on, off = results["shipped"], results["off"]
        print(f"\n  shipped bias vs off: clashes {on['mean_clashes']:.2f} vs {off['mean_clashes']:.2f} "
              f"(delta {on['mean_clashes'] - off['mean_clashes']:+.2f})")
    if "pairwise" in results and "off" in results:
        pw, off = results["pairwise"], results["off"]
        print(f"  pairwise bias vs off: clashes {pw['mean_clashes']:.2f} vs {off['mean_clashes']:.2f} "
              f"(delta {pw['mean_clashes'] - off['mean_clashes']:+.2f})")

    print(f"\nJSON  -> {json_path}")
    print(f"LaTeX -> {tex_path}\n")
    print(latex_table)


if __name__ == "__main__":
    main()
