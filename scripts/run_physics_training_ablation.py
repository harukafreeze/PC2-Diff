# -*- coding: utf-8 -*-
"""
Training-time ablation of the PG-CrossAttention physics bias.

Why this exists
---------------
The inference-time ablation (scripts/run_physics_bias_ablation.py) showed that
switching the soft-core Lennard-Jones bias on or off AT SAMPLING TIME does not
change the receptor clash count. Two independent kernels were tested -- the
shipped constant-sigma kernel and an element-pairwise kernel whose repulsive
wall sits at the physical contact distance -- and clashes moved by +0.04 and
+0.39 per molecule respectively, i.e. not in the claimed direction.

That leaves one live hypothesis: the bias may act as a TRAINING-TIME inductive
prior. The training objective is evaluated on noised coordinates, where atom
pairs routinely fall inside the repulsive core, so the bias can shape the
learned vector field even though it is nearly inert in the final sampling
regime. Testing that hypothesis requires a second training run -- which is what
this script compares.

What this script does
---------------------
1. Loads two checkpoints that (should) differ only in `model.use_physics_bias`.
2. Rebuilds each model from ITS OWN saved config, so each one runs with the
   bias state it was trained with.
3. Samples the same pockets with the same seed under each (paired), using the
   main-benchmark reconstruction engine and the shared evaluator.
4. Prints the config diff so the "only one variable changed" assumption is
   auditable, then writes a metrics table.

With --cross it additionally samples {trained-with, bias off} and
{trained-without, bias on}, which separates the training effect from the
inference effect.

Prerequisite
------------
The second checkpoint is produced by:
    bash scripts/run_8gpu.sh --config configs/models/build_sbdd_no_physics.yaml

Usage
-----
python scripts/run_physics_training_ablation.py \
    --checkpoint-with    checkpoints/sbdd/latest_sbdd_model.pt \
    --checkpoint-without checkpoints/sbdd_no_physics/latest_sbdd_model.pt \
    --num_pockets 20 --samples_per_pocket 5 --seed 42 \
    --output_dir results/ablation_physics_training

Outputs
-------
<output_dir>/physics_training_ablation_summary.json
<output_dir>/physics_training_ablation_table.tex
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import RDLogger

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
from project_build.evaluation.docking_vina import TargetDockingEvaluator  # noqa: E402

# Shared with the inference-time ablation so both report identical metrics.
from run_physics_bias_ablation import (  # noqa: E402
    METRIC_DECIMALS,
    METRIC_ORDER,
    evaluate,
)

# Config keys that are ALLOWED to differ between the two runs. Anything else
# that differs means the comparison is not controlled, and is reported loudly.
BENIGN_CONFIG_KEYS = {
    "model.use_physics_bias",
    "experiment_name",
    "training.checkpoint_dir",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired comparison of two checkpoints that differ only in the physics bias."
    )
    parser.add_argument("--checkpoint_with", type=str, default="checkpoints/sbdd/latest_sbdd_model.pt",
                        help="Checkpoint trained WITH the PG-CrossAttention bias.")
    parser.add_argument("--checkpoint_without", type=str, default="checkpoints/sbdd_no_physics/latest_sbdd_model.pt",
                        help="Checkpoint trained WITHOUT the bias (see build_sbdd_no_physics.yaml).")
    parser.add_argument("--num_pockets", type=int, default=20)
    parser.add_argument("--samples_per_pocket", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/ablation_physics_training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cross", action="store_true",
                        help="Also run {with-trained, bias off} and {without-trained, bias on}.")
    parser.add_argument("--reconstruction", type=str, default="engine", choices=["engine", "legacy"])
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, full))
        else:
            out[full] = value
    return out


def config_diff(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    fa, fb = _flatten(cfg_a), _flatten(cfg_b)
    diffs: List[Tuple[str, Any, Any]] = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, "<missing>"), fb.get(key, "<missing>")
        if va != vb:
            diffs.append((key, va, vb))
    return diffs


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if "config" not in ckpt:
        raise KeyError(f"{path} has no 'config' entry; cannot rebuild its backbone")
    if ckpt.get("ema_state_dict") is None and ckpt.get("model_state_dict") is None:
        raise KeyError(f"{path} has neither 'ema_state_dict' nor 'model_state_dict'")
    return ckpt


def build_model(ckpt: Dict[str, Any], device: torch.device,
                force_use_physics: Optional[bool] = None) -> Tuple[torch.nn.Module, bool]:
    model_cfg = copy.deepcopy(ckpt["config"].get("model"))
    if not isinstance(model_cfg, dict):
        raise ValueError("checkpoint config has no 'model' dict")
    trained_with = bool(model_cfg.get("use_physics_bias", True))
    if force_use_physics is not None:
        model_cfg["use_physics_bias"] = bool(force_use_physics)
    model = BACKBONE_REGISTRY.build(model_cfg).to(device)
    state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state)   # bias params exist in both variants -> strict load
    model.eval()
    return model, trained_with


def _cell(value: float, decimals: int) -> str:
    if isinstance(value, float) and value != value:
        return "--"
    return f"{value:.{decimals}f}"


def generate_latex_table(results: Dict[str, Dict[str, Any]], labels: Dict[str, str]) -> str:
    lines = [
        "% ==========================================================================",
        "% Training-time PG-CrossAttention bias ablation (auto-generated)",
        "% ==========================================================================",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Training-time ablation of the PG-CrossAttention bias on CrossDocked2020. The two "
        "models are trained from the same configuration except for the physics bias; both are "
        "sampled on the same pockets with the same initial noise and the main-benchmark graph "
        "reconstruction. \\textbf{Clashes} are the readout of the bias.}",
        "\\label{tab:physics_training_ablation}",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 90)
    print("Training-time physics-bias ablation")
    print(f"  with    : {args.checkpoint_with}")
    print(f"  without : {args.checkpoint_without}")
    print(f"  device={device}  seed={args.seed}  cross={args.cross}")
    print("=" * 90)

    ckpt_with = load_checkpoint(args.checkpoint_with, device)
    ckpt_without = load_checkpoint(args.checkpoint_without, device)
    cfg_with, cfg_without = ckpt_with["config"], ckpt_without["config"]

    trained_with_bias = bool(cfg_with.get("model", {}).get("use_physics_bias", True))
    trained_without_bias = bool(cfg_without.get("model", {}).get("use_physics_bias", True))
    print(f"\ntrained use_physics_bias: with={trained_with_bias}  without={trained_without_bias}")
    if trained_without_bias:
        print("  !! WARNING: the 'without' checkpoint also trained WITH the bias enabled.")
        print("     Did you launch `bash scripts/run_8gpu.sh --config configs/models/build_sbdd_no_physics.yaml`?")

    diffs = config_diff(cfg_with, cfg_without)
    unexpected = [d for d in diffs if d[0] not in BENIGN_CONFIG_KEYS]
    print(f"\nconfig diff ({len(diffs)} key(s)):")
    for key, a, b in diffs:
        flag = "ok " if key in BENIGN_CONFIG_KEYS else "!! "
        print(f"  {flag}{key}: {a} -> {b}")
    if unexpected:
        print("\n  !! WARNING: the two runs differ in keys beyond model.use_physics_bias.")
        print("     The comparison is only controlled if training differed in that one switch.")
    print()

    dataset_cfg = cfg_with.get("dataset", {})
    if dataset_cfg.get("lmdb_dir") != cfg_without.get("dataset", {}).get("lmdb_dir"):
        print("  !! WARNING: the two configs point at different datasets.")
    sde = DualSpaceVPSDE(schedule_type=cfg_with.get("diffusion", {}).get("schedule_type", "cosine")).to(device)
    dataset = CrossDockedLMDBDataset(lmdb_dir=dataset_cfg["lmdb_dir"], split="test", split_type="name")
    evaluator = TargetDockingEvaluator(clash_cutoff=1.5)

    num_pockets = min(args.num_pockets, len(dataset))
    if num_pockets <= 0:
        raise RuntimeError("test set is empty; cannot run the ablation")

    conditions: List[Tuple[str, Dict[str, Any], Optional[bool], str]] = [
        ("trained_with", ckpt_with, None, "Trained with bias, sampled with bias"),
        ("trained_without", ckpt_without, None, "Trained without bias, sampled without bias"),
    ]
    if args.cross:
        conditions += [
            ("trained_with__bias_off", ckpt_with, False, "Trained with bias, sampled without bias"),
            ("trained_without__bias_on", ckpt_without, True, "Trained without bias, sampled with bias"),
        ]

    print(f"pockets={num_pockets}  samples/pocket={args.samples_per_pocket}  "
          f"total/condition={num_pockets * args.samples_per_pocket}\n")

    results: Dict[str, Dict[str, Any]] = {}
    labels: Dict[str, str] = {}

    for i, (key, ckpt, force, label) in enumerate(conditions, 1):
        print(f"[{i}/{len(conditions)}] {key}")
        model, _ = build_model(ckpt, device, force_use_physics=force)
        t0 = time.time()
        summary = evaluate(
            model, sde, dataset, num_pockets, args.samples_per_pocket,
            args.seed, evaluator, device, args.reconstruction, args.tolerance,
        )
        summary["condition"] = key
        summary["trained_with_bias"] = bool(ckpt["config"].get("model", {}).get("use_physics_bias", True))
        summary["sampled_with_bias"] = bool(
            ckpt["config"].get("model", {}).get("use_physics_bias", True) if force is None else force
        )
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        results[key] = summary
        labels[key] = label
        print(f"    -> clashes={summary['mean_clashes']:.2f}  valid={summary['validity']:.1f}%  "
              f"qed={summary['mean_qed']:.4f}  vina={summary['mean_vina']:.2f}  "
              f"strain={summary['mean_strain']:.1f}\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "meta": {
            "checkpoint_with": os.path.abspath(args.checkpoint_with),
            "checkpoint_without": os.path.abspath(args.checkpoint_without),
            "device": str(device),
            "seed": args.seed,
            "num_pockets": num_pockets,
            "samples_per_pocket": args.samples_per_pocket,
            "reconstruction": args.reconstruction,
            "config_diff": [{"key": k, "with": str(a), "without": str(b)} for k, a, b in diffs],
            "unexpected_config_diff": [k for k, _, _ in unexpected],
        },
        "results": results,
    }

    json_path = os.path.join(args.output_dir, "physics_training_ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    latex_table = generate_latex_table(results, labels)
    tex_path = os.path.join(args.output_dir, "physics_training_ablation_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("=" * 90)
    print("RESULT")
    print("=" * 90)
    print("  %-26s" % "condition" + "".join("%12s" % m.replace("mean_", "") for m in METRIC_ORDER))
    for key in results:
        r = results[key]
        print("  %-26s" % key + "".join("%12s" % _cell(r[m], METRIC_DECIMALS[m]) for m in METRIC_ORDER))

    a, b = results["trained_with"], results["trained_without"]
    print(f"\n  TRAINING-TIME EFFECT ON CLASHES")
    print(f"    trained with bias    : {a['mean_clashes']:.2f} per molecule")
    print(f"    trained without bias : {b['mean_clashes']:.2f} per molecule")
    print(f"    difference (without - with) = {b['mean_clashes'] - a['mean_clashes']:+.2f}")
    if b["mean_clashes"] - a["mean_clashes"] > 0.5:
        print("    -> the bias acts as a training-time prior; the contribution is supported.")
    else:
        print("    -> no material training-time effect either; re-attribute the contribution.")

    print(f"\nJSON  -> {json_path}")
    print(f"LaTeX -> {tex_path}\n")
    print(latex_table)


if __name__ == "__main__":
    main()
