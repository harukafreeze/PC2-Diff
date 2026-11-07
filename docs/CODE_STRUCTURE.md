# PC²Diff code guide

Current main-benchmark and 1,000-attempt ablation commands are in [REPRODUCING_MANUSCRIPT.md](REPRODUCING_MANUSCRIPT.md). The entries below include retained earlier evaluation workflows.

# Code guide

This release keeps the original source layout to preserve imports and relative paths. Core algorithms and recorded numerical summaries were copied without modification; the exploratory script's local cache path is configurable in this release.

## Training and evaluation

| Purpose | Entry point | Required inputs |
| --- | --- | --- |
| Distributed SBDD training | `scripts/train_target_sbdd_ddp.py`, `scripts/run_8gpu.sh` | YAML configuration, CrossDocked LMDB and split |
| Single-GPU training | `scripts/train_target_sbdd.py` | Configuration and dataset |
| Training without physics bias | `scripts/supplementary/train_ablation_no_physics_ddp.py` | Configuration and dataset |
| CrossDocked benchmark | `scripts/evaluate_official_vina_large_scale.py` | Dataset and checkpoint |
| Component ablation | `scripts/run_ablation_matrix_v2.py` | Dataset and checkpoint |
| Training ablation comparison | `scripts/run_physics_training_ablation.py` | With/without-physics checkpoints and dataset |
| Physics-bias evaluation helper | `scripts/run_physics_bias_ablation.py` | Checkpoint and dataset |
| Legacy reconstruction helper | `scripts/run_ablation_matrix.py` | Imported by the legacy branch of the preceding helper |
| LIT-PCBA preparation | `scripts/prepare_lit_pcba_dataset.py` | Network access to structural inputs |
| LIT-PCBA evaluation | `scripts/benchmark_lit_pcba_evaluation.py` | Prepared target files and checkpoint |
| PoseBusters | `scripts/evaluate_posebusters_full100.py` | Default checkpoint and dataset |
| Nucleation generalization | `scripts/test_generalized_nucleation.py` | Dataset and checkpoint |
| Sensitivity analysis | `scripts/test_sensitivity_*.py` | Dataset and checkpoint |

The `test_generalized_nucleation.py` and `test_sensitivity_*.py` files are research experiments, not lightweight unit tests. The existing unit tests are under `tests/`.

## Allosteric screening

Files are under `scripts/supplementary/`:

| File | Purpose |
| --- | --- |
| `extract_allosteric_pocket.py` | General PDB/ligand-based pocket preparation |
| `extract_true_allosteric_5d41.py` | EGFR allosteric pocket preparation |
| `sample_shards_multiGPU.py` | Sharded molecular generation and merging |
| `screen_allosteric_candidates.py` | Validity funnel and docking |
| `compare_validity_effect.py` | Comparison using valid and rejected structures from a shared cache |
| `run_target_campaign.sh` | Chain generation, merge, screening, and comparison |
| `validity_filters.py` | Additional validity-filter utility |
| `export_docked_complex.py` | Complex export utility |
| `analyze_5d41_lead_contacts.py` | EGFR lead contact analysis |
| `audit_chemical_novelty.py` | Novelty analysis |
| `audit_ligand_efficiency.py` | Ligand-efficiency analysis |

`compare_validity_effect.py` imports `screen_allosteric_candidates.py` from the same directory. Do not relocate either independently.

Target campaign inputs are expected at:

```text
real_world_study/data/allosteric_targets/
├── pocket_<target>.pt
└── pocket_<target>.pdbqt
```

Outputs are written under `real_world_study/results/`. For an EGFR-specific script, inspect its own input paths: not every historical EGFR entry uses the generalized campaign directory. The orthosteric case study uses `real_world_study/extract_pocket_and_baseline.py` and `real_world_study/run_combat_500_molecules.py`, with inputs under `real_world_study/data/`.

Novelty and ligand-efficiency analyses may read `results/figures/global_1000_mols_cache.json` and other generated intermediates. These caches are not included; do not interpret missing cache inputs as an empty result.

## Exploratory seed analysis

`results/exploratory_allosteric_pose_20261020/run_exploration.py` is retained because it supports the exploratory supplementary analysis. Its location is preserved to keep root-relative paths working.

- `BUILD_POSE_CACHE` selects the working/cache directory (default: `~/.cache/build-pose-exploration`).
- The script expects `vina_1.2.7_win.exe` inside that directory and is currently a Windows-specific external-binary workflow.
- It reads selected lead PDBQT files under `real_world_study/results/allosteric_screen_100k/`, plus receptor/reference inputs under `real_world_study/data/`.
- It is an exploratory analysis, not part of model training. Required input structures and the executable are not distributed in this update.

## Packaging notes

- `project_build/data/` is Python source and is intentionally tracked. Root datasets and target input files are ignored separately.
- The complete core package is retained because some seemingly historical modules are imported by active components.
- Requirements are version ranges, not a frozen reproducibility environment. `torch-geometric` is optional for the QM9 loader.
- Publication plotting and one-off PDF layout fixes are excluded; they require a separate data-and-figure packaging pass.
- This organization update did not run training, docking, evaluation, or unit tests.
