# PC²Diff

**Physical and Chemical Constraints Steer Diffusion Toward Structure-Based Molecular Generation**

PC²Diff combines pocket-conditioned equivariant diffusion, physical attention, aromatic-core guidance and chemically constrained graph construction. The package supports training, molecular construction, docking evaluation, component ablations and pocket-specific candidate selection.

## Installation

Use Python 3.10 or later. Install PyTorch for the CUDA version on the training machine, then run:

~~~bash
git clone https://github.com/harukafreeze/PC2-Diff.git
cd PC2-Diff
python -m pip install -e .
python -m pip install -r requirements.txt
~~~

The import package remains named project_build so existing scripts and checkpoints retain their paths.

## Current manuscript results

[results/manuscript](results/manuscript) contains the frozen CrossDocked2020 summary, 1,000 per-molecule records, the five-group ablation summary and source hashes.

~~~bash
python scripts/summarize_manuscript_results.py
~~~

This command reads the supplied records and reports the main benchmark and ablation means. The CrossDocked2020 final workflow has QED 0.6307, Vina Dock −8.31 kcal/mol, UFF strain 242.5 kcal/mol and 3.424 receptor clashes per molecule.

## Reproduction

[docs/REPRODUCING_MANUSCRIPT.md](docs/REPRODUCING_MANUSCRIPT.md) lists the input files, stage order and commands for the current workflow.

- Training uses scripts/run_8gpu.sh and configs/models/build_sbdd_full.yaml.
- Molecular construction uses cached pocket-conditioned samples, bounded graph recovery and two-pass conformer selection.
- The current component ablation uses ablation_isolated_20261102/run_aligned_suite_fast.py with 100 pockets and 10 attempts per pocket.
- LIT-PCBA and orthosteric/allosteric entry points are listed in docs/CODE_STRUCTURE.md.
- PoseBusters uses the existing prepared-structure workflow in scripts/evaluate_posebusters_full100.py.

Datasets, trained checkpoints and bulk coordinate caches are external inputs. Place them as described in the reproduction guide. Dependency ranges are in requirements.txt. Recorded evaluation package versions are listed separately in docs/EVALUATION_ENVIRONMENT.md.

Earlier summaries remain in their original results directories. Use results/manuscript for the current main benchmark and component ablation.
