# Project BUILD 3.0: Physics-Grounded & Topology-Conserved Equivariant Diffusion for Structure-Based Drug Design

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official implementation and benchmark suite for **Project BUILD 3.0**, an equivariant generative diffusion architecture for 3D structure-based de novo drug design (SBDD).

---

## 🌟 Key Methodological Highlights

- **PG-CrossAttention**: Embeds a 12-6 Lennard-Jones soft-core potential bias $\mathbf{\Gamma}_{ij}$ directly into equivariant radial cross-attention logits, driving receptor-ligand steric clashes down to **3.75** clashes/molecule on CrossDocked2020 (surpassing the natural co-crystal reference of 5.51).
- **SVD Cavity Principal Axis Nucleation**: Employs singular value decomposition on receptor cavity covariance tensors to align pharmacophore scaffolds along principal cavity openings, combined with rigid regular-polygon manifold projections (achieving **99.6%** aromatic ring planarity in PoseBusters).
- **Continuous Quantum Valency Potential**: Introduces a Fermi-like continuous bond probability field coupled with Min-SNR dynamic annealing, boosting 3D covalent bond length pass rates to **94.0%** and bond angles to **86.2%**.
- **Component Supergraph MST & $sp^2$ Hybridization Conservation**: A discrete graph-theoretic minimum spanning tree engine that guarantees **100.0% single molecular connectivity** without pentavalent carbon errors, elevating de novo drug-likeness to a mean QED of **0.6238** (median **0.65**).

---

## 📁 Repository Structure

```text
Project_BUILD/
├── configs/models/build_sbdd_full.yaml     # Model & diffusion hyperparameters
├── project_build/                          # Core Python package
│   ├── core/registry.py                    # Modular component registry
│   ├── data/datasets/crossdocked.py        # LMDB multi-worker safe dataset loader
│   ├── models/backbones/                   # Equivariant backbone & PG-CrossAttention
│   ├── physics/                            # Quantum valency fields & SNR schedulers
│   ├── diffusion/                          # VP-SDE engine & Reverse SDE SVD sampler
│   ├── postprocess/bond_perception.py      # Component Supergraph MST & Kekule engine
│   └── evaluation/                         # Vina 1.2.7 C++, Kabsch, & Chem metrics
├── scripts/                                # Official production benchmark scripts
│   ├── train_target_sbdd_ddp.py            # Distributed 8-GPU training controller
│   ├── evaluate_official_vina_large_scale.py # Master Table 1 benchmark (1000 molecules)
│   ├── run_ablation_matrix_v2.py           # Table 2 stepwise causal ablation
│   ├── benchmark_lit_pcba_evaluation.py    # Table 3 LIT-PCBA zero-shot generalization
│   ├── test_generalized_nucleation.py      # Table 4 multi-scaffold template test
│   ├── evaluate_posebusters_full100.py     # Table 5 PoseBusters 17-criterion suite
│   ├── generate_interaction_heatmaps.py    # Figure 2 & 4 biophysical heatmaps
│   └── test_sensitivity_*.py               # Tables S1~S4 hyperparameter sensitivity
└── results/                                # LaTeX tables & JSON benchmark logs
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone git@github.com:<your_username>/<your_repo>.git
cd Project_BUILD
pip install -e .
```

### 2. Reproducing Benchmark Tables
```bash
# Master Table 1 (CrossDocked2020 1000 molecules)
python scripts/evaluate_official_vina_large_scale.py --checkpoint checkpoints/sbdd/latest_sbdd_model.pt

# Table 2 (Stepwise Causal Ablation)
python scripts/run_ablation_matrix_v2.py --vina_cpu 8

# Table 3 (LIT-PCBA 15 Clinical Targets)
python scripts/benchmark_lit_pcba_evaluation.py --samples_per_target 100

# Table 5 (Full PoseBusters 17-criterion Suite)
python scripts/evaluate_posebusters_full100.py
```
