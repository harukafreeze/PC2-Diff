# Reproducing the manuscript workflows

Run commands from the repository root unless a command changes directory.

## Required inputs

- SBDD checkpoint at checkpoints/sbdd/latest_sbdd_model.pt. The checkpoint contains its model and dataset configuration.
- CrossDocked LMDB and split files at the directory referenced by that configuration.
- A saved 100-pocket, 10-sample cache. Its manifest.json maps pocket indices to NPZ files and stores file hashes, generation arguments and seeds.
- Target pocket/receptor files for LIT-PCBA and case studies, prepared with the scripts in docs/CODE_STRUCTURE.md.

Checkpoint weights and the saved coordinate cache must be supplied by the authors. The saved cache is required for replay of the frozen experimental cohort.

## Main benchmark

The following commands read the same saved sample cache. Each stage writes a new directory.

~~~bash
python -u scripts/server_pocket_geometry_bounded_trial.py   --cache results/repair_test_20261031   --out results/pc2diff_geometry   --dock --pockets 100 --workers 8

python -u scripts/server_balanced_workflow.py   --cache results/repair_test_20261031   --baseline results/pc2diff_geometry/comparison_docked   --out results/pc2diff_first_pass   --dock --pockets 100 --workers 8

python -u scripts/server_balanced_second_pass.py   --cache results/repair_test_20261031   --baseline results/pc2diff_first_pass/comparison_docked   --out results/pc2diff_second_pass   --pockets 100 --workers 8
~~~

The final comparison_docked directory stores samples.json, summary.json, run_metadata.json, and saved per-molecule structures. The balanced records identify the final outputs. The original records identify the preceding stage used in that comparison.

For new sampling, scripts/server_reconstruction_pilot.py generate accepts --checkpoint, --lmdb-dir, --split, --pockets and --samples. Use explicit split and output arguments. Existing caches preserve the cohort used for the frozen results.

## Five-group component ablation

The isolated directory preserves the experimental worker code and its vendored source snapshot.

~~~bash
cd ablation_isolated_20261102

python -u run_aligned_suite.py prepare   --table1-cache ../results/repair_test_20261031   --table1-final ../results/pc2diff_second_pass/comparison_docked   --out runs/aligned_suite_v1

bash run_aligned_suite_fast.sh
~~~

The suite uses the same pocket inputs, atom counts, checkpoint and seeds. The Full row is read from the final main-benchmark outputs. Variants 1–4 are generated and evaluated by the suite. This produces 1,000 attempts for each variant. The collect command saves five_group_summary.json. Descriptor and docking counts are recorded separately for each variant.

run_aligned_suite_fast.sh supports resume and uses 16 evaluation processes. Individual generate and compare commands accept their own arguments through --help.

## Other evaluations

The LIT-PCBA, template, sensitivity, training-time physics-bias, PoseBusters and allosteric entry points remain in docs/CODE_STRUCTURE.md. In the current manuscript, Table 2 is PoseBusters, Table 3 is LIT-PCBA, Table 4 is the 1,000-attempt component ablation and Table 5 is the independent training-time physics-bias comparison.

The preparation/evaluation scripts have different tasks. Check the input paths and task settings before running an entry point. Package versions and the availability of saved molecular intermediates affect exact replay.

## Source and result records

results/manuscript/crossdocked_molecules.json contains the 1,000 final molecular identities and recorded metrics. crossdocked_summary.json retains the received paired-stage summary. ablation_summary.json retains the received five-group summary. source_manifest.json records hashes of the added workflow source.

The package name and existing filenames are retained for checkpoint and import compatibility.
