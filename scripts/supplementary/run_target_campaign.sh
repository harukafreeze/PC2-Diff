#!/usr/bin/env bash
# scripts/supplementary/run_target_campaign.sh
# =====================================================================================
# One-command campaign for one allosteric target: sharded multi-GPU sampling, merge,
# validity funnel with docking, and the cage-versus-valid control experiment.
#
# Usage (from the repository root, inside build_env):
#     bash scripts/supplementary/run_target_campaign.sh SHP2_Allosteric 100000
#     bash scripts/supplementary/run_target_campaign.sh ABL1_Allosteric 100000
#
# Environment overrides:
#     GPUS=4 BATCH=512 PER_GROUP=1000 WORKERS=26 bash scripts/supplementary/run_target_campaign.sh <target> <N>
#
# Prerequisites: the pocket files for the target must exist, i.e.
#     real_world_study/data/allosteric_targets/pocket_<target>.pt and .pdbqt
# Build them with extract_allosteric_pocket.py if they are missing.
# =====================================================================================
set -euo pipefail

TARGET=${1:?usage: run_target_campaign.sh <target_name> [total_samples]}
TOTAL=${2:-100000}
GPUS=${GPUS:-4}
BATCH=${BATCH:-512}
PER_GROUP=${PER_GROUP:-1000}
WORKERS=${WORKERS:-26}
OUT="real_world_study/results/allosteric_screen_100k_${TARGET}"
DATA="real_world_study/data/allosteric_targets"

for f in "$DATA/pocket_${TARGET}.pt" "$DATA/pocket_${TARGET}.pdbqt"; do
    if [ ! -f "$f" ]; then
        echo "missing input: $f"
        echo "build it first, e.g.:"
        echo "  python scripts/supplementary/extract_allosteric_pocket.py --pdb <PDB> --ligand <LIG> --name ${TARGET}"
        exit 1
    fi
done

mkdir -p "$OUT"
echo "=============================================================================="
echo "campaign: $TARGET | samples $TOTAL | $GPUS GPUs | batch $BATCH | out $OUT"
echo "=============================================================================="

echo "[1/4] sharded sampling"
for i in $(seq 0 $((GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i python scripts/supplementary/sample_shards_multiGPU.py \
        --target "$TARGET" --rank "$i" --world_size "$GPUS" \
        --total_samples "$TOTAL" --batch_size "$BATCH" --out_dir "$OUT" &
done
wait

echo "[2/4] merging shards"
python scripts/supplementary/sample_shards_multiGPU.py --merge --out_dir "$OUT"

echo "[3/4] validity funnel and docking"
python scripts/supplementary/screen_allosteric_candidates.py \
    --target "$TARGET" --reuse_cache --out_dir "$OUT"

echo "[4/4] control experiment (valid versus rejected cages)"
python scripts/supplementary/compare_validity_effect.py \
    --target "$TARGET" --cache "$OUT/gen_cache.npz" --out_dir "$OUT" \
    --per_group "$PER_GROUP" --workers "$WORKERS"

echo "=============================================================================="
echo "done: $OUT"
echo "hand over the whole folder (gen_cache.npz optional)."
echo "=============================================================================="
