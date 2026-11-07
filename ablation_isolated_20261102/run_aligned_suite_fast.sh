#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
suite="runs/aligned_suite_v1"
for variant in 1 2 3 4; do
  if [ -f "$suite/variant_$variant/summary.json" ]; then
    echo "Variant $variant already summarized; retaining completed results."
    continue
  fi
  generation_args=()
  scoring_args=()
  if [ -f "$suite/cache_v$variant/manifest.json" ]; then generation_args+=(--resume); fi
  if [ -f "$suite/variant_$variant/run_metadata.json" ]; then scoring_args+=(--resume); fi
  python -u run_aligned_suite.py generate --suite "$suite" --variant "$variant" \
    --checkpoint ../checkpoints/sbdd/latest_sbdd_model.pt "${generation_args[@]}"
  python -u run_aligned_suite_fast.py compare --suite "$suite" --variant "$variant" \
    --workers 16 "${scoring_args[@]}"
done
python -u run_aligned_suite.py collect --suite "$suite"
