#!/usr/bin/env bash
# End-to-end pipeline check on synthetic data with tiny random models (CPU, ~10 minutes).
# Verifies plumbing only. Its numbers are meaningless and must never be reported.
set -euo pipefail
cd "$(dirname "$0")/.."
export GAM_PROTOCOL=tests/smoke_protocol.yaml
WORK=$(mktemp -d)
sed "s|runs_dir: runs|runs_dir: $WORK/runs|" tests/smoke_protocol.yaml > "$WORK/protocol.yaml"
export GAM_PROTOCOL="$WORK/protocol.yaml"
run() {
  python3 scripts/run_matrix.py "$1" --synthetic > "$WORK/cmds_$1.txt"
  if [ "$1" = A ]; then
    sed -i 's/--steps [0-9]*/--steps 20/; s/--batch_size 256 --micro_batch 64/--batch_size 16 --micro_batch 8/; s/--warmup [0-9]*/--warmup 2 --eval_every 10/' "$WORK/cmds_$1.txt"
  fi
  while read -r c; do if ! eval "$c" > "$WORK/last.log" 2>&1; then echo "FAILED: $c"; tail -20 "$WORK/last.log"; return 1; fi; done < "$WORK/cmds_$1.txt"
  echo "phase $1 ok ($(wc -l < "$WORK/cmds_$1.txt") commands)"
}
for ph in A B1 B2 C D; do run $ph || { echo "SMOKE TEST FAILED in phase $ph"; exit 1; }; done
python3 scripts/benchmark.py --out "$WORK/runs/F/benchmark_fp32" --lengths 32,64 --hidden 64 --heads 4 --warmup 2 --iters 5 > /dev/null
python3 analysis/aggregate.py --runs "$WORK/runs" --out "$WORK/results" > /dev/null
python3 analysis/mass_analysis.py --runs "$WORK/runs" --out "$WORK/results" --synthetic > /dev/null
ls "$WORK/results"
echo "SMOKE TEST PASSED (outputs in $WORK)"
