#!/bin/bash
# Independent replicates of the original (no-relax, 64 draws per Z, rank by total energy) comparison.
# Replicate seed 0 = campaigns/demo_2026-11-25_{llm,random} + demo_2026-11-26_nemotron; this adds seeds 1 and 2.
# usage: run_replicates.sh ARM GPU SEED [SEED ...]     ARM = gpt | random | nemotron
set -u
cd "$(dirname "$0")/.."
PY=${AZA_PY:-/mnt/data1/users/ffmmgg/venvs/clari313/bin/python}
export OMP_NUM_THREADS=2
ARM=$1; GPU=$2; shift 2
OUT=${OUT:-campaigns/replicates}
mkdir -p $OUT
case $ARM in
  gpt)      OPTS="--policy llm" ;;
  random)   OPTS="--policy random" ;;
  nemotron) OPTS="--policy llm --provider openrouter --model nvidia/nemotron-3.5-super-vl:free" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
for SEED in "$@"; do
  D=$OUT/${ARM}_seed$SEED
  [ -d $D ] || $PY -m aza_agent.campaign init --dir $D $OPTS --budget 30 --seed $SEED --gpu $GPU
  echo "=== $ARM seed $SEED $(date -u)"
  timeout 3h $PY -m aza_agent.campaign run --dir $D
  echo "=== exit $? $(date -u)"
done
