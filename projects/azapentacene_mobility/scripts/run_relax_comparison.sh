#!/bin/bash
# 2026-11-26: same 30-molecule comparison with quick relax (100 steps, f32, cell), 50 draws per Z, GPU 1.
# Order: GPT agent, random (all skeletons), random restricted to LLL (fairness baseline). Resumable.
set -u
cd "$(dirname "$0")/.."
PY=${AZA_PY:-/mnt/data1/users/ffmmgg/venvs/clari313/bin/python}   # orchestrator env (Clari + RDKit)
export OMP_NUM_THREADS=2
TAG=${TAG:-relax_2026-11-26}
COMMON="--budget 30 --samples-per-z 50 --relax-steps 100 --gpu ${GPU:-1}"
[ -d campaigns/${TAG}_gpt ] || $PY -m aza_agent.campaign init --dir campaigns/${TAG}_gpt --policy llm $COMMON
[ -d campaigns/${TAG}_random ] || $PY -m aza_agent.campaign init --dir campaigns/${TAG}_random --policy random $COMMON
[ -d campaigns/${TAG}_randomLLL ] || $PY -m aza_agent.campaign init --dir campaigns/${TAG}_randomLLL --policy random --random-skeletons LLL $COMMON
for arm in gpt random randomLLL; do
  echo "=== $arm $(date)"
  $PY -m aza_agent.campaign run --dir campaigns/${TAG}_$arm
done
echo "=== all done $(date)"
