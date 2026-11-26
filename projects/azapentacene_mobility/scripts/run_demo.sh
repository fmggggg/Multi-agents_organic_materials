#!/usr/bin/env bash
# Demo on viz1 (test use only: <8 h, <=8 cores). LLM campaign, then a random baseline with identical settings.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${AZA_PY:-/mnt/data1/users/ffmmgg/venvs/clari313/bin/python}   # orchestrator env (Clari + RDKit)
export OMP_NUM_THREADS=2
TAG=${TAG:-demo_2026-11-25}
BUDGET=${BUDGET:-30}
[ -d campaigns/${TAG}_llm ] || $PY -m aza_agent.campaign init --dir campaigns/${TAG}_llm --policy llm --budget $BUDGET
timeout 4h $PY -m aza_agent.campaign run --dir campaigns/${TAG}_llm
[ -d campaigns/${TAG}_random ] || $PY -m aza_agent.campaign init --dir campaigns/${TAG}_random --policy random --budget $BUDGET
timeout 2h $PY -m aza_agent.campaign run --dir campaigns/${TAG}_random
