#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/users/ffmmgg/scratch/OCAgent"
SESSION_NAME="${SESSION_NAME:-ocagent-hof}"
CAMPAIGN_PATH="${CAMPAIGN_PATH:-${ROOT_DIR}/campaigns/porous_hof.json}"
POLL_SECONDS="${POLL_SECONDS:-300}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1000}"
MOLECULE_BATCH_SIZE="${MOLECULE_BATCH_SIZE:-5}"
LITERATURE_HYPOTHESES="${LITERATURE_HYPOTHESES:-5}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${SESSION_NAME}.log}"
PID_PATH="${LOG_DIR}/${SESSION_NAME}.pid"

mkdir -p "${LOG_DIR}" "$(dirname "${CAMPAIGN_PATH}")"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not available on this server." >&2
  exit 1
fi

if [ ! -f "${CAMPAIGN_PATH}" ]; then
  echo "Campaign file does not exist: ${CAMPAIGN_PATH}" >&2
  echo "Initialize one first with: bash ${ROOT_DIR}/scripts/init_porous_hof_campaign.sh ${CAMPAIGN_PATH}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  echo "Attach with: bash ${ROOT_DIR}/scripts/attach_campaign_tmux.sh ${SESSION_NAME}" >&2
  exit 1
fi

TMUX_CMD=$(cat <<EOF
cd "${ROOT_DIR}"
module load rdkit/2025_03_5-gcc14.2.0
source "${ROOT_DIR}/.venv-rdkit/bin/activate"
echo \$\$ > "${PID_PATH}"
echo "[campaign] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
python "${ROOT_DIR}/scripts/run_campaign_daemon.py" \\
  --campaign-path "${CAMPAIGN_PATH}" \\
  --poll-seconds "${POLL_SECONDS}" \\
  --sample-count "${SAMPLE_COUNT}" \\
  --molecule-batch-size "${MOLECULE_BATCH_SIZE}" \\
  --literature-hypotheses "${LITERATURE_HYPOTHESES}" \\
  >> "${LOG_PATH}" 2>&1
STATUS=\$?
echo "[campaign] exited with status \${STATUS} at \$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_PATH}"
exec bash
EOF
)

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '$TMUX_CMD'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Campaign: ${CAMPAIGN_PATH}"
echo "Log: ${LOG_PATH}"
echo "Attach: bash ${ROOT_DIR}/scripts/attach_campaign_tmux.sh ${SESSION_NAME}"
