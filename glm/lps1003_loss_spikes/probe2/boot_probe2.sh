#!/usr/bin/env bash
# LPS-1003 probe2 boot wrapper — run ON THE LEADER of the devbox.
# usage: bash boot_probe2.sh <audit|stage|stagefix>
#   audit     E0/E1 only: branch+shape audit, out-of-window + duplicate scan
#   stage     + E2: staged sentinel block poisons the wheel's output_indices
#             allocation -> unwritten slots become countable
#   stagefix  + E4: same, but sentinel slots are rewritten to -1 after the call
#             (behaviourally the proposed fill_(-1) glue hardening)
# Dispatches through the standard .devbox_up start_trainer.sh; only exports env.
set -euo pipefail

MODE="${1:?usage: boot_probe2.sh <audit|stage|stagefix>}"
source /root/.cache/user_artifacts/env.sh
LPS=/root/.cache/user_artifacts/lps1003
P2=$LPS/probe2

export BT_DSA_AUDIT=1
case "$MODE" in
  audit)    ;;
  stage)    export BT_DSA_STAGE=1 ;;
  stagefix) export BT_DSA_STAGE=1 BT_DSA_FIX=1 ;;
  *) echo "bad mode '$MODE' (audit|stage|stagefix)" >&2; exit 2 ;;
esac

export BT_TRAINER_CONFIG_PATH=$LPS/trainer-config.flash.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$P2/harness${PYTHONPATH:+:$PYTHONPATH}"

STAMP=$(date +%m%d_%H%M%S)
export BT_DSA_AUDIT_DIR=$P2/runs/$MODE/audit_$STAMP
mkdir -p "$BT_DSA_AUDIT_DIR"
printf '%s\n' "$BT_DSA_AUDIT_DIR" > "$P2/runs/$MODE/latest_audit_dir"

echo "mode=$MODE stage=${BT_DSA_STAGE:-0} fix=${BT_DSA_FIX:-0} audit_dir=$BT_DSA_AUDIT_DIR"
bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh
