#!/usr/bin/env bash
set -euo pipefail
B=/root/.cache/user_artifacts/devboxes/wgpn4vw
STATUS="$B/glm53flash-welsh-sharedouter-50.status"
source "$B/env.sh"
source "$B/.wandb.env"
export HF_MODEL_ID=zai-org/GLM-5.3-Flash
export RENDERER=glm5_3_max_reasoning
test "$(git -C "$B/trainers" rev-parse HEAD)" = 9c395a676bf20866abd79108996667c90ac63d06
test -n "$WANDB_API_KEY"
python3 - "$B/glm53flash-welsh-sharedouter-50-config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)
assert config["moe_lora_config"] == "shared_outer"
assert config["no_qbproj"] is False
assert config["canonical_gu"] is False
PY
cd "$B/sft-baselines-loops"
test -f dataset/train.jsonl
test ! -e validation_50step/reproduced-sharedouter-GLM-5.3-Flash-baseten
started=$(date +%s)
printf 'started_epoch=%s\nstarted_utc=%s\n' "$started" "$(date -u +%FT%TZ)" > "$STATUS"
if uv run --project runtime-baseten-glm --frozen --python 3.12.10 python reproduce_tinker_flash.py \
    --dataset dataset/train.jsonl \
    --model zai-org/GLM-5.3-Flash \
    --renderer glm5_3_max_reasoning \
    --local-url http://127.0.0.1:8001 \
    --log-path validation_50step/reproduced-sharedouter-GLM-5.3-Flash-baseten \
    --run-name GLM-5.3-Flash-welsh-validation50-sharedouter-baseten \
    --steps 50 \
    --save-checkpoint \
    --checkpoint-session-id glm53flash-welsh-sharedouter-50 \
    --wandb-project lora-target-baselines; then
  result=0
else
  result=$?
fi
finished=$(date +%s)
printf 'finished_epoch=%s\nfinished_utc=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
  "$finished" "$(date -u +%FT%TZ)" "$((finished - started))" "$result" >> "$STATUS"
exit "$result"
