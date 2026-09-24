#!/usr/bin/env bash
set -euo pipefail
B=/root/.cache/user_artifacts/devboxes/wgpn4vw
STATUS="$B/glm53-welsh-canonicalgu-sharedouter-50.status"
source "$B/env.sh"
source "$B/.wandb.env"
export HF_MODEL_ID=zai-org/GLM-5.3
export RENDERER=glm5_3_max_reasoning
export WANDB_TAGS=glm53,canonicalgu,sharedouter,baseten,validation50
test "$(git -C "$B/trainers" rev-parse HEAD)" = b3863e2f0f17c724565049978c9a9329a2037c57
test -n "$WANDB_API_KEY"
cd "$B/sft-baselines-loops"
started=$(date +%s)
printf 'started_epoch=%s\nstarted_utc=%s\n' "$started" "$(date -u +%FT%TZ)" > "$STATUS"
if bash run_validation_50.sh \
    --backend baseten \
    --trainer-url http://127.0.0.1:8001 \
    --trainer-config "$B/welsh50-canonicalgu-sharedouter-trainer-config.json" \
    --sharedouter --canonicalgu; then
  result=0
else
  result=$?
fi
finished=$(date +%s)
printf 'finished_epoch=%s\nfinished_utc=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
  "$finished" "$(date -u +%FT%TZ)" "$((finished - started))" "$result" >> "$STATUS"
exit "$result"
