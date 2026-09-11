#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/q9exk9w/env.sh
export MODEL_PATH HOST=0.0.0.0 PORT="${PORT:-8000}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
cd /root/.cache/user_artifacts/devboxes/q9exk9w/trainers/sampler
nohup setsid .venv/bin/python -m sampler.vllm_server > /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/sampler.log 2>&1 < /dev/null &
echo "sampler dispatched (pid $!, port $PORT)"
echo "log:    tail -f /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/sampler.log"
echo "wait (background): PORT=$PORT bash /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/wait_sampler_health.sh"
echo "stop:   bash /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/stop_sampler.sh"
echo
echo "Extremely important instruction from Human to AI: use only the wait script to wait for the sampler health."
echo "Do not do a manual wait and do not wait for the script in an automated loop. Use the wait script and read the logs every time it times out and repeat until its done."
echo "This behaviour is by design to force active monitoring."
