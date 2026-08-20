#!/usr/bin/env bash
# rung3_arm.sh — measure one offload arm (conway, 2026-08-20).
#
#   bash rung3_arm.sh 3b
#
# Order matters. The three boot checks come FIRST and are pass/fail: if the
# offload did not actually engage, every timing number below is a measurement
# of the baseline wearing an arm's label, which is the worst failure mode this
# ladder can produce. Then the forward-only parity leg on fresh weights, then
# the timed run.
#
# NOTHING heavy runs on the box during the timed windows. The rung-1 pair
# learned this the expensive way: a census analysis on the leader node cost
# run B 2.4% throughput and quadrupled its spread.
set -euo pipefail

ARM=${1:?usage: rung3_arm.sh <arm-label, e.g. 3b>}
ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
BENCH=$ART/lps1062_bench
PY=$ART/trainers_main/server-megatron-bridge/.venv/bin/python
L=$PP2/logs/trainer_srun.log
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

banner() { printf '\n=== %s ===\n' "$*"; }

JID=$(squeue -h --name=devbox_trainer -o "%A" | head -1)
[ -n "$JID" ] || { echo "!! no devbox_trainer job"; exit 1; }
SR="srun --jobid=$JID --overlap"

banner "0. topology"
curl -fs -m10 http://127.0.0.1:8001/status | python3 -m json.tool | head -20

banner "1. BOOT CHECK 1/3 — the NVTE import-time latch"
# env=1 latch=True is the only passing state with offload enabled. latch=False
# means TE latched the variable at import before the environment set it, so TE
# is silently on its pre-V1 path while everything else reads as engaged.
grep -a "activation-offload NVTE latch:" "$L" | tail -1 || echo "!! NO LATCH LINE — needs trainers >= 7eec3054"

banner "2. BOOT CHECK 2/3 — hook engagement at the first MoE layer"
grep -a "activation-offload engagement:" "$L" | tail -1 || echo "!! NO ENGAGEMENT LINE"
grep -a "activation-offload engagement probe failed" "$L" | tail -1 || true

banner "3. BOOT CHECK 3/3 — NUMA-local pinned buffers"
# Unbound placement is the process default and measured 15.6/16.8 GB/s per GPU
# against a ~22/24 requirement: it runs, ~30% under what the design needs,
# while looking healthy.
grep -a "BT_OFFLOAD_NUMA_BIND:" "$L" | head -4 || echo "(no NUMA bind lines yet — they appear at first offload)"
grep -a "BT_OFFLOAD_NUMA_VERIFY:" "$L" | head -4 || echo "(no NUMA verify lines yet)"

banner "4. parity leg, forward-only, fresh weights"
$PY "$PP2/parity_driver.py" --label "rung${ARM}-qed7z1w-p1-$STAMP"

banner "5. judged against the rung-1 within-boot floor (ratios, no absolute bar)"
$PY "$PP2/parity_floor_stats.py" \
  "$BENCH/parity_rung1A-qed7z1w-p1-20260820T220915Z.json" \
  "$BENCH/parity_rung${ARM}-qed7z1w-p1-$STAMP.json" \
  --label "arm $ARM vs rung-1 baseline leg" \
  --floor "$PP2/floor_withinboot.json" || true

banner "6. pollers"
mkdir -p "$PP2/mem_$ARM"
$SR -N2 -n2 bash "$PP2/poll_gpu_mem.sh" "$PP2/mem_$ARM" &
POLLER=$!

banner "7. the timed run — d2, 12 steps, same shape as the rung-1 baseline"
$PY "$PP2/profile_driver_new.py" --label "rung${ARM}-qed7z1w-d2-$STAMP" \
  --datums 2 --control-repeats 10

banner "8. preserve artifacts"
$SR -N2 -n2 --label bash -c \
  "mkdir -p $PP2/artifacts/$ARM/\$(hostname -s); \
   cp -a /tmp/checkpoints/profiles/latest/memory/. $PP2/artifacts/$ARM/\$(hostname -s)/ 2>/dev/null || true; \
   cp -a /tmp/checkpoints/profiles/torch_trace/. $PP2/artifacts/$ARM/\$(hostname -s)/ 2>/dev/null || true; \
   ls $PP2/artifacts/$ARM/\$(hostname -s) | wc -l"

banner "9. stop pollers, read the plateau"
kill "$POLLER" 2>/dev/null || true
pkill -f "[p]oll_gpu_mem.sh" 2>/dev/null || true
python3 "$PP2/mem_plateau.py" "$PP2/mem_$ARM"/mem.*.csv --window 26 --tail 2

banner "10. valve telemetry"
# Needs BT_OFFLOAD_VALVE_TELEMETRY=1 and _EVERY=1 in the boot env; both
# default to silent, and _EVERY defaults to 100 training iterations against a
# 12-step run.
grep -a "valve" "$L" | tail -8 || echo "(no valve telemetry — check the boot env)"

banner "11. offload copy traffic in the boot log"
grep -aiE "offload.*(bytes|GiB|pool)" "$L" | tail -5 || true

echo
echo "STAMP=$STAMP"
