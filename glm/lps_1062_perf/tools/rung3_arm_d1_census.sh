#!/usr/bin/env bash
# rung3_arm_d1_census.sh — a SURVIVING 3b run, for composition not throughput
# (turing, 2026-08-20 night).
#
#   bash rung3_arm_d1_census.sh 3bv4d1
#
# Why this script exists. Arm 3b does not fit at 131k/d2: two attempts OOMed at
# 258.4 GiB torch-allocated on a 267.7 GiB card, identically with the valve
# uncapped and capped at 4, so the overshoot is RESIDENT memory and not
# offload copies in flight. The open question is therefore "resident where?",
# and that is a composition question, which needs a run that reaches a
# plateau and dumps allocator snapshots.
#
# Two deliberate differences from rung3_arm.sh:
#
#   1. d1, not d2. At PP2 the first stage holds one microbatch's activation
#      sets instead of two, which is the term that pushed d2 over the ceiling.
#      The throughput number from this run is therefore NOT comparable to the
#      645.1 tok/s/GPU d2 baseline of record — different shape, and d1's
#      pipeline bubble is far larger. Do not quote it as the arm's result.
#
#   2. No parity leg. parity_driver.py's fixed datum set is 262,032 real
#      tokens (9 datums -> 3 partitions of 131k), i.e. the same volume as a d2
#      step, and it is where both d2 attempts died. Running it here would just
#      reproduce the OOM and lose the census. 3b parity stays open and is
#      called out as such in the report.
#
# The census analysis runs LAST, after every timed window has closed. Rung 1
# learned that the expensive way: allocator-snapshot analysis on the leader
# node while a run was stepping cost that run 2.4% throughput and quadrupled
# its spread.
set -euo pipefail

ARM=${1:?usage: rung3_arm_d1_census.sh <label, e.g. 3bv4d1>}
ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
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
grep -a "activation-offload NVTE latch:" "$L" | tail -1 || echo "!! NO LATCH LINE"

banner "2. BOOT CHECK 2/3 — hook engagement at the first MoE layer"
grep -a "activation-offload engagement:" "$L" | tail -1 || echo "!! NO ENGAGEMENT LINE"

banner "3. BOOT CHECK 3/3 — NUMA-local pinned buffers"
# sed -n, not head: `| head -N` kills grep with SIGPIPE, so a `|| echo`
# fallback fires even when matches were printed.
BIND_LINES=$(grep -a "BT_OFFLOAD_NUMA_BIND:" "$L" | sed -n 1,4p)
[ -n "$BIND_LINES" ] && echo "$BIND_LINES" || echo "(no NUMA bind lines yet)"
UNRESOLVED=$(grep -ac "could not resolve the GPU" "$L" || true)
echo "NUMA resolution failures: $UNRESOLVED  (must be 0)"
VER_COUNT=$(grep -ac "BT_OFFLOAD_NUMA_VERIFY: pinned buffer" "$L" || true)
echo "pinned-pool verify lines so far: $VER_COUNT"

banner "4. pollers on both nodes"
mkdir -p "$PP2/mem_$ARM"
$SR -N2 -n2 bash "$PP2/poll_gpu_mem.sh" "$PP2/mem_$ARM" &
POLLER=$!

banner "5. the timed run — d1, 12 windows (warmup + traced + 10 controls)"
$PY "$PP2/profile_driver_new.py" --label "rung${ARM}-qed7z1w-d1-$STAMP" \
  --datums 1 --control-repeats 10

banner "6. preserve allocator snapshots + traces from BOTH nodes"
$SR -N2 -n2 --label bash -c \
  "mkdir -p $PP2/artifacts/$ARM/\$(hostname -s); \
   cp -a /tmp/checkpoints/profiles/latest/memory/. $PP2/artifacts/$ARM/\$(hostname -s)/ 2>/dev/null || true; \
   cp -a /tmp/checkpoints/profiles/torch_trace/. $PP2/artifacts/$ARM/\$(hostname -s)/ 2>/dev/null || true; \
   ls $PP2/artifacts/$ARM/\$(hostname -s) | wc -l"

banner "7. stop pollers, read the plateau"
kill "$POLLER" 2>/dev/null || true
pkill -f "[p]oll_gpu_mem.sh" 2>/dev/null || true
python3 "$PP2/mem_plateau.py" "$PP2/mem_$ARM"/mem.*.csv --window 26 --tail 2

banner "8. valve telemetry (counters are CUMULATIVE — _valve_stats is never cleared)"
grep -a "VALVE_TELEMETRY" "$L" | tail -4 || echo "(none — check the boot env)"

banner "9. offload traffic: distinct pinned destination pools"
# One line per new (shape,dtype) pool per rank, emitted from inside offload()
# just before the real D2H copy — so this is proof that bytes moved, which the
# three boot checks do not establish.
grep -ao "pinned buffer [0-9]* bytes" "$L" \
  | awk '{s+=$3; n++} END {printf "pools=%d  total=%.1f GiB  per-rank avg=%.1f GiB\n", n, s/1073741824, s/1073741824/16}'

banner "10. census — per-module resident attribution (AFTER all timed windows)"
for R in 0 8; do
  echo "--- rank $R ---"
  SNAP=$(ls $PP2/artifacts/$ARM/*/memory.rank${R}.pickle 2>/dev/null | head -1)
  if [ -z "$SNAP" ]; then echo "(no snapshot for rank $R)"; continue; fi
  $PY "$PP2/memory_census.py" "$SNAP" --rank "$R" 2>&1 | tail -40
done

echo
echo "STAMP=$STAMP"
