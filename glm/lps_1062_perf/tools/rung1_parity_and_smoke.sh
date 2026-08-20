#!/usr/bin/env bash
# rung1_parity_and_smoke.sh — everything that must happen on FRESH weights,
# before anything steps the optimizer (conway, 2026-08-20).
#
# Order is load-bearing (the fresh-weight rule): the parity legs are
# forward-only and LoRA B is zero-init, so on a fresh boot the forward equals
# the base model exactly. That is what makes a later cross-boot parity cell
# comparable. The smoke pass DOES step the optimizer, so it comes after.
set -euo pipefail

ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
BENCH=$ART/lps1062_bench
PY=$ART/trainers_main/server-megatron-bridge/.venv/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LABEL_BASE=${LABEL_BASE:-rung1A-qed7z1w}

banner() { printf '\n=== %s ===\n' "$*"; }

banner "0. server identity"
curl -fs -m10 http://127.0.0.1:8001/status | python3 -m json.tool | head -40

banner "0b. boot-log cross-checks"
L=$PP2/logs/trainer_srun.log
echo "-- resolved layer geometry (cross-check of the pre-boot config dump):"
grep -aoE "indexer_types[^,]{0,80}" "$L" | head -1 || echo "(not printed in the boot log)"
grep -aoE "mlp_layer_types[^,]{0,80}" "$L" | head -1 || echo "(not printed in the boot log)"
echo "-- offload surfaces (expected INERT on this baseline boot):"
grep -a "activation-offload NVTE latch:" "$L" | tail -1 || echo "(no latch line)"
grep -a "activation-offload engagement:" "$L" | tail -1 || echo "(no engagement line)"

banner "1. parity leg 1 of 2 (fresh weights)"
$PY "$PP2/parity_driver.py" --label "${LABEL_BASE}-p1-$STAMP"

banner "2. parity leg 2 of 2 (same boot, same weights — the within-boot floor)"
$PY "$PP2/parity_driver.py" --label "${LABEL_BASE}-p2-$STAMP"

banner "3. compare the pair — THIS IS THE NOISE FLOOR, not a pass/fail"
# The driver's built-in 1e-6/1e-3 tolerances are the OLD absolute bars. We do
# NOT judge against them; we record the three statistics as the floor that
# later arms are judged relative to.
# `|| true` is load-bearing: the driver exits non-zero when its OLD absolute
# tolerances trip, and on this intrinsically-nondeterministic path they always
# will. This step MEASURES the floor; it does not gate anything, and its exit
# code must not abort the sequence (it did, and cost the smoke pass).
$PY "$PP2/parity_driver.py" --compare \
  "$BENCH/parity_${LABEL_BASE}-p1-$STAMP.json" \
  "$BENCH/parity_${LABEL_BASE}-p2-$STAMP.json" || true

banner "4. smoke: one d1 driver pass (steps the optimizer — after the parity legs)"
$PY "$PP2/profile_driver_new.py" --label "rung1smoke-qed7z1w-d1-$STAMP" \
  --datums 1 --control-repeats 1

banner "5. smoke pass bars — allocator snapshots and traces on BOTH nodes"
srun --overlap -N2 -n2 --label bash -c \
  'ls /tmp/checkpoints/profiles/latest/memory/ 2>/dev/null | wc -l; \
   ls /tmp/checkpoints/profiles/torch_trace/ 2>/dev/null | wc -l'
echo "-- expect: 8 pickles per node (16 total), and a trace on node 0 (rank 0) and node 1 (rank 8)"
$PY - <<'EOF'
import pickle
p = "/tmp/checkpoints/profiles/latest/memory/memory.rank0.pickle"
s = pickle.load(open(p, "rb"))
ev = sum(len(t) for t in s["device_traces"])
print(f"rank0 pickle: {len(s['segments'])} segments, {ev} device-trace events")
EOF

echo
echo "STAMP=$STAMP"
