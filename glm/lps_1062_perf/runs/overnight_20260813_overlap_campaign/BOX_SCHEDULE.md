# Box window schedule — q8eg0gq (fermi, 2026-08-13 19:5x CDT)

Pre-planned window sequence for when q8eg0gq lands (currently Pending — ali
full, one foreign 2-node job ahead of us). doppler drives all trainer
lifecycle; window owners hand doppler exact configs. `wait_trainer_health.sh`
only; `BT_SAVE_STATE_SYNC=1` everywhere; fixed wheel is a precondition for
every window (W0).

## CANONICAL MISSION ENV (bayes, 2026-08-14 — definitive; cite from every W-spec)

The record anchors (d4 878-886, d16 984) were measured under this FULL env.
Any window that intends comparability with the record runs this block
verbatim; per-window deltas (BT_SKIP_WARMUP, B/F gates, profiling) are
declared explicitly in the window's spec and logged.

```bash
export BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/trainer_config.json
export BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/trainer_server_config.json
export BT_SAVE_STATE_SYNC=1           # F1 landmine: async save hangs under CP>1
# ⚠ 08-14 P4/S3 finding (turing): this var is read ONLY on trees carrying
# db5d1826's megatron_config.py hunk — on 73c24b00-class trees it is a
# SILENT NO-OP (async_save hardcoded True at megatron_config.py:325) and
# the CP>1 save wedge is LIVE. Apply the hunk (sha-verify) on any tree
# that will call /save_state; merge-queue item 6 is the durable fix.
export BT_TF32_LM_HEAD=1              # TF32 CE head patch is ENV-GATED (default 0!)
export NCCL_IB_QPS_PER_CONNECTION=8   # ship NCCL env (3 knobs)
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_NCHANNELS_PER_NET_PEER=8
# traced windows only: BT_PROFILE_RANKS=0,8
```

Origin of this entry: W1a (2026-08-14 early) ran with ONLY the config paths +
BT_SAVE_STATE_SYNC=1 (P0 order under-specified the env — protocol-doc gap, not
driver error), producing anchors ~3.5-4% below record. The TF32-head gate
(chunked_lm_head.py:65, default "0") and the ship NCCL knobs were both absent.
W1c boot 1 runs the full block and doubles as the env-vs-venv cross-check.

| # | window | owner | contents | est | gate to next |
|---|---|---|---|---|---|
| W0 | wheel + regression | doppler | venv bump to cudnn-frontend 1.27.0 (PyPI --no-deps); import+version; both DSA test files (5 passed) | ~30m | tests GREEN |
| W1a | canary + anchors | doppler | d2 canary (12.2–12.4, gn 0.36–0.49) → d4 (bar: 878–886 band) → d16 (bar: 984 ±3%) | ~2–3h | d16 in-band; if out-of-band STOP → fermi |
| W1b | block+K21 A/B | jacobi spec (doppler drives) | ZERO-CODE leg per MEMORY_LEG_DECISION.md: config-only swap to configs/trainer_pp2cp8ep8_131k_blockK21.json (same tree as anchors = cleanest A/B); d4 pair then d16 pair vs W1a anchors; bars pre-registered in the memo (est +6–10% @d16; predicted peak 221 GiB d4 / 240 d16 — also retires the dial's S_eager risk) | ~2–3h | verdict recorded; peak-mem measured |
| W1c | B/F A/B | bohr (doppler drives) | 3-boot shared-off-arm design per REBUILD_NOTE.md (tree swap to bf-rebuild chain, MISSION config): d4 primary (adjudicates smaller-d bet), d16 pair w/ 3 on-record bands (gauss +3–6%, kepler +4–7%, refute <+1.5%); NCCL Arm-1 ablation folds in; kepler owns mechanism check (BT_PROFILE_RANKS=0,8: nonzero count + pure-idle shrink + fixed-wheel rank8 capture). NOTE: B/F × block+K sub-additive (block+K deletes replay-side layout-builder calls) — arms stay single-variable vs mission config; stacking measured at P4 | ~3–4h | pre-registered verdicts recorded either way |
| W2 | shim canary | lebesgue (doppler drives) | REQUIRES frozen-embedding landmine fix (mcore b907b6153, cherry-picked onto shim chain — first flag-ON boot is also that fix's first hardware validation). Per OVERLAP_AB_DESIGN.md: 32k NOT 131k (VPP2 = 2 chunks in flight, 131k OOMs); config trainer_pp2cp8ep8_32k_selective_vpp2_overlap.json; control = plain PP2 flag-OFF (flag-OFF VPP2 unrunnable, filed bug); ladder: control → d2 flag-ON canary → parity leg (≤1e-6 rel / ≤1e-3 per-token) → memory ramp 32k→64k→96k; BT_SKIP_WARMUP=1 hatch | ~2h | canary + parity PASS |
| W3 | W1 V0–V2 | hausdorff spec (doppler drives) | V0 boot (16/16 armed + ONE FULL optimizer step; hang → HOLD, no retry), V1 mechanism, V2 d2 canary | ~1h | V0–V2 PASS |
| W4 | W1 V3 + option-6 | hausdorff spec | V3 d4 A/B vs anchor under mean-vs-spread; option-6 canary only after, vs W1-armed baseline, never co-armed first boot | ~2–4h | — |
| W5+ | memory leg / overlap A/B | jacobi+lebesgue | per MEMORY_LEG_DECISION.md (pending); first real overlap A/B at 131k | TBD | — |

Ordering rationale: W1a re-anchors everything (no measurement counts without
it). B/F before shim canary because it's boot-cheap, its EV just rose
(kepler's 6.97s pure-idle leak from the nonzero class), and its Arm-1 NCCL
fold-in amortizes boots. W1/option-6 stay P3 per orders. Windows may
interleave opportunistically if a lane's Mac-side gate kills an arm (bohr's
NCCL Arm-2 gate query pending).

Revise on data; fermi owns this file.
