#!/usr/bin/env python3
"""LPS-1062 dispatcher host-sync patch — mechanical A/B acceptance checker.

Takes a kineto .pt.trace.json (one-step profiled capture), measures the
REVIEW_FIXA.md §2c counters 1-7, and prints PASS/FAIL per row against an
expectation profile.

Usage:
    python check_acceptance.py TRACE.pt.trace.json --profile baseline-exp05d
    python check_acceptance.py TRACE.pt.trace.json --profile post-patch
    python check_acceptance.py TRACE --profile post-patch --set memcpy_cpu_s.max=15.5
    python check_acceptance.py TRACE --profile post-patch --overrides my_expectations.json

Profiles:
    baseline-exp05d : reproduces the exp05d baseline trace column exactly
                      (counts exact, times +/-2%). Validating the checker =
                      running this profile on exp05d.pt.trace.json.
    baseline-4mb131k: DERIVED unpatched expectations for the 4 x 131,072 bench
                      shape (pure mb-scaling of exp05d; no measured baseline at
                      this shape). Used to validate the scaling derivations
                      against gibbs's unpatched 4mb capture.
    post-patch-A    : FIX A only (DSA bwd async nonempty flag). The 156 drains
                      go; layout nonzeros, tolist memcpys, kernel count stay.
    post-patch-B    : FIX B only (per-mb CP layout cache). Layout nonzeros and
                      the launch storm go; the 156 bwd drains STAY (19.2s).
    post-patch-AB   : FIX A + FIX B, no FIX F (tolist memcpys stay).
    post-patch-ABF  : all gates on (= post-patch), 2-mb shape (exp05d-calibrated).
    post-patch      : alias for post-patch-ABF (REVIEW_FIXA §2c accept column).
    post-patch-ABF-4mb131k : all gates on, BENCH shape 4 x 131,072 (4 mb/step,
                       524,288 tok/step, max_seq_len=131072 boot). Every
                       mb-scaling row re-derived from the 2-mb baseline — see
                       the derivation block above the profile.
    post-patch-BFC-4mb131k : SHIP config (B+F, A parked) + FIX C
                       (BT_MOE_DISPATCH_REPLAY_CACHE). Replay dispatcher
                       eventSyncs 300->~0, replay preprocess all-gathers
                       600->300 (GPU-side Long allgather row), fwd halves
                       untouched.
    post-patch-W1-4mb131k  : B+F + W1 (BT_MOE_PROBS_A2A_COMM). Probs SendRecv
                       (Float) all off the token stream; dispatch->combine gap
                       avg <= 7 ms (was 10.85); token a2a flat +/-5%; wall
                       <= 43.18 s.
    post-patch-W2-4mb131k  : B+F + W2 v1 (BT_MOE_A2A_PIPELINE=2). Token
                       SendRecv ~3600 at ~half payload; a2a x compute overlap
                       >= 15% of residency (baseline 1.5%); wall <= 44.68 s.
    If FIX F is not in the patch under test, use post-patch-AB, or relax the
    two memcpy rows: --set memcpy_cpu_s.max=15.5 --set memcpy_gt1ms.max=600

Gate semantics (from ATTRIBUTION.md + 2026-08-09 adjudication): FIX A removes
the 156 FSA-bwd nonzero drains (19.2s CPU) and adds 156 us-scale event syncs +
156 tiny pinned DtoH. FIX B removes 26,520 layout-builder nonzeros (~1.34s
CPU), ~29k pinned-DtoH nonzero internals, and ~150k kernel launches. FIX F
(BT_THD_ROPE_HOST_CACHE) removes the ~587 blocking pageable-DtoH copies
(14.7s CPU) from THD RoPE cu_seqlens host reads (_apply_rotary_pos_emb_thd,
rope_utils.py — site corrected from sort_chunks_by_idxs; the op-level counters
are unchanged). Post-F memcpy_gt1ms floors at ~2/step (one blocking D2H per
unique cu_seqlens tensor = per-microbatch cache miss), hence bound <= 8.
The 4 step-level _index_put_impl_ nonzero drains (>5ms) are out of scope for
all gates — nonzero_gt5ms floors at 4, not 0.

Overrides: --set <metric>.<bound>=<value> (bound = min|max|tol) or a JSON file
{"memcpy_cpu_s": {"max": 1.0}, ...}. Metrics are listed in the output table.

Counter 5 of §2c (per-step fallback counter) is log-based, not trace-based —
the checker prints a REMINDER row for it; gibbs fills it from trainer logs.

Requires: pip install perfetto pandas numpy   (python>=3.10)

NOTE on versions: the baseline column was validated with the perfetto python
API (trace_processor_shell v56.1). trace_processor >=v57 KEEPS ~5,405
overlapping complete events that v56.1 drops from the slice table, which shifts
kernel_count by that amount (361,032 vs 355,627 on exp05d). Baseline tolerance
on kernel_count is ±2% to absorb this; the post-patch bound (<=230,000) is far
below either value, so the A/B verdict is version-independent. Where two
numbers exist, the checker's own measurement on the baseline trace is
authoritative (REVIEW_FIXA §2c row 6a listed the v57.2 figure 361,032).

NOTE on checker provenance (2026-08-10, boltzmann): on-box copies of THIS
checker drift stale (the box copy was found two revisions behind on 2026-08-10
— missing the 4mb131k profiles and the C′ re-baseline block). Any on-box
verdict must first md5-proof the box copy against the Mac authority copy
(current Mac md5 recorded in the verification ledger); a checker without the
profile you think you're running is a silent-adjudication trap (bug class 4
applied to tooling).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    from perfetto.trace_processor import TraceProcessor
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "ERROR: perfetto python package not found.\n"
        "  pip install perfetto pandas numpy\n"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure(trace_path: str) -> dict:
    tp = TraceProcessor(trace=trace_path)

    def q(sql):
        return tp.query(sql).as_pandas_dataframe()

    m = {}

    # --- counters 1/2/4: cpu-side sync ops ---------------------------------
    nz = q("""
        SELECT s.dur, p.name AS parent
        FROM slice s LEFT JOIN slice p ON s.parent_id = p.id
        WHERE s.name = 'aten::nonzero'
    """)
    m["nonzero_calls"] = int(len(nz))
    m["nonzero_cpu_s"] = float(nz.dur.sum() / 1e9)
    m["nonzero_gt5ms"] = int((nz.dur > 5e6).sum())
    # 1b: layout-builder boolean-index nonzeros (parent aten::index), fwd + replay
    m["nonzero_layout"] = int((nz.parent == "aten::index").sum())

    ss = q("SELECT dur FROM slice WHERE name = 'cudaStreamSynchronize'")
    m["streamsync_calls"] = int(len(ss))
    m["streamsync_cpu_s"] = float(ss.dur.sum() / 1e9)

    mc = q("SELECT dur FROM slice WHERE name = 'cudaMemcpyAsync'")
    m["memcpy_calls"] = int(len(mc))
    m["memcpy_cpu_s"] = float(mc.dur.sum() / 1e9)
    m["memcpy_gt1ms"] = int((mc.dur > 1e6).sum())

    es = q("SELECT dur FROM slice WHERE name = 'cudaEventSynchronize'")
    m["eventsync_calls"] = int(len(es))
    m["eventsync_cpu_s"] = float(es.dur.sum() / 1e9)
    m["eventsync_gt1ms"] = int((es.dur > 1e6).sum())
    # Split by parent: FIX A's flag events are direct children of
    # FusedSparseAttentionFuncBackward; the dispatcher's d2h_event syncs sit under
    # CheckpointFunction{,Backward}. (Validated against gated-v2-4mb-steady.)
    es2 = q("""
        SELECT s.dur, p.name AS parent
        FROM slice s JOIN slice p ON s.parent_id = p.id
        WHERE s.name = 'cudaEventSynchronize'
    """)
    a = es2[es2.parent == "FusedSparseAttentionFuncBackward"]
    disp = es2[es2.parent != "FusedSparseAttentionFuncBackward"]
    m["eventsync_a_calls"] = int(len(a))
    m["eventsync_a_cpu_s"] = float(a.dur.sum() / 1e9)
    m["eventsync_a_gt1ms"] = int((a.dur > 1e6).sum())
    m["eventsync_dispatcher_cpu_s"] = float(disp.dur.sum() / 1e9)
    # FIX C split: the dispatcher's d2h_event syncs divide into first-pass
    # (parent CheckpointFunction) and recompute-replay (parent
    # CheckpointFunctionBackward). FIX C removes the replay half.
    # W3 (2026-08-10, boltzmann): under BT_MOE_LOOKAHEAD_RECOMPUTE the
    # checkpoint Function is LookaheadCheckpointFunction{,Backward} — include
    # both names (pre-W3 traces simply match nothing on the Lookahead names).
    disp_fwd = es2[es2.parent.isin(["CheckpointFunction", "LookaheadCheckpointFunction"])]
    disp_replay = es2[
        es2.parent.isin(["CheckpointFunctionBackward", "LookaheadCheckpointFunctionBackward"])
    ]
    m["eventsync_dispatcher_fwd_calls"] = int(len(disp_fwd))
    m["eventsync_dispatcher_replay_calls"] = int(len(disp_replay))

    # FIX C: the dispatcher's preprocess all-gather, counted GPU-side where it
    # is cleanly separable from the DSA CP gathers: ncclDevKernel_AllGather
    # with args.dtype == 'Long' on the EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
    # (256-elem input; validated 2026-08-09 on gated-v2-4mb-steady: exactly
    # 600 = 75 MoE x 4 mb x 2 passes; the CP gathers are BFloat16 on
    # CONTEXT_PARALLEL_GROUP and never match). FIX C removes the 300 replay
    # ones: 600 -> 300. (CPU-side, the call runs under
    # _GatherFromSequenceParallelRegion, but that Function is shared with the
    # DSA CP gather — 696/696 fwd/replay there — so the GPU-side dtype filter
    # is the authoritative row.)
    ag = q("""
        SELECT COUNT(*) AS n
        FROM slice s
        JOIN args dt ON dt.arg_set_id = s.arg_set_id AND dt.key = 'args.dtype'
        JOIN args pg ON pg.arg_set_id = s.arg_set_id AND pg.key = 'args.Process Group Description'
        WHERE s.category='kernel' AND s.name LIKE '%AllGather%'
          AND dt.display_value = 'Long'
          AND pg.display_value = 'EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP'
    """)
    m["dispatcher_allgather_calls"] = int(ag.n.iloc[0])

    # --- W1/W2 (overlap_design/TRACE_ACCEPTANCE.md): EP a2a structure -------
    # SendRecv kernels carry args.dtype (BFloat16 = tokens, Float = probs),
    # args.stream, and args.In msg nelems. Baseline (gated-v2-4mb-steady,
    # validated 2026-08-09): 1800 BF16 / 20.90 s, 900 Float / 5.13 s, ALL on
    # one stream (83); dispatch->combine gap (consecutive-pair definition,
    # reproduces laplace's 10.85/16.64 exactly) avg 10.85 ms; a2a x compute
    # overlap 0.395 s = 1.5% of SendRecv residency.
    sr = q("""
        SELECT s.ts, s.dur, dt.display_value AS dtype, st.display_value AS stream,
               ine.display_value AS in_nelems
        FROM slice s
        JOIN args dt ON dt.arg_set_id = s.arg_set_id AND dt.key = 'args.dtype'
        JOIN args st ON st.arg_set_id = s.arg_set_id AND st.key = 'args.stream'
        JOIN args ine ON ine.arg_set_id = s.arg_set_id AND ine.key = 'args.In msg nelems'
        WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
        ORDER BY s.ts
    """)
    tok = sr[sr.dtype == "BFloat16"]
    probs = sr[sr.dtype == "Float"]
    m["token_a2a_calls"] = int(len(tok))
    m["token_a2a_gpu_s"] = float(tok.dur.sum() / 1e9)
    m["probs_a2a_calls"] = int(len(probs))
    m["probs_a2a_gpu_s"] = float(probs.dur.sum() / 1e9)
    if len(tok):
        token_stream = tok.stream.mode().iloc[0]
        m["probs_a2a_off_token_stream"] = int((probs.stream != token_stream).sum())
        m["token_a2a_in_nelems_p50"] = float(tok.in_nelems.astype(float).median())
        # dispatch->combine gap: consecutive (dispatch, combine) pairs of token
        # kernels on the comm stream (validated against the probs-anchored
        # definition on the baseline; robust when W1 moves probs off-stream).
        tts = tok.ts.to_numpy(dtype=np.int64)
        tdur = tok.dur.to_numpy(dtype=np.int64)
        tend = tts + tdur
        d_end = tend[0::2][: len(tts[1::2])]
        c_ts = tts[1::2]
        if len(d_end) and len(c_ts):
            gaps = (c_ts - d_end) / 1e6
            m["dispatch_combine_gap_avg_ms"] = float(gaps.mean())
            m["dispatch_combine_gap_p90_ms"] = float(np.percentile(gaps, 90))
        else:
            m["dispatch_combine_gap_avg_ms"] = 0.0
            m["dispatch_combine_gap_p90_ms"] = 0.0
    else:
        m["probs_a2a_off_token_stream"] = 0
        m["token_a2a_in_nelems_p50"] = 0.0
        m["dispatch_combine_gap_avg_ms"] = 0.0
        m["dispatch_combine_gap_p90_ms"] = 0.0
    # a2a x compute overlap: union of non-SendRecv kernel intervals intersected
    # with SendRecv kernel intervals, as a fraction of total SendRecv residency
    # (baseline 0.395 s / 1.5%).
    ck = q("""
        SELECT ts, dur FROM slice
        WHERE category='kernel' AND name NOT LIKE '%SendRecv%' AND dur > 0
    """)
    if len(sr) and len(ck):
        ca = ck.ts.to_numpy(dtype=np.int64)
        cb = ca + ck.dur.to_numpy(dtype=np.int64)
        order = np.argsort(ca, kind="stable")
        ca, cb = ca[order], cb[order]
        merged_s, merged_e = [], []
        cs, ce = int(ca[0]), int(cb[0])
        for s_, e_ in zip(ca[1:], cb[1:]):
            if s_ <= ce:
                if e_ > ce:
                    ce = int(e_)
            else:
                merged_s.append(cs)
                merged_e.append(ce)
                cs, ce = int(s_), int(e_)
        merged_s.append(cs)
        merged_e.append(ce)
        merged_s = np.array(merged_s)
        merged_e = np.array(merged_e)
        sa = sr.ts.to_numpy(dtype=np.int64)
        sb = sa + sr.dur.to_numpy(dtype=np.int64)
        # for each SendRecv kernel, walk merged compute intervals from bisect
        import bisect as _bisect

        overlap = 0
        for s_, e_ in zip(sa, sb):
            i = max(0, _bisect.bisect_right(merged_s, s_) - 1)
            while i < len(merged_s) and merged_s[i] < e_:
                lo = s_ if s_ > merged_s[i] else merged_s[i]
                hi = e_ if e_ < merged_e[i] else merged_e[i]
                if hi > lo:
                    overlap += hi - lo
                i += 1
        m["a2a_compute_overlap_s"] = float(overlap / 1e9)
        residency = int(sr.dur.sum())
        m["a2a_compute_overlap_pct"] = float(100.0 * overlap / residency) if residency else 0.0
    else:
        m["a2a_compute_overlap_s"] = 0.0
        m["a2a_compute_overlap_pct"] = 0.0

    # --- counter 6: GPU-side ------------------------------------------------
    m["sendrecv_gpu_s"] = float(q(
        "SELECT COALESCE(SUM(dur),0)/1e9 AS s FROM slice "
        "WHERE category='kernel' AND name LIKE '%SendRecv%'"
    ).s.iloc[0])
    m["kernel_count"] = int(q(
        "SELECT COUNT(*) AS n FROM slice WHERE category='kernel'"
    ).n.iloc[0])
    m["dtoh_pinned_count"] = int(q(
        "SELECT COUNT(*) AS n FROM slice WHERE category='gpu_memcpy' "
        "AND name='Memcpy DtoH (Device -> Pinned)'"
    ).n.iloc[0])

    # --- counter 7: GPU-union idle inside ProfilerStep#0 --------------------
    step = q("SELECT ts, dur FROM slice WHERE name='ProfilerStep#0'")
    if len(step) == 0:
        # fall back to whole-trace extent
        ext = q("SELECT MIN(ts) AS lo, MAX(ts+dur) AS hi FROM slice WHERE dur>0")
        S, E = float(ext.lo.iloc[0]), float(ext.hi.iloc[0])
    else:
        S = float(step.ts.iloc[0])
        E = S + float(step.dur.iloc[0])
    gpu = q(
        "SELECT ts, dur FROM slice "
        "WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0"
    )
    a = gpu.ts.to_numpy(dtype=np.int64)
    b = a + gpu.dur.to_numpy(dtype=np.int64)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    busy = 0
    cs, ce = int(a[0]), int(b[0])
    for s_, e_ in zip(a[1:], b[1:]):
        if s_ <= ce:
            if e_ > ce:
                ce = int(e_)
        else:
            busy += ce - cs
            cs, ce = int(s_), int(e_)
    busy += ce - cs
    m["gpu_idle_s"] = float(((E - S) - busy) / 1e9)
    m["step_wall_s"] = float((E - S) / 1e9)

    return m


# ---------------------------------------------------------------------------
# expectation profiles
# ---------------------------------------------------------------------------

# bound kinds: "eq" (exact), "tol" (|v-e| <= tol*e), "max", "min"
def _eq(v):
    return {"eq": v}

def _tol(v, t=0.02):
    return {"eq": v, "tol": t}

def _mx(v):
    return {"max": v}

def _rng(lo, hi):
    return {"min": lo, "max": hi}


PROFILES = {
    # The exp05d baseline column itself (counts exact, times ±2%).
    "baseline-exp05d": {
        "nonzero_calls": _eq(26684),
        "nonzero_cpu_s": _tol(20.594605),
        "nonzero_gt5ms": _eq(160),
        "nonzero_layout": _eq(26524),
        "streamsync_calls": _eq(29131),
        "streamsync_cpu_s": _tol(19.98),
        "memcpy_calls": _eq(33630),
        "memcpy_cpu_s": _tol(15.01),
        "memcpy_gt1ms": _eq(587),
        "eventsync_calls": _eq(300),
        "eventsync_cpu_s": _tol(4.13),
        "eventsync_gt1ms": _eq(300),
        "sendrecv_gpu_s": _tol(24.48),
        "kernel_count": _tol(355627, 0.02),  # v56.1 shell; v57.2 measures 361,032 (see header note)
        "dtoh_pinned_count": _eq(29502),
        "gpu_idle_s": _tol(4.74),
    },
    # REVIEW_FIXA §2c accept column (FIX A + B + F on).
    "post-patch-ABF": {
        "nonzero_calls": _mx(400),          # ~200 expected
        "nonzero_cpu_s": _mx(0.5),
        "nonzero_gt5ms": _mx(4),            # FIX A removes 156; 4 step-level _index_put_impl_ remain
        "nonzero_layout": _mx(400),         # FIX B: ~170 (85 x 2 mb first-layers)
        "streamsync_calls": _mx(1000),
        "streamsync_cpu_s": _mx(0.6),
        "memcpy_calls": _mx(36000),         # informational bound
        "memcpy_cpu_s": _mx(1.0),           # requires FIX F; relax via --set if F off
        "memcpy_gt1ms": _mx(8),             # FIX F: ~2/step cache-miss D2Hs remain (was 587)
        "eventsync_calls": _mx(500),        # 300 dispatcher + 156 FIX A events
        "eventsync_cpu_s": _mx(4.5),
        "eventsync_gt1ms": _mx(300),        # FIX A's events must be sub-ms
        "sendrecv_gpu_s": _rng(23.26, 25.70),  # 24.48 +/-5%, unchanged
        "kernel_count": _mx(230000),        # FIX B removes ~150k launches
        "dtoh_pinned_count": _mx(3000),
        "gpu_idle_s": _mx(4.5),             # 4.74 baseline minus 0.7 sync idle
    },
    # FIX A only: drains + their streamSyncs go; everything else stays.
    "post-patch-A": {
        "nonzero_calls": _rng(26400, 26600),   # 26,684 - 156 = 26,528
        "nonzero_cpu_s": _mx(1.6),             # 20.59 - 19.20 = 1.39
        "nonzero_gt5ms": _mx(4),               # 156 gone; 4 _index_put_impl_ remain
        "nonzero_layout": _rng(26400, 26650),  # unchanged by A
        "streamsync_calls": _rng(28900, 29050),  # 29,131 - 156
        "streamsync_cpu_s": _mx(1.0),          # 19.98 - 19.18 = 0.80
        "memcpy_calls": _mx(34000),            # +156 tiny flag copies
        "memcpy_cpu_s": _mx(15.5),             # unchanged by A (tolists remain)
        "memcpy_gt1ms": _rng(580, 600),        # unchanged by A
        "eventsync_calls": _rng(440, 470),     # 300 + 156
        "eventsync_cpu_s": _mx(4.5),
        "eventsync_gt1ms": _mx(300),           # A's 156 events must be sub-ms
        "sendrecv_gpu_s": _rng(23.26, 25.70),
        "kernel_count": _mx(370000),           # unchanged by A (sanity bound)
        "dtoh_pinned_count": _mx(30000),       # 29,502 + 156 flag copies
        "gpu_idle_s": _mx(4.4),                # 4.74 - 0.41 (nonzero-window idle)
    },
    # FIX B only: layout nonzeros + launch storm + pinned DtoH go; drains STAY.
    "post-patch-B": {
        "nonzero_calls": _mx(600),             # ~174 + 4
        "nonzero_cpu_s": _rng(18.5, 19.7),     # drains remain: 20.59 - 1.34 = 19.25
        "nonzero_gt5ms": _rng(155, 165),       # 160 drains UNCHANGED by B
        "nonzero_layout": _mx(400),            # ~170
        "streamsync_calls": _mx(3000),         # 29,131 - 26,520 = 2,611
        "streamsync_cpu_s": _rng(18.5, 19.7),  # drain syncs remain
        "memcpy_calls": _mx(6000),             # 33,630 - ~29k nonzero internals
        "memcpy_cpu_s": _rng(14.0, 15.5),      # tolist blocking copies UNCHANGED by B
        "memcpy_gt1ms": _rng(580, 600),        # unchanged by B
        "eventsync_calls": _rng(295, 305),     # unchanged by B
        "eventsync_cpu_s": _mx(4.5),
        "eventsync_gt1ms": _mx(300),
        "sendrecv_gpu_s": _rng(23.26, 25.70),
        "kernel_count": _mx(230000),           # the FIX B row: launch storm gone
        "dtoh_pinned_count": _mx(3000),        # nonzero internals gone
        "gpu_idle_s": _mx(4.75),               # ~unchanged by B (small syncs didn't idle GPU)
    },
    # FIX A + B, no F: tolist memcpys stay.
    "post-patch-AB": {
        "nonzero_calls": _mx(600),
        "nonzero_cpu_s": _mx(0.3),
        "nonzero_gt5ms": _mx(4),
        "nonzero_layout": _mx(400),
        "streamsync_calls": _mx(3000),
        "streamsync_cpu_s": _mx(0.3),
        "memcpy_calls": _mx(6000),
        "memcpy_cpu_s": _mx(15.5),             # F off: 14.7s remains
        "memcpy_gt1ms": _rng(580, 600),        # F off: unchanged
        "eventsync_calls": _rng(440, 470),
        "eventsync_cpu_s": _mx(4.5),
        "eventsync_gt1ms": _mx(300),
        "sendrecv_gpu_s": _rng(23.26, 25.70),
        "kernel_count": _mx(230000),
        "dtoh_pinned_count": _mx(3200),        # ~2,100 + 156 flag copies
        "gpu_idle_s": _mx(4.4),
    },
}
PROFILES["post-patch"] = PROFILES["post-patch-ABF"]  # alias

# ---------------------------------------------------------------------------
# post-patch-BFC-4mb131k: SHIP config (B+F on, A parked) plus FIX C
# (BT_MOE_DISPATCH_REPLAY_CACHE) at the 4 x 131,072 bench shape.
#
# FIX C removes the recompute replay's share of the dispatcher host syncs:
# the replay's d2h_event.synchronize() (300/step at 4mb, parent
# CheckpointFunctionBackward), the replay's preprocess all-gather (300/step),
# and the replay's side-stream D2H copies (~half the dispatcher dtoh). The
# first-pass (fwd) halves are UNTOUCHED. Everything B+F did is unchanged.
#
# Sharp rows:
#   eventsync_dispatcher_replay_calls  300 -> ~0 (the FIX C signature)
#   eventsync_dispatcher_fwd_calls     300 unchanged
#   dispatcher_allgather_calls         600 -> 300 (GPU-side Long allgather on
#                                      the expert group; validated 600 exactly
#                                      on gated-v2-4mb-steady)
# A is parked in this profile: eventsync_a_* rows assert its absence. For an
# A+B+F+C capture, override with --set eventsync_calls.min=600 --set
# eventsync_calls.max=630 --set eventsync_a_calls.eq=312 (etc.).
#
# RE-BASELINE (2026-08-09/10, boltzmann; approved by kepler, not silent):
# nonzero_cpu_s 21.0 -> 28.0, streamsync_cpu_s 21.0 -> 28.5. Mechanism: with
# the replay throttle removed, the CPU parks at the DSA-bwd drains instead —
# redistributed replay-throttle wait, drains 97.4% GPU-covered, eventsync+
# nonzero wait conserved (34.21s C′-on vs 34.01s C′-off control), wall flat.
# Measured on the C′ timed arm: 24.19/24.12 and 24.50/24.83 (rank0/rank8).
# C′-OFF captures must still be checked against the OLD bounds: use the
# post-patch-W1-4mb131k profile (keeps <=21.0 on both rows) or
#   --set nonzero_cpu_s.max=21.0 --set streamsync_cpu_s.max=21.0
# ---------------------------------------------------------------------------
PROFILES["post-patch-BFC-4mb131k"] = {
    "nonzero_calls": _mx(700),           # A parked: 348 layout + 312 DSA-bwd drains + ~8
    "nonzero_cpu_s": _mx(28.0),          # RE-BASELINED C′-on (was 21.0 — see block above)
    "nonzero_gt5ms": _mx(330),           # A parked: 312 drains + ~4-8 step-level
    "nonzero_layout": _mx(450),
    "streamsync_calls": _mx(1000),       # A parked: drains' syncs + item syncs
    "streamsync_cpu_s": _mx(28.5),       # RE-BASELINED C′-on (was 21.0 — see block above)
    "memcpy_calls": _mx(12000),           # replay dispatcher dtoh copies gone (~-1.8k vs ABF)
    "memcpy_cpu_s": _mx(1.0),
    "memcpy_gt1ms": _mx(8),
    "eventsync_calls": _rng(290, 315),    # 300 fwd dispatcher; ~0 replay; A off
    "eventsync_cpu_s": _mx(15.0),         # informational: fwd dispatcher only, runahead-inflated
    "eventsync_gt1ms": _mx(320),
    "eventsync_a_calls": _eq(0),          # A parked
    "eventsync_a_cpu_s": _mx(0.1),
    "eventsync_a_gt1ms": _eq(0),
    "eventsync_dispatcher_cpu_s": _mx(15.0),  # informational
    "eventsync_dispatcher_fwd_calls": _rng(290, 310),     # 75 MoE x 4 mb, untouched
    "eventsync_dispatcher_replay_calls": _mx(10),         # THE FIX C ROW: 300 -> ~0
    "dispatcher_allgather_calls": _rng(280, 320),         # FIX C: 600 -> 300 (replay half gone)
    "sendrecv_gpu_s": _rng(22.0, 26.9),
    "kernel_count": _mx(430000),          # -300 replay all-gather kernels vs ABF
    "dtoh_pinned_count": _mx(3000),       # replay dispatcher dtoh gone (~1.5k vs ABF)
    "gpu_idle_s": _mx(3.5),
}

# ---------------------------------------------------------------------------
# post-patch-W1-4mb131k: SHIP config (B+F) + W1 (BT_MOE_PROBS_A2A_COMM=1 —
# probs a2a on a second EP communicator/stream). A parked, C off.
# Baseline reference: gated-v2-4mb-steady (step 46.678 s; token-a2a 20.90 s /
# 1800; probs 5.13 s / 900 all on the token stream; dispatch->combine gap avg
# 10.85 ms). Metric definitions validated against that trace (2026-08-09):
# consecutive-pair gap reproduces 10.85/16.64 exactly; overlap reproduces
# 0.395 s / 1.5%.
# Sharp rows (TRACE_ACCEPTANCE W1):
#   probs_a2a_off_token_stream   0 -> 900 (Float rows move to a distinct stream)
#   dispatch_combine_gap_avg_ms  10.85 -> <= 7.0 (GEMM-only serial slot)
#   token_a2a_gpu_s              20.90 +/-5% (token comm unperturbed)
#   step_wall_s                  46.678 -> <= 43.18 (-3.5 s; accept >=70% of -5.1 model)
# Log/bench gates (REMINDER rows): telemetry issues==waits==900; loss BITWISE
# (not 2e-3) vs gates-off same-seed.
# ---------------------------------------------------------------------------
PROFILES["post-patch-W1-4mb131k"] = {
    "nonzero_calls": _mx(700),           # A parked: 348 layout + 312 DSA-bwd drains + ~8
    "nonzero_cpu_s": _mx(21.0),          # A parked: the 312 drains remain (~19s, M-invariant)
    "nonzero_gt5ms": _mx(330),           # A parked: 312 drains + ~4-8 step-level
    "nonzero_layout": _mx(450),
    "streamsync_calls": _mx(1000),       # A parked: drains' syncs + item syncs
    "streamsync_cpu_s": _mx(21.0),       # A parked: the drain syncs remain (~19-20s)
    "memcpy_calls": _mx(14000),
    "memcpy_cpu_s": _mx(1.0),
    "memcpy_gt1ms": _mx(8),
    "eventsync_calls": _rng(590, 620),    # C off: 600 dispatcher (300 fwd + 300 replay)
    "eventsync_cpu_s": _mx(40.0),         # informational (runahead-inflated)
    "eventsync_gt1ms": _mx(650),
    "eventsync_dispatcher_fwd_calls": _rng(290, 310),
    "eventsync_dispatcher_replay_calls": _rng(290, 310),   # C off: replay syncs present
    "token_a2a_calls": _eq(1800),         # W1 does not change token call count
    "token_a2a_gpu_s": _rng(19.85, 21.95),  # 20.90 +/-5% — W1 must not perturb token comm
    "probs_a2a_calls": _eq(900),
    "probs_a2a_off_token_stream": _eq(900),  # THE W1 ROW: all Float rows off the token stream
    "dispatch_combine_gap_avg_ms": _mx(7.0),  # 10.85 -> ~5 (GEMM-only)
    "sendrecv_gpu_s": _rng(22.0, 26.9),
    "kernel_count": _mx(430000),
    "dtoh_pinned_count": _mx(5500),
    "gpu_idle_s": _mx(3.5),
    "step_wall_s": _mx(43.18),            # -3.5 s vs 46.678 baseline
}

# ---------------------------------------------------------------------------
# post-patch-W2-4mb131k: SHIP config (B+F) + W2 v1 (BT_MOE_A2A_PIPELINE=2 —
# intra-layer chunked pipeline, K=2 by local-expert groups). A parked, C/W1 off.
# Sharp rows (TRACE_ACCEPTANCE W2):
#   token_a2a_calls            1800 -> ~3600 (2x per phase at ~half payload)
#   token_a2a_in_nelems_p50    402.7M -> ~201M (dispatch chunks)
#   a2a_compute_overlap_pct    1.5 -> >= 15 (% of SendRecv residency)
#   step_wall_s                46.678 -> <= 44.68 (-2 s standalone)
# Log/bench gates (REMINDER rows): V1 numerics gate (torch.equal fwd + grads,
# incl. imbalance + zero-count-peer) BEFORE any timed run; peak mem delta
# <= +1 GiB at both shapes; loss bitwise.
# ---------------------------------------------------------------------------
PROFILES["post-patch-W2-4mb131k"] = {
    "nonzero_calls": _mx(700),           # A parked: 348 layout + 312 DSA-bwd drains + ~8
    "nonzero_cpu_s": _mx(21.0),          # A parked: the 312 drains remain (~19s, M-invariant)
    "nonzero_gt5ms": _mx(330),           # A parked: 312 drains + ~4-8 step-level
    "nonzero_layout": _mx(450),
    "streamsync_calls": _mx(1000),       # A parked: drains' syncs + item syncs
    "streamsync_cpu_s": _mx(21.0),       # A parked: the drain syncs remain (~19-20s)
    "memcpy_calls": _mx(14000),
    "memcpy_cpu_s": _mx(1.0),
    "memcpy_gt1ms": _mx(8),
    "eventsync_calls": _rng(590, 620),
    "eventsync_cpu_s": _mx(40.0),
    "eventsync_gt1ms": _mx(650),
    "eventsync_dispatcher_fwd_calls": _rng(290, 310),
    "eventsync_dispatcher_replay_calls": _rng(290, 310),
    "token_a2a_calls": _rng(3400, 3800),  # THE W2 ROW: ~2x = 3600
    "token_a2a_gpu_s": _rng(19.85, 24.0), # ~flat expected; loose upper for chunk overhead
    "token_a2a_in_nelems_p50": _rng(150e6, 280e6),  # ~201M (half of 402.7M; routing-dependent)
    "probs_a2a_calls": _eq(900),
    "probs_a2a_off_token_stream": _eq(0),  # W2 standalone: probs still serialized
    "a2a_compute_overlap_pct": {"min": 15.0},  # THE W2 OVERLAP ROW (baseline 1.5%)
    "sendrecv_gpu_s": _rng(22.0, 26.9),
    "kernel_count": _mx(460000),          # +chunked launches
    "dtoh_pinned_count": _mx(5500),
    "gpu_idle_s": _mx(3.5),
    "step_wall_s": _mx(44.68),            # -2.0 s vs 46.678 baseline
}

# ---------------------------------------------------------------------------
# baseline-4mb131k: DERIVED unpatched expectations for the 4 x 131,072 bench
# shape. Pure mb-scaling of the exp05d measurements (M: 2 -> 4, layer-passes
# 312 -> 624, per-pass counts unchanged).
#
# VALIDATION vs the measured unpatched-4mb-steady capture (2026-08-09,
# gatesoff-step3, wall 53.95s): 10/16 PASS.
#   COUNT rows: all exact (nonzero 53,368; layout 53,048; streamsync 58,223;
#   memcpy 67,200; eventsync 600; kernels 704,681; dtoh-pinned 58,982) — the
#   mb-scaling derivation is dead-on for counts.
#   WAIT-TIME rows: all MISSED HIGH by ~2x (nonzero_cpu 41.2->21.35s;
#   streamsync_cpu 40->19.6s; memcpy_cpu 30->13.9s; memcpy_gt1ms 1174->630;
#   eventsync_cpu 8.26->2.22s). CORRECTED SCALING LAW: per-block wait is
#   proportional to the compute-stream queue ahead of it ~ per-pass work ~
#   1/M at fixed total tokens, so total host-block CPU is ~M-INVARIANT
#   (nonzero 20.6->21.4s, memcpy 15.0->13.9s), not M-linear. The >1ms-threshold
#   COUNT rows (memcpy_gt1ms) fall with shape because per-call waits shorten,
#   not because fewer copies exist. Dispatcher eventsync per-call wait fell
#   2x beyond invariance (13.8ms->3.7ms) — queue-depth sensitive.
#   gpu_idle_s MISSED LOW (5.7 derived vs 8.52 measured): inter-kernel bubbles
#   scale with launch count (704k kernels at 4mb), not with wall.
# Lesson for future derivations: scale COUNTS by M, scale WAIT-TIME TOTALS by
# ~1 (invariant), and scale launch-bubble idle by kernel-count ratio.
# ---------------------------------------------------------------------------
# Values below are the MEASURED unpatched-4mb-steady baseline (post-validation),
# tolerances ±3% counts / ±10% times; derivation history in the header above.
PROFILES["baseline-4mb131k"] = {
    "nonzero_calls": _tol(53368, 0.03),
    "nonzero_cpu_s": _tol(21.35, 0.10),     # derived 41.2 — wait-time ~M-invariant, see header
    "nonzero_gt5ms": _rng(312, 328),
    "nonzero_layout": _tol(53048, 0.02),
    "streamsync_calls": _tol(58223, 0.03),
    "streamsync_cpu_s": _tol(19.64, 0.10),
    "memcpy_calls": _tol(67200, 0.05),
    "memcpy_cpu_s": _tol(13.85, 0.10),      # derived 30.0 — see header
    "memcpy_gt1ms": _tol(630, 0.10),        # derived 1174 — threshold-count falls with per-call wait
    "eventsync_calls": _eq(600),
    "eventsync_cpu_s": _tol(2.22, 0.15),    # derived 8.26 — queue-depth sensitive, see header
    "eventsync_gt1ms": _tol(582, 0.05),
    "sendrecv_gpu_s": _rng(20.8, 28.2),     # bytes/step conserved
    "kernel_count": _tol(704681, 0.03),
    "dtoh_pinned_count": _tol(58982, 0.03),
    "gpu_idle_s": _tol(8.52, 0.15),         # launch-bubble dominated at 4mb, see header
}

# ---------------------------------------------------------------------------
# Derivation of post-patch-ABF-4mb131k (bench shape 4 x 131,072 = 524,288
# tok/step, max_seq_len=131072 boot, same EP16/CP16 mesh, 78 layers / 75 MoE).
#
# Shape scaling vs the exp05d baseline (2 mb x 262,144):
#   microbatches M: 2 -> 4. Layer-passes: 2*L*M = 312 -> 624 (fwd + replay).
#   Per-pass op counts are UNCHANGED (CP16 => 85 = 17x5 layout nonzeros per
#   layer-pass regardless of doc count; 1 FSA-bwd drain per layer-pass).
#   Per-rank tokens per pass halve (16,384 -> 8,192): per-call sizes shrink,
#   per-call counts per pass are unchanged.
#
# Row-by-row (E[x] = expected post-patch value at 4mb):
#   nonzero_calls      E=85x4 (FIX B first-layer computes per mb) + ~4-6
#                      step-level = ~345.  Unpatched 4mb ~= 85x624 + 312 + 4
#                      ~= 53,356.
#   nonzero_cpu_s      E=340 x ~50us + ~0.05s step-level ~= 0.07s.
#   nonzero_gt5ms      312 FSA-bwd drains removed by A. Remaining: the 4
#                      step-level _index_put_impl_ drains (NON-scaling) + up to
#                      ~4 per-mb metrics (packing.py:415 runs per mb, so the
#                      "other" 2 at 2mb may scale to ~4 at 4mb). Bound <= 8.
#   nonzero_layout     E=85 x 4 = 340 (cache is per-mb carrier; replay hits).
#                      MEASURED refinement (patched 4mb capture, 2026-08-09):
#                      348 = 87 x 4 — the per-mb first-compute is 87, not 85
#                      (+2/mb from a secondary layout path; bounded, immaterial).
#   streamsync_calls   E=340 layout + ~700 non-RoPE item syncs (1.1/layer-pass
#                      x 624) + 34 step-level (NON-scaling). RoPE's ~6
#                      syncs/layer-pass (2 tolist + 4 item) are host-side under
#                      FIX F. Unpatched 4mb ~= 57,900.
#   streamsync_cpu_s   step-level items ~0.26s (NON-scaling) + small syncs.
#   memcpy_calls       E=374 layout first-layer internals + 3,600 dispatcher
#                      dtoh (6 x 75 x 4 x 2) + 312 A flag copies + 4 F misses
#                      + ~6,600 DtoD (mb-scaled) + ~700 item copies ~= 11.6k.
#                      Unpatched 4mb ~= 73k. (Loose row; discriminators are
#                      memcpy_cpu_s / memcpy_gt1ms.)
#   memcpy_cpu_s       E=4 F cache-miss D2Hs (early-fwd, shallow queue) + sub-ms
#                      tail ~= 0.4-0.7s.
#   memcpy_gt1ms       E=4/step: one blocking D2H per unique cu_seqlens tensor
#                      = per-mb FIX F cache miss (M=4). Bound 2x margin.
#   eventsync_calls    E=600 dispatcher (75 MoE x 4 mb x 2 passes; UNTOUCHED by
#                      the gates) + 312 A flag events (78 x 4, replay pass only,
#                      grad-gated) = 912.
#   eventsync_cpu_s    dispatcher 4.13s at 2mb x 2 (mb-scaling; pre-existing
#                      cost the gates don't touch) ~= 8.3s. A's 312 events are
#                      us-scale (the REPLAY argument — this is the check).
#   eventsync_gt1ms    E=600 (all dispatcher; was 300/300 at 2mb). A's 312 must
#                      be sub-ms. Bound allows ~50 marginal.
#   sendrecv_gpu_s     ASSUMPTION (loosest row): total a2a bytes/step conserved
#                      (same 524K tokens, same EP16/CP16) — call count doubles,
#                      per-call size halves; per-call latency at these sizes is
#                      straggler-dominated, so total may drift. Bound = 24.48
#                      +/-10%; a bigger move indicates a config/measurement
#                      discrepancy, not a patch effect (the gates touch no comm).
#   kernel_count       E=(355,627 - ~150k FIX B launch removal) x 2 = ~411k.
#                      Unpatched 4mb ~= 711k.
#   dtoh_pinned_count  E=680 layout first-layer (340 x ~2) + 3,600 dispatcher +
#                      312 A flags ~= 4.6k. Unpatched 4mb ~= 62k.
#                      MEASURED refinement (patched 4mb capture): 3,115 with A
#                      inactive (no 312 flag copies) — dispatcher dtoh is ~4
#                      tensors/layer-pass, not 6 (some moves are conditional).
#   gpu_idle_s         E=ramp/tail ~0.5s (NON-scaling) + launch bubbles ~4s
#                      (scales with launches: 411k/356k x 3.5s) - 0.7s
#                      sync-attributable idle removed by A+F ~= 4.5-5s. Loose
#                      row: catches a reintroduced drain class (would add tens
#                      of seconds), not tuned for small moves.
# ---------------------------------------------------------------------------
PROFILES["post-patch-ABF-4mb131k"] = {
    "nonzero_calls": _mx(500),
    "nonzero_cpu_s": _mx(0.4),
    "nonzero_gt5ms": _mx(8),
    "nonzero_layout": _mx(450),
    "streamsync_calls": _mx(1600),
    "streamsync_cpu_s": _mx(0.7),
    "memcpy_calls": _mx(14000),
    "memcpy_cpu_s": _mx(1.0),
    "memcpy_gt1ms": _mx(8),
    "eventsync_calls": _rng(850, 950),
    # eventsync split (validated on gated-v2-4mb-steady, 2026-08-09): A's 312 flag
    # events are us-scale (0.46s total, p50 5.5us, 12 >1ms) — the sharp rows for
    # A's replay argument. The dispatcher's 600 are INFLATED by host runahead
    # under decompression (2.22s unpatched -> 33.6s gated): expected side effect,
    # informational bound only. The pre-split eventsync_cpu_s <= 9.0 row was
    # retired — it encoded the wrong (us-scale-for-everything) assumption.
    "eventsync_cpu_s": _mx(40.0),            # informational (dispatcher-dominated)
    "eventsync_gt1ms": _mx(650),
    "eventsync_a_calls": _eq(312),           # A activation proof (78 layers x 4 mb)
    "eventsync_a_cpu_s": _mx(1.5),           # measured 0.459s
    "eventsync_a_gt1ms": _mx(25),            # measured 12
    "eventsync_dispatcher_cpu_s": _mx(40.0), # informational; 33.6s measured gated
    "sendrecv_gpu_s": _rng(22.0, 26.9),
    "kernel_count": _mx(430000),
    "dtoh_pinned_count": _mx(5500),
    # measured gated-v2: 1.82s (unpatched 8.52s). Bound = clear separation from
    # unpatched while allowing boot variance.
    "gpu_idle_s": _mx(3.5),
}


def check(m: dict, profile: dict) -> list[tuple[str, float, str, bool]]:
    rows = []
    for key, spec in profile.items():
        if key not in m:
            continue
        v = m[key]
        if "eq" in spec:
            tol = spec.get("tol", 0.0)
            ok = abs(v - spec["eq"]) <= tol * abs(spec["eq"])
            exp = f"== {spec['eq']}" + (f" ±{tol*100:.0f}%" if tol else "")
        else:
            ok = True
            parts = []
            if "max" in spec:
                ok &= v <= spec["max"]
                parts.append(f"<= {spec['max']}")
            if "min" in spec:
                ok &= v >= spec["min"]
                parts.append(f">= {spec['min']}")
            exp = " and ".join(parts)
        rows.append((key, v, exp, bool(ok)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="path to .pt.trace.json")
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES), help="expectation profile")
    ap.add_argument("--set", action="append", default=[], metavar="METRIC.BOUND=VALUE",
                    help="override one bound, e.g. --set memcpy_cpu_s.max=15.5 (repeatable)")
    ap.add_argument("--overrides", help="JSON file with {metric: {bound: value}} overrides")
    ap.add_argument("--json", dest="json_out", help="write measured metrics to this JSON file")
    args = ap.parse_args()

    profile = json.loads(json.dumps(PROFILES[args.profile]))  # deep copy
    if args.overrides:
        with open(args.overrides) as f:
            for k, spec in json.load(f).items():
                profile.setdefault(k, {}).update(spec)
    for item in args.set:
        try:
            lhs, rhs = item.split("=", 1)
            metric, bound = lhs.rsplit(".", 1)
            assert bound in ("min", "max", "eq", "tol")
            profile.setdefault(metric, {})[bound] = float(rhs)
        except (ValueError, AssertionError):
            ap.error(f"bad --set '{item}', expected METRIC.BOUND=VALUE with BOUND in min|max|eq|tol")

    m = measure(args.trace)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(m, f, indent=2)

    rows = check(m, profile)
    width = max(len(k) for k, _, _, _ in rows)
    print(f"\ntrace: {args.trace}")
    print(f"profile: {args.profile}   (step wall measured: {m['step_wall_s']:.2f}s)\n")
    n_fail = 0
    for key, v, exp, ok in rows:
        n_fail += not ok
        vs = f"{v:,.4f}" if isinstance(v, float) and not float(v).is_integer() else f"{int(v):,}"
        print(f"  {'PASS' if ok else 'FAIL'}  {key:<{width}}  {vs:>12}  expected {exp}")
    print(f"\n  REMINDER  fallback_counter  —  §2c counter 5 is log-based: check trainer logs")
    print(f"            for the patch's per-step fallback count (expect 0/156 per step at")
    print(f"            2 mb/step, 0/312 at 4 mb/step — 78 layers x microbatches).")
    print(f"  REMINDER  fixc_counters     —  FIX C is log-based too: check trainer logs for")
    print(f"            'BT_MOE_DISPATCH_REPLAY_CACHE window' — expect hits=300/step at")
    print(f"            4 mb/step (75 MoE x 4 mb), misses=0, shape_mismatches=0; and the")
    print(f"            one-time 'armed' + 'first replay cache hit' lines at boot.")
    print(f"  REMINDER  w1/w2 gates       —  W1: telemetry issues==waits==900/step on the new")
    print(f"            comm; loss BITWISE-identical (not 2e-3) vs gates-off same-seed.")
    print(f"            W2: V1 numerics gate (torch.equal fwd output + input grads, incl.")
    print(f"            imbalance + zero-count-peer) BEFORE any timed run; peak mem delta")
    print(f"            <= +1 GiB at 131k x d4 AND 16k x d32; loss bitwise.")
    print(f"\n{'ALL PASS' if n_fail == 0 else f'{n_fail} ROW(S) FAILED'}\n")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
