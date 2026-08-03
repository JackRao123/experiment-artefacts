#!/usr/bin/env python
"""LPS-1003 arming-condition lab.

Single GPU, single process, UNFIXED wheel (PYTHONPATH shadow). Geometry =
the PR #875 CI repro: out [8192, 73728] fp32 = 2.25 GiB > 2^31 B, so the
-inf prefill is two launches split at row 4096; F2 = rows [4096, 8192).

Per rep: lever -> exec1 -> exec2 -> scan both execs independently +
bitwise compare. "Fired" = fully-erased rows (whole row -inf where the
causal reference expects finite scores).

Levers (--mode):
  warm            no lever (control)
  ec              torch.cuda.empty_cache() before each rep (arming lever)
  ec_premap       empty_cache, then map two same-size dummies (fill+free)
                  so BOTH execs' out allocs reuse warm, already-mapped blocks
  ec_prealloc     out= passed in, allocated once at boot; empty_cache lever
  ec_map_elsewhere out prealloc'd; lever = map+fill+free+empty_cache a big
                  UNRELATED buffer (mapping churn without touching out)

Flags: --warmup-small (sub-threshold call before rep0: same compile_key,
same cubins, same tvm-ffi path, single-launch fill so it cannot fire),
--prelaunch-streams N (N dummy launches on N distinct streams at boot).

Every witness/detector buffer is preallocated once (allocation churn arms
the race — investigation pitfall list).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["warm", "ec", "ec_premap", "ec_prealloc",
                             "ec_map_elsewhere", "busy", "busy_ec"])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-small", action="store_true")
    ap.add_argument("--prelaunch-streams", type=int, default=0)
    ap.add_argument("--expect-conn", default=None,
                    help="expected CUDA_DEVICE_MAX_CONNECTIONS value; "
                         "default expects UNSET")
    ap.add_argument("--allow-fixed", action="store_true",
                    help="permit the FIXED wheel (control runs)")
    args = ap.parse_args()

    conn = os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS")
    if args.expect_conn is None:
        assert conn is None, f"conn must be UNSET, got {conn!r}"
    else:
        assert conn == args.expect_conn, f"conn={conn!r} != {args.expect_conn!r}"

    import torch

    from cudnn.deepseek_sparse_attention.indexer_forward import _interface as iface

    src_path = iface.__file__
    src = open(src_path).read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    is_fixed = "_get_kernel_stream" in src
    if not args.allow_fixed:
        assert not is_fixed, f"FIXED wheel resolved at {src_path} — shadow failed"

    dev = torch.cuda.get_device_name()
    cap = torch.cuda.get_device_capability()
    assert cap >= (10, 0), f"need sm100+, got {cap}"

    header = {
        "header": True,
        "mode": args.mode,
        "reps": args.reps,
        "warmup_small": args.warmup_small,
        "prelaunch_streams": args.prelaunch_streams,
        "interface": src_path,
        "interface_sha256": sha,
        "torch": torch.__version__,
        "device": dev,
        "capability": list(cap),
        "conn": conn,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "module_loading": os.environ.get("CUDA_MODULE_LOADING"),
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
    }

    fout = open(args.out, "a", buffering=1)

    def emit(rec: dict) -> None:
        line = json.dumps(rec)
        fout.write(line + "\n")
        print(line, flush=True)

    emit(header)

    device = "cuda"
    torch.manual_seed(1234)
    N_HEADS, HEAD_DIM = 32, 128
    TOTAL_Q, SEG_K, N_SEGS = 8192, 73728, 2
    SEG_Q = TOTAL_Q // N_SEGS
    TOTAL_K = SEG_K * N_SEGS
    OFFSET = SEG_K - SEG_Q

    case = {
        "q": torch.randn(TOTAL_Q, N_HEADS, HEAD_DIM, dtype=torch.bfloat16,
                         device=device),
        "k": torch.randn(TOTAL_K, 1, HEAD_DIM, dtype=torch.bfloat16,
                         device=device),
        "w": torch.randn(TOTAL_Q, N_HEADS, dtype=torch.bfloat16, device=device),
        "cu_q": torch.arange(0, TOTAL_Q + 1, SEG_Q, dtype=torch.int32,
                             device=device),
        "cu_k": torch.arange(0, TOTAL_K + 1, SEG_K, dtype=torch.int32,
                             device=device),
        "offs": torch.tensor([OFFSET] * N_SEGS, dtype=torch.int32,
                             device=device),
    }

    def call(out=None):
        return iface.indexer_fwd(
            case["q"], case["k"], case["w"], ratio=1,
            cu_seqlens_q=case["cu_q"], cu_seqlens_k=case["cu_k"],
            max_seqlen_q=SEG_Q, max_seqlen_k=SEG_K,
            q_causal_offsets=case["offs"], out=out,
        )

    # reference: row r (segment b, local t) has finite cols
    # [0, min(OFFSET + t + 1, SEG_K)); the rest are -inf by mask.
    t_local = torch.arange(TOTAL_Q, device=device) % SEG_Q
    expected_fin = torch.clamp(OFFSET + t_local + 1, max=SEG_K)
    expected_neg = (SEG_K - expected_fin).to(torch.int64)  # [TOTAL_Q]

    # preallocated detector workspace
    negbuf = torch.empty(TOTAL_Q, SEG_K, dtype=torch.bool, device=device)
    rowcnt = torch.empty(TOTAL_Q, dtype=torch.int64, device=device)

    def scan(out: torch.Tensor) -> dict:
        torch.isneginf(out, out=negbuf)
        torch.sum(negbuf, dim=1, out=rowcnt)
        erased = (rowcnt == SEG_K) & (expected_neg < SEG_K)
        partial = (rowcnt != expected_neg) & ~erased
        n_er = int(erased.sum())
        n_pa = int(partial.sum())
        lo = hi = -1
        if n_er:
            idx = erased.nonzero().flatten()
            lo, hi = int(idx[0]), int(idx[-1])
        return {"erased": n_er, "lo": lo, "hi": hi, "partial": n_pa}

    if args.prelaunch_streams:
        streams = [torch.cuda.Stream() for _ in range(args.prelaunch_streams)]
        tiny = torch.ones(64, device=device)
        for s in streams:
            with torch.cuda.stream(s):
                tiny.mul_(1.0)
        torch.cuda.synchronize()
        emit({"prelaunched": args.prelaunch_streams})

    if args.warmup_small:
        # sub-threshold: out = 8192*65024*4 B = 2.13e9 < 2^31 -> single fill
        # launch, no F2 to reorder; same compile_key (seqlens are runtime
        # args) -> same DSA cubin, same fill cubin, same tvm-ffi executor.
        SK = 65024
        t0 = time.time()
        w_out = iface.indexer_fwd(
            case["q"], case["k"][: SK * N_SEGS], case["w"], ratio=1,
            cu_seqlens_q=case["cu_q"],
            cu_seqlens_k=torch.arange(0, SK * N_SEGS + 1, SK,
                                      dtype=torch.int32, device=device),
            max_seqlen_q=SEG_Q, max_seqlen_k=SK,
            q_causal_offsets=torch.tensor([SK - SEG_Q] * N_SEGS,
                                          dtype=torch.int32, device=device),
        )
        torch.cuda.synchronize()
        nneg = int(torch.isneginf(w_out).sum())
        del w_out
        emit({"warmup_small": True, "wall_s": round(time.time() - t0, 3),
              "neg_total": nneg})

    # busy modes: enqueue ~50ms of matmuls on stream 0 WITHOUT syncing, so
    # the channel has pending predecessor work when F1/F2/S are submitted.
    # Buffers preallocated once (no churn).
    busy_a = busy_c = None
    if args.mode in ("busy", "busy_ec"):
        busy_a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=device)
        busy_c = torch.empty_like(busy_a)
        torch.cuda.synchronize()

    prealloc = args.mode in ("ec_prealloc", "ec_map_elsewhere")
    OUT = BASE = None
    if prealloc:
        OUT = torch.empty(TOTAL_Q, SEG_K, dtype=torch.float32, device=device)
        BASE = torch.empty_like(OUT)

    def lever() -> None:
        if args.mode == "warm":
            return
        if args.mode == "busy":
            for _ in range(40):
                torch.matmul(busy_a, busy_a, out=busy_c)
            return  # NO sync — junk pending when the call submits
        if args.mode == "busy_ec":
            torch.cuda.empty_cache()
            for _ in range(40):
                torch.matmul(busy_a, busy_a, out=busy_c)
            return  # NO sync
        if args.mode in ("ec", "ec_prealloc"):
            torch.cuda.empty_cache()
            return
        if args.mode == "ec_premap":
            torch.cuda.empty_cache()
            d1 = torch.empty(TOTAL_Q, SEG_K, dtype=torch.float32,
                             device=device)
            d1.fill_(0.0)
            d2 = torch.empty(TOTAL_Q, SEG_K, dtype=torch.float32,
                             device=device)
            d2.fill_(0.0)
            torch.cuda.synchronize()
            del d1, d2
            return
        if args.mode == "ec_map_elsewhere":
            d = torch.empty(TOTAL_Q, SEG_K, dtype=torch.float32,
                            device=device)
            d.fill_(0.0)
            torch.cuda.synchronize()
            del d
            torch.cuda.empty_cache()
            return

    for rep in range(args.reps):
        t0 = time.time()
        lever()
        t_lever = time.time()
        if prealloc:
            call(out=OUT)
            torch.cuda.synchronize()
            s1 = scan(OUT)
            BASE.copy_(OUT)
            torch.cuda.synchronize()
            call(out=OUT)
            torch.cuda.synchronize()
            s2 = scan(OUT)
            equal = bool(torch.equal(OUT, BASE))
        else:
            out1 = call()
            torch.cuda.synchronize()
            out2 = call()
            torch.cuda.synchronize()
            s1 = scan(out1)
            s2 = scan(out2)
            equal = bool(torch.equal(out1, out2))
            del out1, out2
        emit({
            "rep": rep,
            "process_fresh": rep == 0,
            "mode": args.mode,
            "exec1": s1,
            "exec2": s2,
            "bitwise_equal": equal,
            "fired": s1["erased"] > 0 or s2["erased"] > 0 or not equal,
            "lever_s": round(t_lever - t0, 3),
            "wall_s": round(time.time() - t0, 3),
        })

    fout.close()
    print(f"DONE mode={args.mode} reps={args.reps}", flush=True)


if __name__ == "__main__":
    main()
