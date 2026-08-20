#!/usr/bin/env python3
"""memory_census.py — per-module activation-memory census from trainers
memory.rank<N>.pickle allocator snapshots (LPS-1062 rung 1, gate #1).

What it answers: at the peak memory instant of a post-plateau step, what was
live on the GPU, attributed to the module buckets of the Activation Placement
Plan (MoE experts / dispatcher / attention projections / core-attn internals /
norms-residual-router glue / loss head / base). The "glue" row is the
deliverable — measured, replacing the plan's 0.44-0.89 GiB/layer/mb
by-subtraction inference.

Snapshot format (torch private, produced by ProfilingSession with
stacks="python", context="alloc"): see the analyze-torch-traces-loops skill.
  snap["segments"]      : allocator segments; blocks[] carry state/size/frames
                          -> the live set AT DUMP TIME (step end).
  snap["device_traces"] : alloc/free event log with python stacks on allocs
                          -> replay for the live set AT THE PEAK INSTANT.
  snap["allocator_settings"]: PYTORCH_CUDA_ALLOC_CONF state.

Frees carry no stack (context="alloc" by design, pytorch#179536); attribution
is on the alloc side only. max_entries caps the event ring — if the log does
not cover a full final step, re-run with a bigger max_entries and say so.

Usage:
  python3 memory_census.py node0/ node1/            # dirs of pickles
  python3 memory_census.py memory.rank0.pickle ...  # or explicit files
  python3 memory_census.py node0/ --sets 0-7:76 8-15:40   # layer-mb sets/rank
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import sys
from pathlib import Path

# Bucket rules: (label, regex against frame filename). Checked top-down the
# alloc stack; first matching frame wins. Order matters — specific before
# general. File names verified against jackrao/lps-1062-actplace's vendored
# mcore (server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM)
# and the bridge package (server-megatron-bridge/src/trainers_server_megatron_bridge).
# DSA topk-selection stash (carnot's map, banach hazard #2): leader layers
# (~1 in 4) write their [1, 16384, 2048] integer token-selection into a plain
# Python dict on the per-microbatch object — OUTSIDE autograd's saved-tensor
# system, never popped within the microbatch. int64 = 268,435,456 B; int32 =
# 134,217,728 B. These are live allocations belonging to NO layer module's
# saved set; left unattributed they would land in the residual and inflate
# the glue row by ~4-16% on rank 0. Isolated by EXACT SIZE as their own
# census row; the alloc frames of size-matched blocks are printed for
# verification (expect dsa.py / indexer call sites). Paid TODAY under full
# recompute — NOT a new cost of the plan.
TOPK_STASH_SIZES = {268_435_456, 134_217_728}

RULES: list[tuple[str, str]] = [
    ("loss_head", r"chunked_lm_head\.py|/loss\.py|lm_head"),
    ("moe_experts", r"moe/experts\.py"),
    ("moe_dispatcher", r"token_dispatcher\.py|fused_a2a\.py|moe/ops/"),
    ("moe_router_layer", r"moe/moe_layer\.py|moe/router|router_replay\.py|moe/paged_stash\.py"),
    ("core_attn_dsa", r"dsa\.py|dsa_cudnn_kernels\.py|dsa_indexer|dsa_layout\.py|dsa_masking\.py|experimental_attention_variant|dot_product_attention\.py"),
    ("attn_projections", r"absorbed_mla\.py|glm_absorbed_mla\.py|multi_latent_attention\.py|/attention\.py"),
    ("norms", r"norm\.py|layer_norm|_norm\.py"),
    ("mlp", r"/mlp\.py"),
    ("checkpoint_wrapper", r"tensor_parallel/checkpoint|checkpoint\.py|random\.py"),
    ("embedding", r"embedding"),
    ("optimizer", r"/optimizer/|optimizer_param_scheduler|param_and_grad_buffer"),
    ("distributed_comm", r"/distributed/|c10d|process_group"),
    ("transformer_misc", r"transformer_block\.py|transformer_layer\.py|module\.py"),
    ("bridge_runner", r"training_runner\.py|training_stack\.py|backend\.py|thd_cp\.py|packer\.py"),
]

# Coarse roll-up into the plan's census rows.
PLAN_ROWS = {
    "core_attn_internals": ["core_attn_dsa"],
    "moe_act": ["moe_experts"],
    "combine_dispatch": ["moe_dispatcher"],
    "attn_proj": ["attn_projections"],
    "glue": ["moe_router_layer", "norms", "mlp", "transformer_misc", "checkpoint_wrapper"],
    "dsa_topk_stash": ["dsa_topk_stash"],  # OWN ROW — never folded into glue
    "loss_head": ["loss_head"],
    "base": ["embedding", "optimizer", "distributed_comm", "bridge_runner"],
}


def bucket_of(frames, size: int | None = None) -> str:
    if size in TOPK_STASH_SIZES:
        return "dsa_topk_stash"
    for fr in frames or []:
        fn = fr.get("filename", "")
        for label, pat in RULES:
            if re.search(pat, fn):
                return label
    return "unattributed"


def rank_of(path: Path) -> int:
    m = re.search(r"memory\.rank(\d+)(?:\.\w+)?\.pickle$", path.name)
    if not m:
        raise ValueError(f"cannot parse rank from {path.name}")
    return int(m.group(1))


def replay_peak(events):
    """Replay alloc/free events -> (peak_live_bytes, bucket Counter at peak,
    peak_reserved_bytes, n_events, window_span_s). free_requested removes;
    free_completed removes only if still tracked (it always follows
    free_requested for the same addr)."""
    live: dict[int, tuple[int, str]] = {}
    total = 0
    peak = 0
    peak_buckets: collections.Counter = collections.Counter()
    peak_stash_count = 0
    stash_frames: collections.Counter = collections.Counter()
    reserved = 0
    peak_reserved = 0
    n_events = 0
    t_first = None
    t_last = None
    for ev in events:
        n_events += 1
        t = ev.get("time_us")
        if t is not None:
            t_first = t if t_first is None else min(t_first, t)
            t_last = t if t_last is None else max(t_last, t)
        action = ev.get("action")
        addr = ev.get("addr")
        size = ev.get("size", 0)
        if action == "alloc":
            b = bucket_of(ev.get("frames"), size)
            live[addr] = (size, b)
            total += size
            if b == "dsa_topk_stash":
                top = next((f"{fr.get('filename','?').split('/')[-1]}:{fr.get('line','?')}"
                            for fr in (ev.get("frames") or [])), "<no stack>")
                stash_frames[top] += 1
            if total > peak:
                peak = total
                peak_buckets = collections.Counter()
                for sz, bk in live.values():
                    peak_buckets[bk] += sz
                peak_stash_count = sum(1 for _, bk in live.values()
                                       if bk == "dsa_topk_stash")
        elif action in ("free_requested", "free_completed"):
            if addr in live:
                total -= live.pop(addr)[0]
        elif action in ("segment_alloc", "segment_map"):
            reserved += size
            peak_reserved = max(peak_reserved, reserved)
        elif action in ("segment_free", "segment_unmap"):
            reserved -= size
    span_s = ((t_last - t_first) / 1e6) if (t_first is not None and t_last is not None) else 0.0
    return peak, peak_buckets, peak_reserved, n_events, span_s, peak_stash_count, stash_frames


def live_at_dump(snap):
    """Segments view: live allocated blocks at dump time, by bucket."""
    out: collections.Counter = collections.Counter()
    total = 0
    for seg in snap.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") == "active_allocated":
                out[bucket_of(blk.get("frames"), blk.get("size"))] += blk["size"]
                total += blk["size"]
    return total, out


def analyze(path: Path) -> dict:
    snap = pickle.load(open(path, "rb"))
    rank = rank_of(path)
    traces = snap.get("device_traces", [])
    events = [ev for t in traces for ev in t]
    events.sort(key=lambda e: e.get("time_us", 0))
    peak, peak_buckets, peak_reserved, n_events, span_s, stash_count, stash_frames = replay_peak(events)
    dump_total, dump_buckets = live_at_dump(snap)
    return {
        "rank": rank,
        "file": str(path),
        "n_events": n_events,
        "window_span_s": span_s,
        "topk_stash_count_at_peak": stash_count,
        "topk_stash_alloc_sites": dict(stash_frames.most_common(10)),
        "peak_live_gib": peak / 2**30,
        "peak_reserved_gib": peak_reserved / 2**30,
        "peak_buckets_gib": {k: v / 2**30 for k, v in peak_buckets.most_common()},
        "dump_live_gib": dump_total / 2**30,
        "dump_buckets_gib": {k: v / 2**30 for k, v in dump_buckets.most_common()},
        "allocator_settings": snap.get("allocator_settings"),
    }


def rollup(buckets: dict[str, float]) -> dict[str, float]:
    out = {}
    for row, members in PLAN_ROWS.items():
        out[row] = sum(buckets.get(m, 0.0) for m in members)
    out["unattributed"] = buckets.get("unattributed", 0.0)
    return out


def parse_rank_spec(spec: str) -> tuple[list[int], dict[str, int]]:
    """'0-7:moe70,dense6' or '8:moe40' -> ([ranks...], {kind: sets})."""
    rng, kinds = spec.split(":", 1)
    if "-" in rng:
        lo, hi = (int(x) for x in rng.split("-"))
        ranks = list(range(lo, hi + 1))
    else:
        ranks = [int(rng)]
    kv = {}
    for part in kinds.split(","):
        k, v = part.split("=") if "=" in part else part.split("x")
        kv[k.strip()] = int(v)
    return ranks, kv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="pickle files or dirs of them")
    ap.add_argument("--layer-sets", nargs="*",
                    default=["0-7:moe=70,dense=6", "8-15:moe=40,dense=0"],
                    help="layer-type x microbatch sets per rank range. Default is the "
                         "PP2/CP8/EP8 1F1B layout: first stage (ranks 0-7) = 3 dense "
                         "+ 35 MoE layers x 2 in-flight; last stage (ranks 8-15) = "
                         "40 MoE x 1 in-flight + loss head.")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--step-seconds-hint", type=float, default=23.0,
                    help="expected seconds per step; the event-ring coverage "
                         "check warns when a pickle's window is shorter "
                         "(default 23 = d2 @131k on the mission tree)")
    args = ap.parse_args()

    sets_per_rank: dict[int, dict[str, int]] = {}
    for spec in args.layer_sets:
        ranks, kv = parse_rank_spec(spec)
        for r in ranks:
            sets_per_rank[r] = kv

    files: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            files += sorted(p.glob("memory.rank*.pickle"),
                            key=lambda f: rank_of(f))
        else:
            files.append(p)
    if not files:
        sys.exit("no memory.rank*.pickle files found")

    results = [analyze(f) for f in files]
    for r in results:
        sets = sets_per_rank.get(r["rank"], {})
        r["layer_type_sets"] = sets
        r["plan_rows_at_peak_gib"] = rollup(r["peak_buckets_gib"])
        # Per-MoE-layer-mb figures: MoE buckets are divided by the MoE set
        # count ONLY (dense layers have no expert block or dispatcher — a
        # blended denominator corrupts the offload sizing, banach decision 4).
        moe_sets = sets.get("moe")
        if moe_sets:
            r["per_moe_layer_mb_gib"] = {
                k: r["plan_rows_at_peak_gib"].get(k, 0.0) / moe_sets
                for k in ("core_attn_internals", "moe_act", "combine_dispatch",
                          "attn_proj", "glue")
            }

    # Markdown summary
    print(f"# Memory census — {len(results)} ranks\n")
    hdr = ["rank", "moe/dense sets", "peak_live", "peak_rsvd", "dump_live"] + list(PLAN_ROWS) + ["unattr"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in results:
        rows = r["plan_rows_at_peak_gib"]
        s = r["layer_type_sets"]
        cells = [
            str(r["rank"]), f"{s.get('moe', '?')}/{s.get('dense', '?')}",
            f"{r['peak_live_gib']:.1f}", f"{r['peak_reserved_gib']:.1f}",
            f"{r['dump_live_gib']:.1f}",
        ] + [f"{rows.get(k, 0):.2f}" for k in PLAN_ROWS] + [f"{rows.get('unattributed', 0):.2f}"]
        print("| " + " | ".join(cells) + " |")
    print("\nGiB at the peak instant of the recorded window (device_traces replay).")

    # Event-ring coverage check: the replay peak is only trustworthy if the
    # 100k-entry ring covers at least one full post-plateau step. Two
    # symptoms of a wrapped ring: the window span is shorter than a step, or
    # the step-end dump holds MORE live bytes than the replay ever saw.
    step_hint = args.step_seconds_hint
    warned = False
    for r in results:
        if step_hint and r["window_span_s"] < step_hint:
            print(f"!! rank {r['rank']}: event window spans {r['window_span_s']:.1f}s "
                  f"< the ~{step_hint:.0f}s step — the max_entries ring wrapped; "
                  f"the replay peak is a PARTIAL-window peak. Re-run with a bigger "
                  f"max_entries (POST /memory_profile/start {{\"max_entries\": N}}).")
            warned = True
        if r["dump_live_gib"] > r["peak_live_gib"] * 1.10:
            print(f"!! rank {r['rank']}: dump-time live ({r['dump_live_gib']:.1f} GiB) "
                  f"exceeds the replay peak ({r['peak_live_gib']:.1f} GiB) by >10% — "
                  f"the ring likely wrapped mid-step; treat the replay composition "
                  f"as window-local and bump max_entries.")
            warned = True
    if not warned:
        spans = ", ".join(f"rank {r['rank']}: {r['window_span_s']:.0f}s" for r in results)
        print(f"[coverage] event window OK on all ranks ({spans}; "
              "dump-live <= replay-peak +10%).")

    print("\n## Per-MoE-layer-per-microbatch (GiB) — MoE buckets over MoE sets only")
    hdr2 = ["rank", "moe_sets", "core_attn", "moe_act", "combine", "attn_proj", "glue"]
    print("| " + " | ".join(hdr2) + " |")
    print("|" + "---|" * len(hdr2))
    for r in results:
        per = r.get("per_moe_layer_mb_gib")
        if not per:
            continue
        print("| " + " | ".join([
            str(r["rank"]), str(r["layer_type_sets"]["moe"]),
            f"{per['core_attn_internals']:.3f}", f"{per['moe_act']:.3f}",
            f"{per['combine_dispatch']:.3f}", f"{per['attn_proj']:.3f}",
            f"{per['glue']:.3f}",
        ]) + " |")

    # DSA topk-stash verification (banach hazard #2): count check against the
    # expected leader-layers x in-flight product, and the alloc sites so the
    # size-based isolation can be eyeballed. Expected (topk rule: leader iff
    # 1-based layer <=3 or (layer-3)%4==0): stage 0 = 11 leaders x 2 in-flight
    # = 22; stage 1 = 10 leaders x 1 = 10. The boot-time indexer_types dump
    # overrides these expectations if it disagrees.
    print("\n## DSA topk-selection stash (own row — excluded from glue; paid "
          "TODAY under full recompute, not a plan cost)")
    print("| rank | stash GiB at peak | count at peak | expected | top alloc sites |")
    print("|---|---|---|---|---|")
    for r in results:
        exp = {0: 22, 8: 10}.get(r["rank"])  # stage leaders; other ranks same stage
        if exp is None:
            exp = 22 if r["rank"] < 8 else 10
        gb = r["plan_rows_at_peak_gib"].get("dsa_topk_stash", 0.0)
        sites = ", ".join(f"{k} x{n}" for k, n in
                          list(r["topk_stash_alloc_sites"].items())[:3])
        print(f"| {r['rank']} | {gb:.2f} | {r['topk_stash_count_at_peak']} | "
              f"~{exp} | {sites} |")

    # Dense-layer isolation (banach decision 4): the uniform-MoE last stage
    # (rank 8 family) establishes the clean per-MoE-layer number; the first
    # stage's residual over its MoE sets isolates the dense-layer cost.
    last = [r for r in results if r["layer_type_sets"].get("dense") == 0 and r.get("per_moe_layer_mb_gib")]
    first = [r for r in results if r["layer_type_sets"].get("dense")]
    if last and first:
        ref = min(last, key=lambda r: r["rank"])
        ref_per = ref["per_moe_layer_mb_gib"]
        print(f"\n## Dense-layer isolation (reference: rank {ref['rank']} per-MoE-layer)")
        for r in first:
            s = r["layer_type_sets"]
            resid = {
                k: (r["plan_rows_at_peak_gib"].get(k, 0.0)
                    - s["moe"] * ref_per.get(k, 0.0)) / s["dense"]
                for k in ("core_attn_internals", "moe_act", "combine_dispatch",
                          "attn_proj", "glue")
            }
            r["per_dense_layer_mb_gib_residual"] = resid
            print(f"rank {r['rank']}: per-dense-layer-mb residual (GiB, /{s['dense']}): "
                  + ", ".join(f"{k}={v:.3f}" for k, v in resid.items()))
            print(f"  (cross-check: rank {r['rank']} MoE rows vs rank {ref['rank']} "
                  f"should agree within noise; a material gap = stage asymmetry, report it)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\n[written] {args.json_out}")


if __name__ == "__main__":
    main()
