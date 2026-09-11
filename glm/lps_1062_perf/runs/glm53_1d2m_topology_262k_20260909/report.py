"""Render benchmark results without hiding control variance or topology confounds."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LABELS=["cp8ep8","cp8ep1","cp1ep1"]
def main():
    results={label:json.loads((ROOT/label/"summary.json").read_text()) for label in LABELS}
    traces={label:json.loads((ROOT/label/"analysis/summary.json").read_text()) for label in LABELS}
    lines=["# GLM-5.3 1d2m, 262144 tokens: topology results","",
    "All results: one warmup, five Kineto-unprofiled controls with lightweight CUDA-event timers, one memory capture, one runtime capture. LoRA rank 32, full one-block recompute, native-FP8 expert weights. Timing is forward/backward only unless specified. Same main pin as prior full-model run.",
    "",
    "## Measured three-block proxy","",
    "| Topology | GPUs | FB mean ± SD (s) | Tokens/s total | Tokens/s/GPU | Peak allocated GiB | Peak reserved GiB |",
    "|---|---:|---:|---:|---:|---:|---:|"]
    for label,r in results.items():
        lines.append(f"| {label} | {r['gpus']} | {r['fb_s']['mean']:.4f} ± {r['fb_s']['sd']:.4f} | {r['tps']:,.0f} | {r['tps_per_gpu']:,.0f} | {r['peak_control_allocated_gib']:.2f} | {r['peak_control_reserved_gib']:.2f} |")
    lines+=["","Memory is the maximum allocator peak across the distributed ranks and five controls, not total physical HBM used by every library.",
    "","## Whole-block layer timings","",
    "GPU CUDA-event elapsed milliseconds, rank 0, mean ± sample SD over all five controls. Each block includes attention and MLP. Actual backward excludes recomputation; minor checkpoint boundary setup is included. All-rank raw records and conservative max-rank summaries are in each summary.json.",
    "",
    "| Topology | Block type | Forward ms | Recompute ms | Actual backward ms | Backward incl. recompute ms |",
    "|---|---|---:|---:|---:|---:|"]
    categories=["Dense / indexer","MoE / shared indices","MoE / indexer"]
    keys=["forward_ms","recompute_ms","backward_excluding_recompute_ms","backward_including_recompute_ms"]
    for label,r in results.items():
        for layer,category in enumerate(categories):
            row=r["layer_timings"]["rank0"]["by_layer"][str(layer)]
            values=[f"{row[k]['mean']:.2f} ± {row[k]['sd']:.2f}" for k in keys]
            lines.append("| "+ " | ".join([label,category]+values)+" |")
    lines+=["","## Weighted full-model estimate","",
    "Actual checkpoint counts: 3 dense/indexer + 57 MoE/shared + 18 MoE/indexer. No dense/shared blocks exist in the full model. The following is a timing extrapolation, not a claim that the full model fits the EP1 topologies.",
    "",
    "| Topology | Forward s | Recompute s | Actual backward s | Blocks FB mean ± SD (s) | FB including proxy non-block residual (s) |",
    "|---|---:|---:|---:|---:|---:|"]
    for label,r in results.items():
        l=r["layer_timings"]["rank0"];w=l["full_model_blocks_seconds"];t=l["full_blocks_fb_s"]
        lines.append(f"| {label} | {w['forward_ms']:.3f} | {w['recompute_ms']:.3f} | {w['backward_excluding_recompute_ms']:.3f} | {t['mean']:.3f} ± {t['sd']:.3f} | {l['full_fb_with_proxy_residual_s']['mean']:.3f} |")
    lines+=["","The residual is each control's measured FB time minus its measured three-block GPU time. Adding that residual assumes embedding/head/loss, orchestration, and gradient-finalization costs remain comparable; optimizer time is excluded.",
    "All five controls are retained. CP8EP1 has large outliers and its weighted estimate is correspondingly noisy; do not treat its mean as a reliable speed ranking against CP8EP8.",
    "The previous full CP8EP8 model measured 22.133 s FB. Agreement with the proxy is a sanity check, not an accuracy guarantee. Depth-dependent routing, index-cache lifetimes, communication overlap and memory pressure can change.",
    "",
    "## Communication evidence from the separate runtime captures","",
    "Kernel-duration sums in ms, rank 0, one profiled step including optimizer. These are not additive critical-path penalties or pure network-transfer time. HybridEP fuses permutation, dispatch/combine, and communication; local routing work can remain at EP1.",
    "",
    "| Topology | NCCL AllGather | NCCL ReduceScatter | NCCL AllReduce | Dispatch/combine-named kernels | Total GPU kernel union / span ms |",
    "|---|---:|---:|---:|---:|---:|"]
    for label,t in traces.items():
        get=lambda k:t["groups"].get(k,{}).get("sum_ms",0)
        lines.append(f"| {label} | {get('nccl_allgather'):.3f} | {get('nccl_reducescatter'):.3f} | {get('nccl_allreduce'):.3f} | {get('dispatch_combine'):.3f} | {t['gpu_union_busy_ms']:.1f} / {t['gpu_span_ms']:.1f} |")
    lines+=["",
    "- CP8EP8 uses HybridEP. HybridEP EP1 failed at startup with a cooperative-launch-too-large CUDA error; failure log is retained. CP8EP1 and CP1EP1 use the existing alltoall dispatcher's EP1 local path. This is therefore not a dispatcher-controlled communication-only ablation.",
    "- EP1 removes inter-GPU expert dispatch/combine transfer, not local expert sorting, permutation, gathering/scattering, or grouped GEMM overhead.",
    "- CP1 removes context-parallel collectives. It also puts all 262144 tokens on one GPU instead of 32768 tokens per GPU, so raw layer latency is not directly comparable as a communication-only delta.",
    "- CP8 with TP1 uses CP collectives for attention; other NCCL calls can include gradient synchronization and request metadata. Attribution SQL and raw kernel lists are retained.",
    "- No SM/tensor-core saturation claim is made from Kineto; hardware counters were not collected.",
    "",
    "## Confirmed CP1 indexer slow path","",
    "The CP1 runtime trace contains 16,384 calls to cutlass_80_simt_sgemm_128x64_8x5_nn_align1, totaling 25.596 seconds, and no cuDNN indexer-forward kernels. The three dominant FP32 elementwise kernels add 11.819 + 7.674 + 5.118 seconds of kernel duration. This is an unfused FP32 SIMT score-computation path, not merely eight times more BF16 fused work.",
    "The count matches two indexer blocks × 256 score chunks × 32 heads. Source: dsa_cudnn_kernels.py _indexer_topk_from_score_chunks selects _compute_indexer_scores_chunk_with_global_rows, which loops over heads using FP32 bmm, ReLU, weighting and accumulation. The packed-CP path instead supplies bottom_right_key_start and invokes the cuDNN indexer with causal offsets. Sparse attention itself still uses fused sparse kernels.",
    "CP1 has zero NCCL kernels in this capture, so context/expert inter-GPU communication does disappear. The large slowdown is dominated by the changed indexer execution path. The 666.5-second full-model block estimate describes this current slow path; it is not an estimate of an optimized CP1 implementation, nor a feasible full-model EP1 deployment on one B300.",
    "Source locations in the pinned Megatron-Core tree: transformer/experimental_attention_variant/dsa_cudnn_kernels.py:537 (FP32 head loop), :645 (cuDNN vs fallback branch), :885 (packed-CP causal-offset route), :1186/:1232 (generic chunked entry points). No fix was applied.",
    "",
    "## Files","",
    "- Config: ../../configs/glm53-debug-1d2m-config.json",
    "- Per topology: benchmark.json, summary.json, layer_timings/rank*.jsonl, runtime/*.pt.trace.json, memory/memory.rank0.pickle, analysis/*.csv and analysis/summary.json.",
    "- Artifacts were copied locally, hashed, and parsed. Original tools/profile_driver.py and tools/mfu.py were not edited for this experiment.",
    ""]
    for label in LABELS:
        runtime=next((ROOT/label/"runtime").glob("*.pt.trace.json"))
        memory=next((ROOT/label/"memory").glob("*.pickle"))
        lines.append(f"- {label}: [Kineto trace]({runtime}) · [Memory snapshot]({memory}) · [Detailed statistics]({ROOT/label/'summary.json'})")
    lines.append("")
    (ROOT/"RESULTS.md").write_text("\n".join(lines))
    (ROOT/"comparison.json").write_text(json.dumps({"benchmarks":results,"trace_summaries":{k:{x:v for x,v in t.items() if x not in ["kernels","block_attribution"]} for k,t in traces.items()}},indent=2)+"\n")
    print("\n".join(lines[:16]))
if __name__=="__main__": main()
