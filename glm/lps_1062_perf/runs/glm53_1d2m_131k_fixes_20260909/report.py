"""Render only accepted result attempts, with explicit protocol and caveats."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TOPOLOGIES=['cp8ep8','cp8ep1','cp1ep1']
CATEGORIES=['Dense + indexer','MoE + shared indices','MoE + indexer']
def main():
    results={t:json.loads((ROOT/t/'result/summary.json').read_text()) for t in TOPOLOGIES}
    lines=['# GLM-5.3 1d2m: 131072-token results with fixes','',
    'Tracking PR: [trainers #1355](https://github.com/basetenlabs/trainers/pull/1355), stacked on #1157. Dependencies: [Megatron-Core #76](https://github.com/basetenlabs/Megatron-LM/pull/76) and [Bridge #84](https://github.com/basetenlabs/Megatron-Bridge/pull/84).',
    '', '## Configuration and protocol','',
    'Real GLM-5.3 blocks 0, 3, 6 remapped to dense/indexer, MoE/shared, MoE/indexer. LoRA rank/alpha 32, TP1/PP1/ETP1, full one-block recompute, native-FP8 expert storage. All three use BT_FREEZE_GC_AFTER_WARMUP=1. EP1 automatically selects local dispatch; EP8 uses HybridEP. Tested source c9a723bf431621ca05580726f4acd01ec618326a, Bridge 2076a9e81ae36e7890d017f4af8d7c707c2d633b, Core 3b893e39e (full pins available from dependency commits).',
    'Three complete warmup FB+optimizer iterations, five unprofiled controls, one memory step, one all-rank runtime step. Headline timings retain all five controls and exclude optimizer time. CUDA-event instrumentation remains enabled in controls, with no synchronization between sublayers. One-time GC collection and its rank barrier are included in the warmup optimizer time.',
    '', '## Five-control measurements','',
    '| Topology | GPUs | FB mean ± SD, s | Tokens/s total | Tokens/s/GPU | Peak allocated GiB | Peak reserved GiB |',
    '|---|---:|---:|---:|---:|---:|---:|']
    for t,r in results.items():
        lines.append(f"| {t} | {1 if t=='cp1ep1' else 8} | {r['fb_s']['mean']:.4f} ± {r['fb_s']['sd']:.4f} | {r['tps']:,.0f} | {r['tps_per_gpu']:,.0f} | {r['peak_allocated_gib']:.3f} | {r['peak_reserved_gib']:.3f} |")
    lines+=['','Peaks are the distributed maximum of PyTorch allocator peaks across controls, not all physical HBM usage. TPS/GPU is efficiency, not single-request latency. These are three-block proxy TPS values, not full-model throughput.',
    '', '## Separate attention / MLP measurements','',
    'GPU elapsed milliseconds, rank 0, mean ± sample SD over five controls. Attention includes its normalization/residual path and any indexer computation. MLP includes routed/shared experts, dispatch/combine, normalization and residual processing. Backward excludes forward recomputation; tensor-gradient boundary hooks partition the backward path. Full block and all-rank raw timings remain available.',
    '', '| Topology | Block | Fwd attn | Fwd MLP | Recompute attn | Recompute MLP | Bwd attn | Bwd MLP |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
    fields=['forward_attention_ms','forward_mlp_ms','recompute_attention_ms','recompute_mlp_ms','attention_backward_ms','mlp_backward_ms']
    for t,r in results.items():
        for layer,category in enumerate(CATEGORIES):
            row=r['layers'][str(layer)]
            lines.append('| '+' | '.join([t,category]+[f"{row[k]['mean']:.2f} ± {row[k]['sd']:.2f}" for k in fields])+' |')
    lines+=['','## Weighted full-model block estimate','',
    'Weight by 3 dense/indexer + 57 MoE/shared + 18 MoE/indexer blocks. This excludes embedding/head/loss, optimizer, and inter-block overhead. It is not a claim that the full model fits EP1; depth-dependent routing/cache lifetimes and memory pressure limit extrapolation.',
    '', '| Topology | Forward s | Recompute s | Actual backward s | Total block FB s |',
    '|---|---:|---:|---:|---:|']
    for t,r in results.items():
        f=r['full_blocks_seconds'];total=f['forward_ms']+f['backward_including_recompute_ms']
        lines.append(f"| {t} | {f['forward_ms']:.3f} | {f['recompute_ms']:.3f} | {f['backward_excluding_recompute_ms']:.3f} | {total:.3f} |")
    lines+=['','## Diagnosis and fixes','',
    '1. **CP1 indexer wiring:** plain-causal score chunks now call the existing cuDNN scorer with global query offsets. The old per-head FP32 bmm fallback was unnecessary for this case. Bounds/key-count probes passed; selected-key overlap was 1.0 at 2k/4k and 0.99999994 at 8k.',
    '2. **Singleton HybridEP:** no network transfer does not mean no dispatch kernel. Its cooperative dispatch failed with 256 local experts. The singleton case now takes the existing local alltoall permutation path, preserving non-overlapped shared-expert scheduling.',
    '3. **Host GC stragglers:** the 131k baseline recorded 598–636 ms generation-2 pauses in slow controls, and the all-rank trace showed one rank collecting for 591 ms while peers waited. No CUDA allocation/free driver activity accompanied that captured stall. The experimental GC policy freezes warmed long-lived objects but leaves automatic GC enabled for new objects, then unfreezes before resource teardown.',
    'GC policy is default-off and remains experimental pending longer process-lifetime/full-model validation. No tests were added or modified. Three-warmup protocol and cached diagnostic counters avoid accepting cold or observer-contaminated measurements. Earlier exploratory attempts are retained but are not in this table.',
    '', '## Additional stability checks','']
    for t in TOPOLOGIES:
        p=ROOT/t/'stability.json'
        if not p.exists():continue
        b=json.loads(p.read_text());w=[v for v in b['windows'] if v['phase']=='control'];times=[v['fb_elapsed_s'] for v in w]
        lines.append(f"- {t}: {len(w)} additional controls, FB {statistics.mean(times):.4f} ± {statistics.stdev(times):.4f} s, range {min(times):.4f}–{max(times):.4f} s; allocated peaks {min(v['peak_allocated_bytes'] for v in w)/2**30:.3f}–{max(v['peak_allocated_bytes'] for v in w)/2**30:.3f} GiB; reserved peaks {min(v['peak_reserved_bytes'] for v in w)/2**30:.3f}–{max(v['peak_reserved_bytes'] for v in w)/2**30:.3f} GiB.")
    lines+=['','## Traces and raw data','',
    'Runtime captures include profiler start/stop and may perturb the step, especially with all ranks recording. Their CPU API duration sums are not additive wall time. Use control CUDA-event timings for the attention/MLP comparison. Memory and runtime captures are separate steps. Trace and memory bytes were copied locally, hashed, and parsed.', '']
    for t in TOPOLOGIES:
        folder=ROOT/t/'result'
        lines.append(f"### {t}")
        lines.append('')
        lines.append(f"[Detailed statistics]({folder/'summary.json'}) · [Raw benchmark]({folder/'benchmark.json'})")
        lines.append('')
        for p in sorted((folder/'runtime').glob('*.json')):lines.append(f'- [Kineto: {p.name}]({p})')
        for p in sorted((folder/'memory').glob('*')):lines.append(f'- [Memory: {p.name}]({p})')
        lines.append('')
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    (ROOT/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print('\n'.join(lines[:17]))
if __name__=='__main__':main()
