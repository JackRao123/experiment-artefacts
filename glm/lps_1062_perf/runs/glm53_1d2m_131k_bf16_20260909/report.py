"""Report measured storage deltas and explain topology ordering without assuming it."""
import json
import math
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'glm53_1d2m_131k_fixes_20260909'
TOPOLOGIES=['cp8ep8','cp8ep1','cp1ep1']
FIELDS=['forward_attention_ms','forward_mlp_ms','recompute_attention_ms','recompute_mlp_ms','attention_backward_ms','mlp_backward_ms']
def read(p):return json.loads(p.read_text())
def main():
    new={t:read(ROOT/t/'result/summary.json') for t in TOPOLOGIES}
    old={t:read(OLD/t/'result/summary.json') for t in TOPOLOGIES}
    lines=['# BF16 expert storage: GLM-5.3 1d2m at 131072 tokens','',
    'Same source c9a723bf431621ca05580726f4acd01ec618326a, checkpoint values, LoRA rank/alpha32, full one-block recompute, TP1 PP1 ETP1, GC policy and instrumentation as the FP8-storage run. Only expert_weight_storage changes to bf16. Three full warmups, five unprofiled controls, one memory capture and one all-rank runtime capture. No source/test changes for this storage experiment.',
    '', '## Headline: five controls','',
    '| Topology | GPUs | BF16 FB mean ± SD (s) | BF16 TPS/GPU | FP8-storage TPS/GPU | Nominal change | BF16 peak allocated GiB |',
    '|---|---:|---:|---:|---:|---:|---:|']
    for t,r in new.items():
        gain=100*(r['tps_per_gpu']/old[t]['tps_per_gpu']-1)
        lines.append(f"| {t} | {1 if t=='cp1ep1' else 8} | {r['fb_s']['mean']:.4f} ± {r['fb_s']['sd']:.4f} | {r['tps_per_gpu']:,.0f} | {old[t]['tps_per_gpu']:,.0f} | {gain:+.2f}% | {r['peak_allocated_gib']:.3f} |")
    lines+=['','The five-control BF16 means match the proposed TPS/GPU ordering: CP1EP1 > CP8EP1 > CP8EP8. The middle pair is NOT robust: EP8 has one 0.743-second control, while three controls are near 0.603–0.609 seconds. Its 20 additional controls average 26,803 TPS/GPU, versus EP1\'s five-control 26,229. Do not treat this small, tuning-sensitive gap as a universal topology law or a statistically proven ordering.',
    'Storage deltas are observational, not pure dequantization speedups: independently autotuned permutation launch configurations differ across ranks and runs. All five controls are retained; supplemental controls are not substituted for the headline.',
    '', '## Control attention/MLP breakdown','',
    'GPU elapsed ms, rank0, mean over five controls. Forward/recompute include their complete sublayer paths. Backward numbers partition the gradient dependency path, including waits; they are not exclusive kernel ownership or pure communication times.',
    '', '| Topology | Block | Fwd attention | Fwd MLP | Recompute attention | Recompute MLP | Bwd attention interval | Bwd MLP interval |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
    for t,r in new.items():
        for l,name in enumerate(['Dense/indexer','MoE/shared','MoE/indexer']):
            lines.append('| '+' | '.join([t,name]+[f"{r['layers'][str(l)][k]['mean']:.2f}" for k in FIELDS])+' |')
    lines+=['','## Account for the whole step','',
    'Sums across all three blocks, ms per control. Residual = measured FB minus the six sublayer timing sums; includes embedding/head/loss, RPC and unassigned scheduling gaps. It is not exclusively CPU time.',
    '', '| Topology | Fwd attn | Fwd MLP | RC attn | RC MLP | Bwd attn | Bwd MLP | Residual |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t,r in new.items():
        totals=[sum(r['layers'][str(l)][k]['mean'] for l in range(3)) for k in FIELDS]
        residual=r['fb_s']['mean']*1000-sum(totals)
        lines.append('| '+' | '.join([t]+[f'{v:.2f}' for v in totals]+[f'{residual:.2f}'])+' |')
    lines+=['','## Why the intuitive ordering is not guaranteed','',
    'CP1 wins TPS/GPU here because its latency is less than eight times the CP8 latency. One GPU does all 131072 tokens; each CP8 GPU gets 16384. Removing CP also eliminates its collectives and amortizes fixed costs over more local work. CP1 remains much slower in request latency, despite better efficiency per GPU.',
    'EP1 removes cross-GPU dispatch/combine, but retains token routing, permutation and expert GEMMs. EP8 has 32 local experts versus EP1\'s 256. The theoretical balanced mean is 4096 routed tokens/expert at EP8 versus 512 at EP1. Actual routing is not force-balanced. Expert weight reuse and launch geometry change, so communication removal alone cannot establish the ordering.',
    '', '## A concrete remaining autotuning bug','',
    'Transformer Engine 2.16.0 common/triton/permutation.py decorates _sort_chunks_by_map_kernel with triton.autotune(key=["hidden_size"]). The key omits token count. Its choice is first made during the server\'s tiny 64-token startup warmup and can be reused for the 131072-row routed activation tensor. Different ranks select different cached BLOCK_SIZE values.',
    '', '| Same-width BF16 chunk copy | Rank 0 trace | Rank 6 trace |',
    '|---|---:|---:|',
    '| Grid | 131072 × 6 | 131072 × 96 |',
    '| BLOCK_SIZE inferred from H=6144 | 1024 | 64 |',
    '| Kernel time | 0.522 ms | 6.464 ms |',
    '', 'The slow configuration launches 16 times as many blocks and takes about 12.4 times as long. An isolated same-input microbenchmark directly confirms it: BLOCK_SIZE64 6.485 ms, 1024 0.512 ms, 4096 0.476 ms, with bitwise-identical output and permuted probabilities. This is a real configuration problem, not a fundamental EP1 limitation. chunk_sort_probe.py and its log are retained. The installed TE implementation was NOT modified for this comparison.',
    '', 'The MoE/shared recompute trace shows almost identical GEMM kernel sums on ranks0/6 (14.40/14.12 ms), but permutation/top-k sums of 2.66/10.46 ms. MLP elapsed is 12.70/20.69 ms. The gap is primarily in permutation, not a large GEMM regression between these ranks. Kernel sums overlap across streams and are not additive wall time.',
    'Five-control timing agrees: rank6 spends 20.49+20.04=40.53 ms in shared-MoE MLP recompute+backward versus rank0\'s 12.45+14.76=27.21 ms. Rank0 then spends 50.10 ms in the attention-backward interval versus rank6\'s 36.46 ms. A late rank\'s local work appears as waiting in its peers.',
    '', 'A proper next fix is to key the permutation tuner by workload size and relevant direction/probability/stride modes, or retune at the actual workload. Pinning a good configuration is a useful experiment, but is not yet an end-to-end fix in this run.',
    '', '## Communication is not larger at EP1','',
    'Kineto process-group metadata identifies the CP operations directly. Both CP8 runs have 11 CP AllGathers with 130088960 input / 1040711680 output bytes per rank, and three CP ReduceScatters with 452984832 input / 56623104 output bytes. These are logical tensor sizes, not measured wire bytes.',
    'The captured AllGather kernel sums are 21.55 ms (EP8) and 21.63 ms (EP1). ReduceScatter sums are 6.11 versus 30.06 ms despite identical sizes; those durations include peer waits. The 26,537,984-byte coalesced gradient allreduce is also the same in both. HybridEP additionally performs its own dispatch/combine and tiny metadata reductions at EP8.',
    '', '## What removing dequantization changed','',
    'Old FP8-storage EP1 issued 512 weight-dequant kernels per MoE forward (~6.2 ms summed GPU kernel time); BF16 issues zero. Control MoE-forward MLP times changed 33.34→16.90 ms (shared-index block) and 35.42→20.82 ms (indexer block). Do not attribute the entire difference to dequantization: the old rank0 chunk sort used a slower cached configuration too.',
    'Runtime inventories confirm every routed expert weight is an ordinary BF16 Parameter with no quantized payload. EP1 has 1024 weight tensors across the two MoE blocks; EP8 has 128 per GPU. Resident expert matrices total 36 GiB/GPU at EP1 versus 4.5 GiB/GPU at EP8. This is capacity footprint, not a DRAM-traffic measurement.',
    '', '## Weighted full-model block estimate','',
    '3 dense/indexer + 57 MoE/shared + 18 MoE/indexer. This excludes embedding/head/loss/optimizer overhead and is not a claim that the full model fits EP1.',
    '', '| Topology | Fwd s | Recompute s | Actual bwd s | Block total s |',
    '|---|---:|---:|---:|---:|']
    for t,r in new.items():
        f=r['full_blocks_seconds']
        lines.append(f"| {t} | {f['forward_ms']:.3f} | {f['recompute_ms']:.3f} | {f['backward_excluding_recompute_ms']:.3f} | {f['forward_ms']+f['backward_including_recompute_ms']:.3f} |")
    lines+=['','## Files and limitations','',
    'Only expert storage changed. No trainer code or tests changed. GC fix remains enabled and no old hundreds-of-ms GC pauses appeared in the five accepted controls. Profile captures are separate steps; their startup/shutdown and synchronization overhead are not headline measurements. GPU-scope tables use interval envelopes across streams to include TE GEMMs lacking CPU ownership links; boundary attribution is inferred. No hardware counters were collected.', '']
    for t in TOPOLOGIES:
        d=ROOT/t/'result';lines.append(f'### {t}');lines.append('')
        lines.append(f"[Statistics]({d/'summary.json'}) · [Benchmark]({d/'benchmark.json'}) · [GPU scope breakdown]({d/'analysis/gpu-scopes.json'})")
        lines.append('')
        for p in sorted((d/'runtime').glob('*.json')):lines.append(f'- [Kineto {p.name}]({p})')
        for p in sorted((d/'memory').glob('*.pickle')):lines.append(f'- [Memory {p.name}]({p})')
        lines.append('')
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    (ROOT/'comparison.json').write_text(json.dumps(dict(bf16=new,fp8_storage=old),indent=2)+'\n')
    print('\n'.join(lines[:17]))
if __name__=='__main__':main()
