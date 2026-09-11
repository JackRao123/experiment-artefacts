"""Reproducible control, sublayer and fixed-overhead accounting."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOPOLOGIES = ('cp1ep1','cp8ep1','cp8ep8')
PARTS = ('forward_attention_ms','forward_mlp_ms','recompute_attention_ms','recompute_mlp_ms','attention_backward_ms','mlp_backward_ms')

def stats(values):
    return dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else 0.,min=min(values),max=max(values),n=len(values))

data={}
for topology in TOPOLOGIES:
    folder=ROOT/topology
    if not (folder/'stability.json').exists(): continue
    headline=json.loads((folder/'result/summary.json').read_text())
    bench=json.loads((folder/'stability.json').read_text())
    windows=bench['windows']
    assert len(windows)==20 and all(w['phase']=='control' for w in windows)
    gpus=bench['num_gpus']
    steps=set(range(bench['initial_status']['step']+2,bench['final_status']['step']+2))
    rows=[json.loads(line) for p in (folder/'stability_timings').glob('rank*.jsonl') for line in p.read_text().splitlines()]
    rows=[r for r in rows if r['step'] in steps]
    assert len(rows)==20*3*gpus, (topology,len(rows))
    assert all(r['owns_gc_freeze'] and r['gc_enabled'] for r in rows)
    by_rank={str(rank):{str(layer):{part:stats([r[part] for r in rows if r['rank']==rank and r['layer']==layer]) for part in PARTS} for layer in range(3)} for rank in range(gpus)}
    fb=stats([w['fb_elapsed_s'] for w in windows])
    rpc=stats([w['fb_elapsed_s']+w['optim_elapsed_s'] for w in windows])
    server=stats([w['server_step_seconds'] for w in windows])
    extra=stats([w['fb_elapsed_s']+w['optim_elapsed_s']-w['server_step_seconds'] for w in windows])
    item=dict(headline=headline,stability=dict(fb_s=fb,http_fb_optim_s=rpc,backend_fb_optim_s=server,outside_backend_timer_s=extra,tps_per_gpu=131072/gpus/fb['mean']),rank_layers=by_rank)
    item['block_parts_ms']={part:sum(by_rank['0'][str(layer)][part]['mean'] for layer in range(3)) for part in PARTS}
    item['weighted_full_model_parts_s']={part:sum(by_rank['0'][str(layer)][part]['mean']*count for layer,count in enumerate((3,57,18)))/1000 for part in PARTS}
    data[topology]=item

(ROOT/'accounting.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# GLM-5.3 singleton-EP root cause: 131072-token BF16 1d2m','',
       'IN PROGRESS: FSDP and isolated GEMM follow-ups are separate experiments.','',
       'Same checkpoint values, LoRA rank/alpha32, full one-block recompute, TP1/PP1/ETP1. Core identity-sort elimination is the only model-code change from the previous BF16 comparison. No tests or original driver/MFU tools changed.','',
       '## Whole-request controls','',
       '| Topology | Five-control FB seconds ± SD | Five-control TPS/GPU | Additional 20 FB seconds ± SD | Additional 20 TPS/GPU | Peak allocated GiB |',
       '|---|---:|---:|---:|---:|---:|']
for t,d in data.items():
    h=d['headline'];s=d['stability'];f=h['fb_s'];g=s['fb_s']
    lines.append(f"| {t} | {f['mean']:.4f} ± {f['sd']:.4f} | {h['tps_per_gpu']:,.0f} | {g['mean']:.4f} ± {g['sd']:.4f} | {s['tps_per_gpu']:,.0f} | {h['peak_allocated_gib']:.3f} |")
lines += ['', 'Do not substitute profiler durations for controls. The five-control and 20-control populations are separate and retained in full. TPS uses HTTP forward/backward time, excluding optimizer, as in all previous runs.','',
          '## Timing boundary decomposition (20 controls)','',
          'The backend metric includes FB + optimizer. Compare it to HTTP FB + optimizer, not HTTP FB alone. The difference includes serialization, dispatch, return values, polling, and instrumentation outside the backend timer; it is not a direct measurement of network time.','',
          '| Topology | HTTP FB+optim ms | Backend FB+optim ms | Outside backend ms |',
          '|---|---:|---:|---:|']
for t,d in data.items():
    s=d['stability'];lines.append('| '+t+' | '+' | '.join(f"{1000*s[k]['mean']:.2f}" for k in ('http_fb_optim_s','backend_fb_optim_s','outside_backend_timer_s'))+' |')
lines += ['', '## Attention / MLP (20 controls, rank0 mean milliseconds)','',
          '| Topology | Layer type | Fwd attn | Fwd MLP | RC attn | RC MLP | Bwd attn interval | Bwd MLP interval |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
for t,d in data.items():
    for layer,label in enumerate(('Dense/indexer','MoE/shared','MoE/indexer')):
        x=d['rank_layers']['0'][str(layer)]
        lines.append('| '+t+' | '+label+' | '+' | '.join(f"{x[p]['mean']:.2f}" for p in PARTS)+' |')
lines += ['', 'Backward intervals partition the gradient dependency path and include waits. They are not exclusive ownership of backward kernels. accounting.json retains all-rank distributions, not just rank0.','',
          '## Evidence','',
          '- Identity sorts are unnecessary only when EP=ETP=1. The ordinary token permutation remains. identity-sort-probe.log verifies bitwise output/probability/gradient equivalence. New EP1 traces must contain zero _sort_chunks_by_map_kernel launches.','- Runtime traces and memory snapshots live under each topology/result/runtime and topology/result/memory. Artifact sizes and SHA256 digests are in result/summary.json.','- The old BF16 EP8 trace has approximately 20.5 ms summed dispatch/combine kernel duration for a complete step. This is a few percent of a ~0.6-second request, not most of its time. Kernel overlap and peer waits mean the sum is not an exact hypothetical wall-time saving.','- EP1 still performs routing, local permutation and expert GEMMs. It has 256 resident experts per GPU instead of 32. At balanced routing, CP8EP1 has 512 rows/expert versus CP8EP8 4096. Isolated probes distinguish shape efficiency from communication.','- No hardware SM/tensor-core counters have been collected in these Kineto runs; occupancy or kernel duration alone is not a saturation measurement.','']
(ROOT/'RESULTS.md').write_text('\n'.join(lines))
if len(data)==3:
    cp1=data['cp1ep1']['stability'];cp8=data['cp8ep1']['stability']
    gap=cp8['http_fb_optim_s']['mean']-cp1['http_fb_optim_s']['mean']/8
    outside=cp8['outside_backend_timer_s']['mean']-cp1['outside_backend_timer_s']['mean']/8
    lines += ['## Why CP1 has higher TPS/GPU','',
              'Normalize the CP1 run by dividing its latency by eight: that is the same amount of token work per GPU as CP8. This is an efficiency comparison, not the actual request latency (CP1 is still much slower to finish one request).','',
              f"On matching FB+optimizer boundaries, normalized HTTP time is {cp1['http_fb_optim_s']['mean']*125:.2f} ms for CP1 versus {cp8['http_fb_optim_s']['mean']*1000:.2f} ms for CP8. The gap is {gap*1000:.2f} ms; {outside*1000:.2f} ms ({outside/gap:.1%}) lies outside the backend timer.",'',
              '| Normalized component, ms | CP1EP1 / 8 | CP8EP1 | CP8 penalty |',
              '|---|---:|---:|---:|']
    for part in PARTS:
        a=data['cp1ep1']['block_parts_ms'][part]/8;b=data['cp8ep1']['block_parts_ms'][part]
        lines.append(f'| {part} | {a:.2f} | {b:.2f} | {b-a:+.2f} |')
    a=(cp1['backend_fb_optim_s']['mean']*1000-sum(data['cp1ep1']['block_parts_ms'].values()))/8
    b=cp8['backend_fb_optim_s']['mean']*1000-sum(data['cp8ep1']['block_parts_ms'].values())
    lines += [f'| Backend remainder, including optimizer | {a:.2f} | {b:.2f} | {b-a:+.2f} |','',
              'The MLP penalty is consistent with more local experts, smaller GEMMs, repeated weight reads, and launch/scheduling overhead. The isolated GEMM probe quantifies only the base-GEMM shape effect; it does not measure hardware saturation or the actual unbalanced routing distribution.','',
              'The proxy has just three blocks but pays a complete LM head/loss and request-processing cost. Extrapolate block categories separately; do not extrapolate its raw TPS to a 78-block model.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({t:{'five_tps':d['headline']['tps_per_gpu'],'twenty':d['stability']} for t,d in data.items()},indent=2))
