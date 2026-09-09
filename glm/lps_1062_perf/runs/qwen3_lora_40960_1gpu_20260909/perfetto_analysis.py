"""Reproducible Perfetto SQL rollups for the two final Kineto captures."""
import csv
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "perfetto_analysis"
QUERIES = {
    "kernels": """SELECT name, COUNT(*) calls, SUM(dur)/1e6 ms,
        AVG(dur)/1e3 avg_us, MIN(dur)/1e3 min_us, MAX(dur)/1e3 max_us
        FROM slice WHERE category='kernel' AND dur>0 GROUP BY name ORDER BY ms DESC""",
    "cpu": """SELECT category,name,COUNT(*) calls,SUM(dur)/1e6 inclusive_ms,
        MAX(dur)/1e6 max_ms FROM slice
        WHERE category IN ('cpu_op','cuda_runtime','user_annotation') AND dur>0
        GROUP BY category,name ORDER BY inclusive_ms DESC LIMIT 100""",
    "busy": """WITH k AS (SELECT ts,ts+dur en,dur FROM slice
          WHERE category='kernel' AND dur>0),
        ordered AS (SELECT *,MAX(en) OVER (ORDER BY ts,en ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_en FROM k)
        SELECT COUNT(*) kernels,(MAX(en)-MIN(ts))/1e6 span_ms,SUM(dur)/1e6 sum_ms,
        SUM(CASE WHEN prev_en IS NULL THEN en-ts WHEN en>prev_en THEN en-MAX(ts,prev_en) ELSE 0 END)/1e6 union_busy_ms,
        SUM(CASE WHEN ts>prev_en THEN ts-prev_en ELSE 0 END)/1e6 gaps_ms,
        MAX(CASE WHEN ts>prev_en THEN ts-prev_en ELSE 0 END)/1e6 max_gap_ms FROM ordered""",
    "gaps": """WITH k AS (SELECT ts,dur,name,MAX(ts+dur) OVER
        (ORDER BY ts,ts+dur ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_en
        FROM slice WHERE category='kernel' AND dur>0)
        SELECT ts, (ts-prev_en)/1e6 gap_ms,name next_kernel FROM k
        WHERE ts>prev_en ORDER BY gap_ms DESC LIMIT 30""",
    "gpu_owners": """WITH owners AS MATERIALIZED
        (SELECT id,name,EXTRACT_ARG(arg_set_id,'args.External id') ext FROM slice WHERE category='cpu_op')
        SELECT o.name,COUNT(*) kernels,SUM(k.dur)/1e6 gpu_ms FROM slice k
        JOIN owners o ON o.ext=EXTRACT_ARG(k.arg_set_id,'args.External id')
        WHERE k.category='kernel' GROUP BY o.name ORDER BY gpu_ms DESC LIMIT 70""",
    "kernel_tracks": """SELECT track_id,COUNT(*) kernels,SUM(dur)/1e6 ms
        FROM slice WHERE category='kernel' GROUP BY track_id ORDER BY ms DESC""",
    "attention": """SELECT name,COUNT(*) calls,SUM(dur)/1e6 ms,
        AVG(EXTRACT_ARG(arg_set_id,'args.est. achieved occupancy %')) estimated_occupancy,
        AVG(EXTRACT_ARG(arg_set_id,'args.blocks per SM')) blocks_per_sm
        FROM slice WHERE category='kernel' AND name LIKE '%sdpa%'
        GROUP BY name ORDER BY ms DESC""",
}


def category(name):
    lower = name.lower()
    if 'sdpa' in lower or 'fmha' in lower or 'flash_attn' in lower:
        return 'attention'
    if 'nccl' in lower:
        return 'nccl'
    if any(x in lower for x in ('nvjet','gemm','cublas','cutlass','grouped_mm')):
        return 'gemm'
    if any(x in lower for x in ('permute','sort_chunk','chunk_sort','topk','routing')):
        return 'moe_routing_permutation'
    if 'catarray' in lower:
        return 'concatenation'
    if any(x in lower for x in ('copy','memcpy')):
        return 'copy_conversion'
    if 'norm' in lower:
        return 'normalization'
    if any(x in lower for x in ('cross_entropy','softmax')):
        return 'softmax_cross_entropy'
    if any(x in lower for x in ('elementwise','fused_','neg_kernel','mul','add','silu')):
        return 'elementwise_fusions'
    return 'other'


def main():
    OUT.mkdir(exist_ok=True)
    all_results = {}
    for model in ('06b','30b-a3b'):
        result = {}
        trace = ROOT / f'qwen3-{model}.pt.trace.json'
        for key, sql in QUERIES.items():
            (OUT/f'{key}.sql').write_text(sql+';\n')
            p = subprocess.run(['trace_processor','query',str(trace),sql],
                               capture_output=True,text=True,check=True)
            (OUT/f'{model}-{key}.csv').write_text(p.stdout)
            (OUT/f'{model}-{key}.log').write_text(p.stderr)
            result[key] = list(csv.DictReader(io.StringIO(p.stdout)))
        categories = {}
        for row in result['kernels']:
            group = category(row['name'])
            value = categories.setdefault(group, {'ms':0.,'calls':0})
            value['ms'] += float(row['ms'])
            value['calls'] += int(row['calls'])
        result['categories'] = categories
        all_results[model] = result
        print(model,json.dumps({'busy':result['busy'],'categories':categories,
                               'owners':result['gpu_owners'][:15]},indent=2))
    (OUT/'results.json').write_text(json.dumps(all_results,indent=2)+'\n')


if __name__=='__main__':
    main()
