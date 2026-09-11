"""Measure observed GPU interval overlap, not counterfactual communication cost.

Read-only with respect to traces/training. Writes derived analysis beside the
existing reports. Collective ownership comes from Kineto process-group metadata.
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


SQL = """
WITH comms AS MATERIALIZED (
 SELECT CAST(EXTRACT_ARG(arg_set_id,'args.External id') AS INT) external_id,
 EXTRACT_ARG(arg_set_id,'args.Process Group Description') pg,
 EXTRACT_ARG(arg_set_id,'args.Collective name') collective
 FROM slice WHERE name='record_param_comms'
)
SELECT k.id,k.ts,k.dur,
 CASE WHEN k.category!='kernel' THEN 'memory'
      WHEN LOWER(k.name) LIKE '%nccl%' THEN 'comm' ELSE 'compute' END kind,
 CASE WHEN LOWER(k.name) LIKE '%nvjet%' OR LOWER(k.name) LIKE '%gemm%'
      OR LOWER(k.name) LIKE '%cublas%' THEN 1 ELSE 0 END gemm,
 COALESCE(c.pg,'') pg,COALESCE(c.collective,'') collective,'' marker
FROM slice k LEFT JOIN comms c
 ON k.category='kernel' AND LOWER(k.name) LIKE '%nccl%'
 AND CAST(EXTRACT_ARG(k.arg_set_id,'args.External id') AS INT)=c.external_id
WHERE k.category IN ('kernel','gpu_memcpy','gpu_memset') AND k.dur>0
UNION ALL
SELECT id,ts,dur,'marker',0,'','',name FROM slice
WHERE category='gpu_user_annotation' AND dur>0
AND (name='CustomFSDP.forward' OR name='ProfilerStep#0'
     OR name GLOB 'block_*/backward_including_recompute')
ORDER BY ts,id
"""


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersect(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def seconds(intervals):
    return sum(end - start for start, end in intervals) / 1e9


def metrics(intervals, compute, work, gemms):
    united = merge(intervals)
    duration = seconds(united)
    overlap = seconds(intersect(united, compute))
    work_overlap = seconds(intersect(united, work))
    return {
        'input_interval_count': len(intervals),
        'kernel_sum_s': seconds(intervals),
        'interval_union_s': duration,
        'overlap_noncomm_kernels_s': overlap,
        'no_noncomm_kernel_s': duration - overlap,
        'overlap_noncomm_kernels_pct': 100 * overlap / duration if duration else None,
        'overlap_noncomm_kernels_or_copies_s': work_overlap,
        'no_noncomm_kernel_or_copy_s': duration - work_overlap,
        'overlap_gemm_s': seconds(intersect(united, gemms)),
    }


def analyze(csv_path):
    groups = defaultdict(list)
    markers = defaultdict(list)
    seen = set()
    for row in csv.DictReader(csv_path.open()):
        identity = int(row['id'])
        if identity in seen:
            raise ValueError(f'Duplicate slice after collective join: {identity}')
        seen.add(identity)
        interval = (int(row['ts']), int(row['ts']) + int(row['dur']))
        if row['kind'] == 'marker':
            markers[row['marker']].append(interval)
            continue
        groups['all_gpu'].append(interval)
        groups[row['kind']].append(interval)
        if row['kind'] == 'compute' and row['gemm'] == '1':
            groups['gemm'].append(interval)
        if row['kind'] != 'comm':
            continue
        pg, collective = row['pg'], row['collective']
        groups[f'pg: {pg} / {collective}'].append(interval)
        if pg == 'EXPERT_DATA_PARALLEL_GROUP' and 'allgather' in collective:
            groups['fsdp_expert_weight_ag'].append(interval)
            groups['fsdp_weight_ag'].append(interval)
        elif pg == 'DATA_PARALLEL_GROUP_WITH_CP' and 'allgather' in collective:
            groups['fsdp_nonexpert_weight_ag'].append(interval)
            groups['fsdp_weight_ag'].append(interval)
        elif pg == 'DATA_PARALLEL_GROUP_WITH_CP' and 'reduce_scatter' in collective:
            groups['fsdp_grad_reduce'].append(interval)
        elif pg == 'CONTEXT_PARALLEL_GROUP':
            groups['cp'].append(interval)
        elif not pg:
            groups['unattributed_comm'].append(interval)
    compute = merge(groups['compute'])
    work = merge(groups['compute'] + groups['memory'])
    gemms = merge(groups['gemm'])
    all_gpu = merge(groups['all_gpu'])
    window = [(all_gpu[0][0], all_gpu[-1][1])]
    result = {
        'gpu_span_s': seconds(window),
        'gpu_work_union_s': seconds(all_gpu),
        'no_gpu_activity_s': seconds(window) - seconds(all_gpu),
        'compute_union_s': seconds(compute),
        'gemm_union_s': seconds(gemms),
        'communication': {
            key: metrics(value, compute, work, gemms)
            for key, value in groups.items()
            if key not in ('all_gpu', 'compute', 'memory', 'gemm')
        },
    }
    first_gather = min(groups['fsdp_weight_ag'])
    result['first_weight_gather'] = {
        'start_ns': first_gather[0],
        'end_ns': first_gather[1],
        'metrics': metrics([first_gather], compute, work, gemms),
    }
    after_first = [(first_gather[1], window[0][1])]
    result['after_first_weight_gather'] = metrics(
        intersect(merge(groups['fsdp_weight_ag']), after_first), compute, work, gemms
    )
    phase_windows = {'forward': merge(markers['CustomFSDP.forward'])}
    backward = [v for key, values in markers.items() if key.startswith('block_') for v in values]
    if backward:
        phase_windows['backward_blocks_including_recompute'] = [
            (min(a for a, _ in backward), max(b for _, b in backward))
        ]
    result['phase_windows'] = {
        phase: {
            'span_s': seconds(region),
            'fsdp_weight_ag': metrics(intersect(merge(groups['fsdp_weight_ag']), region), compute, work, gemms),
            'fsdp_expert_weight_ag': metrics(intersect(merge(groups['fsdp_expert_weight_ag']), region), compute, work, gemms),
        }
        for phase, region in phase_windows.items()
    }
    # Independent union identity validates the main overlap calculation.
    comm = merge(groups['comm'])
    check = seconds(comm) + seconds(compute) - seconds(merge(comm + compute))
    assert abs(check - result['communication']['comm']['overlap_noncomm_kernels_s']) < 1e-7
    assert not groups.get('unattributed_comm'), 'Unclassified NCCL kernels need investigation'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--ranks', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--trace-processor', default='/Users/jackrao/bin/trace_processor')
    parser.add_argument('--reuse-extracted', action='store_true')
    args = parser.parse_args()
    output = args.folder / 'analysis' / 'overlap'
    output.mkdir(exist_ok=True)
    (output / 'extract.sql').write_text(SQL + ';\n')
    ranks = {}
    for rank in map(int, args.ranks.split(',')):
        traces = list((args.folder / 'runtime').glob(f'rank{rank}.*.pt.trace.json'))
        assert len(traces) == 1, traces
        trace = traces[0]
        extracted = output / f'rank{rank}-intervals.csv'
        if args.reuse_extracted:
            previous = json.loads((output / f'rank{rank}.json').read_text())
            with trace.open('rb') as source:
                assert hashlib.file_digest(source, 'sha256').hexdigest() == previous['trace_sha256']
            assert extracted.stat().st_mtime_ns >= trace.stat().st_mtime_ns
        else:
            with extracted.open('w') as stdout, (output / f'rank{rank}-extract.log').open('w') as stderr:
                subprocess.run([args.trace_processor, 'query', str(trace), SQL], stdout=stdout, stderr=stderr, check=True)
        result = analyze(extracted)
        result['trace_file'] = str(trace)
        with trace.open('rb') as source:
            result['trace_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
        ranks[str(rank)] = result
        (output / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'rank': rank, 'gpu_span_s': result['gpu_span_s'], 'weights': result['communication']['fsdp_weight_ag']}), flush=True)
    summaries = {}
    for group in ('fsdp_weight_ag', 'fsdp_expert_weight_ag', 'fsdp_nonexpert_weight_ag', 'cp', 'comm'):
        summaries[group] = {}
        for field in ('interval_union_s', 'overlap_noncomm_kernels_s', 'no_noncomm_kernel_s', 'overlap_noncomm_kernels_pct', 'no_noncomm_kernel_or_copy_s'):
            values = [r['communication'][group][field] for r in ranks.values()]
            summaries[group][field] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
    report = {
        'definitions': {
            'compute': 'Union of all non-NCCL GPU kernels, including elementwise/copy kernels; excludes GPU memcpy/memset activities.',
            'overlap': 'Intersection of communication interval union and compute interval union on the same rank.',
            'no_noncomm_kernel': 'Communication executing without a concurrent non-NCCL kernel. NOT a proven counterfactual step-time saving.',
            'caveats': 'One profiled step per rank; not unprofiled control timing. Kernel time includes peer waits. Overlap can contend for resources. Groups/phases are not necessarily additive.',
        },
        'rank_summary': summaries,
        'ranks': ranks,
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
