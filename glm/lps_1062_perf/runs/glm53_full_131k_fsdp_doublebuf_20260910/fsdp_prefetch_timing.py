"""Inspect expert all-gather scheduling against preceding-block GPU work.

This is a timeline diagnostic, not a causal counterfactual or a training change.
"""

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

from fsdp_overlap import intersect, merge, seconds


SQL = """
SELECT name,MIN(ts) ts,MAX(ts+dur) en FROM slice
WHERE category='gpu_user_annotation' AND dur>0
AND (name GLOB 'block_*/forward' OR name GLOB 'block_*/backward_including_recompute'
 OR name GLOB 'layer_*/forward/*' OR name GLOB 'layer_*/recompute/*')
GROUP BY name ORDER BY ts
"""


def subtract(left, right):
    result = []
    j = 0
    for start, end in left:
        while j < len(right) and right[j][1] <= start:
            j += 1
        position = start
        k = j
        while k < len(right) and right[k][0] < end:
            a, b = right[k]
            if a > position:
                result.append((position, min(a, end)))
            position = max(position, b)
            if position >= end:
                break
            k += 1
        if position < end:
            result.append((position, end))
    return result


def describe(values):
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'n': len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', type=Path)
    ap.add_argument('--rank', type=int, default=0)
    args = ap.parse_args()
    out = args.folder / 'analysis' / 'overlap'
    trace, = (args.folder / 'runtime').glob(f'rank{args.rank}.*.pt.trace.json')
    (out / 'prefetch-markers.sql').write_text(SQL + ';\n')
    markers_path = out / f'rank{args.rank}-prefetch-markers.csv'
    with markers_path.open('w') as stdout, (out / f'rank{args.rank}-prefetch-markers.log').open('w') as stderr:
        subprocess.run(['/Users/jackrao/bin/trace_processor', 'query', str(trace), SQL],
                       stdout=stdout, stderr=stderr, check=True)
    markers = {r['name']: (int(r['ts']), int(r['en']))
               for r in csv.DictReader(markers_path.open())}
    rows = list(csv.DictReader((out / f'rank{args.rank}-intervals.csv').open()))
    interval = lambda r: (int(r['ts']), int(r['ts']) + int(r['dur']))
    compute = merge([interval(r) for r in rows if r['kind'] == 'compute'])
    experts = sorted([interval(r) for r in rows if r['kind'] == 'comm'
                      and r['pg'] == 'EXPERT_DATA_PARALLEL_GROUP'
                      and 'allgather' in r['collective']])
    cp = merge([interval(r) for r in rows if r['kind'] == 'comm'
                and r['pg'] == 'CONTEXT_PARALLEL_GROUP'])
    assert len(experts) == 150
    origin = min(interval(r)[0] for r in rows if r['kind'] != 'marker')
    detail = []
    # This run has 3 dense then 75 MoE blocks, sequential forward/reverse backward.
    # Verify the first 75 gathers precede the end of the last forward block.
    last_forward_end = markers['layer_77/forward/mlp'][1]
    assert experts[74][1] <= last_forward_end
    assert experts[75][0] >= last_forward_end
    for index, (start, end) in enumerate(experts):
        phase = 'forward' if index < 75 else 'backward'
        layer = index + 3 if index < 75 else 77 - (index - 75)
        predecessor = layer - 1 if phase == 'forward' else layer + 1
        predecessor_key = (f'layer_{predecessor}/forward/mlp' if phase == 'forward'
                           else f'block_{predecessor}/backward_including_recompute')
        previous_end = markers.get(predecessor_key, (None, None))[1]
        visible_compute = seconds(intersect([(start, end)], compute))
        detail.append({
            'phase': phase, 'target_layer_inferred_from_order': layer,
            'start_s': (start - origin) / 1e9, 'end_s': (end - origin) / 1e9,
            'duration_ms': (end - start) / 1e6,
            'concurrent_compute_ms': visible_compute * 1000,
            'no_concurrent_compute_ms': (end - start) / 1e6 - visible_compute * 1000,
            'gather_end_minus_preceding_block_end_ms':
                (end - previous_end) / 1e6 if previous_end is not None else None,
            'gather_start_minus_preceding_block_end_ms':
                (start - previous_end) / 1e6 if previous_end is not None else None,
        })
    uncovered = subtract(merge(experts), compute)
    summary = {
        'rank': args.rank,
        'expert_gather_ms': describe([r['duration_ms'] for r in detail]),
        'nonoverlap_s': seconds(uncovered),
        'nonoverlap_concurrent_cp_s': seconds(intersect(uncovered, cp)),
        'uncovered_intervals': len(uncovered),
        'uncovered_time_in_gaps_up_to_us': {
            str(limit): seconds([(a, b) for a, b in uncovered if b - a <= limit * 1000])
            for limit in (10, 50, 100, 500, 1000)
        },
        'phases': {},
        'gathers': detail,
    }
    for phase in ('forward', 'backward'):
        calls = [r for r in detail if r['phase'] == phase]
        valid = [r for r in calls if r['gather_end_minus_preceding_block_end_ms'] is not None]
        summary['phases'][phase] = {
            'duration_ms': describe([r['duration_ms'] for r in calls]),
            'no_compute_ms': sum(r['no_concurrent_compute_ms'] for r in calls),
            'gathers_ready_by_preceding_block_end': sum(r['gather_end_minus_preceding_block_end_ms'] <= 0 for r in valid),
            'comparable_gathers': len(valid),
            'late_gather_tail_ms': sum(max(0, r['gather_end_minus_preceding_block_end_ms']) for r in valid),
            'ready_margin_before_preceding_block_end_ms': describe([
                -r['gather_end_minus_preceding_block_end_ms'] for r in valid
            ]),
            'caveat': 'Preceding-block GPU annotation endpoint is a proxy deadline, not proof all other prerequisites were ready.',
        }
    (out / f'rank{args.rank}-prefetch.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'gathers'}, indent=2))


if __name__ == '__main__':
    main()
