"""Reusable all-rank Perfetto rollups for host GC, allocation APIs, and GPU kernels."""
import argparse
import csv
import io
import json
import re
import subprocess
from pathlib import Path

QUERIES={
    'host':"SELECT category,name,COUNT(*) calls,SUM(dur)/1e6 sum_ms,MAX(dur)/1e6 max_ms FROM slice WHERE dur>0 AND (name GLOB 'python_gc/*' OR name GLOB '*cudaMalloc*' OR name GLOB '*cudaFree*' OR name GLOB '*cuMem*' OR name GLOB '*cudaEventSynchronize*' OR name GLOB '*cudaStreamSynchronize*') GROUP BY category,name ORDER BY sum_ms DESC",
    'kernels':"SELECT name,COUNT(*) calls,SUM(dur)/1e6 sum_ms,MAX(dur)/1e6 max_ms FROM slice WHERE category='kernel' AND dur>0 GROUP BY name ORDER BY sum_ms DESC",
    'sublayers':"SELECT category,name,ts/1e3 ts_us,dur/1e6 ms FROM slice WHERE category='user_annotation' AND (name GLOB 'layer_*' OR name GLOB 'python_gc/*') ORDER BY ts",
}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args()
    out=a.folder/'analysis';out.mkdir(exist_ok=True);summary={}
    for trace in sorted((a.folder/'runtime').glob('*.json')):
        match=re.search(r'rank(\d+)',trace.name);rank=match.group(1) if match else '0'
        result={}
        for label,sql in QUERIES.items():
            (out/f'{label}.sql').write_text(sql+';\n')
            p=subprocess.run(['/Users/jackrao/bin/trace_processor','query',str(trace),sql],capture_output=True,text=True,check=True)
            (out/f'rank{rank}-{label}.csv').write_text(p.stdout)
            (out/f'rank{rank}-{label}.log').write_text(p.stderr)
            result[label]=list(csv.DictReader(io.StringIO(p.stdout)))
        summary[rank]=result
        print('rank',rank,json.dumps(result['host']),flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
