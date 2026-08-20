import sys
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

print("=== autograd/backward node names containing AllToAll or Dispatcher or MoE ===")
print(q("""
SELECT DISTINCT name FROM slice
WHERE name LIKE '%llToAll%' OR name LIKE '%ispatcher%' OR name LIKE '%MoE%' OR name LIKE '%moe%'
LIMIT 40
""").to_string())

print("=== deep ancestors of nccl:all_to_all annotations (depth 2-5) ===")
print(q("""
SELECT gp.name L2, ggp.name L3, gggp.name L4, COUNT(*) c
FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='record_param_comms'
JOIN slice gp ON p.parent_id=gp.id
LEFT JOIN slice ggp ON gp.parent_id=ggp.id
LEFT JOIN slice gggp ON ggp.parent_id=gggp.id
WHERE s.name='nccl:all_to_all'
GROUP BY L2, L3, L4 ORDER BY c DESC LIMIT 25
""").to_string())

print("=== achieved BW distribution, BF16 token calls (nelems*2 bytes / dur) ===")
print(q("""
WITH sr AS (
  SELECT s.dur, CAST(ine.display_value AS DOUBLE)*2 AS bytes
  FROM slice s
  JOIN args dt ON dt.arg_set_id=s.arg_set_id AND dt.key='args.dtype' AND dt.display_value='BFloat16'
  JOIN args ine ON ine.arg_set_id=s.arg_set_id AND ine.key='args.In msg nelems'
  WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
)
SELECT COUNT(*) c,
  AVG(bytes/dur*1e3) avg_gbs,
  MIN(bytes/dur*1e3) min_gbs,
  MAX(bytes/dur*1e3) max_gbs
FROM sr
""").to_string())

print("=== BF16 duration percentiles ===")
print(q("""
SELECT s.dur/1e6 dur_ms, CAST(ine.display_value AS DOUBLE)*2/1e9 bytes_gb
FROM slice s
JOIN args dt ON dt.arg_set_id=s.arg_set_id AND dt.key='args.dtype' AND dt.display_value='BFloat16'
JOIN args ine ON ine.arg_set_id=s.arg_set_id AND ine.key='args.In msg nelems'
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
ORDER BY s.dur DESC LIMIT 20
""").to_string())
