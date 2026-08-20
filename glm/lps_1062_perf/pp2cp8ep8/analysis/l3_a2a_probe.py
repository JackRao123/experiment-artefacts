import sys
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

print("=== kernel names matching comm ===")
print(q("""
SELECT name, COUNT(*) c, SUM(dur)/1e9 s FROM slice
WHERE category='kernel' AND (name LIKE '%SendRecv%' OR name LIKE '%AllToAll%' OR name LIKE '%AllGather%' OR name LIKE '%ReduceScatter%' OR name LIKE '%AllReduce%')
GROUP BY name ORDER BY s DESC LIMIT 15
""").to_string())

print("=== args available on SendRecv kernels (sample) ===")
print(q("""
SELECT a.key, COUNT(*) c FROM slice s JOIN args a ON a.arg_set_id = s.arg_set_id
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%' GROUP BY a.key LIMIT 30
""").to_string())

print("=== SendRecv by dtype x stream ===")
print(q("""
SELECT dt.display_value dtype, st.display_value stream, COUNT(*) c, SUM(s.dur)/1e9 s,
       AVG(s.dur)/1e3 avg_ms, MAX(s.dur)/1e3 max_ms
FROM slice s
JOIN args dt ON dt.arg_set_id = s.arg_set_id AND dt.key = 'args.dtype'
JOIN args st ON st.arg_set_id = s.arg_set_id AND st.key = 'args.stream'
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
GROUP BY dtype, stream ORDER BY s DESC
""").to_string())

print("=== nelems stats by dtype ===")
print(q("""
SELECT dt.display_value dtype, COUNT(*) c,
       AVG(CAST(ine.display_value AS DOUBLE))/1e6 avg_melem,
       MIN(CAST(ine.display_value AS DOUBLE))/1e6 min_melem,
       MAX(CAST(ine.display_value AS DOUBLE))/1e6 max_melem
FROM slice s
JOIN args dt ON dt.arg_set_id = s.arg_set_id AND dt.key = 'args.dtype'
JOIN args ine ON ine.arg_set_id = s.arg_set_id AND ine.key = 'args.In msg nelems'
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
GROUP BY dtype
""").to_string())

print("=== nccl:all_to_all annotation parent chain sample ===")
print(q("""
SELECT p.name pn, gp.name gpn, COUNT(*) c FROM slice s
JOIN slice p ON s.parent_id=p.id
LEFT JOIN slice gp ON p.parent_id=gp.id
WHERE s.name='nccl:all_to_all'
GROUP BY pn, gpn ORDER BY c DESC LIMIT 15
""").to_string())

print("=== correlation arg on SendRecv kernels? ===")
print(q("""
SELECT a.key, a.display_value FROM slice s JOIN args a ON a.arg_set_id=s.arg_set_id
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%' AND a.key LIKE '%orrelation%' LIMIT 5
""").to_string())
