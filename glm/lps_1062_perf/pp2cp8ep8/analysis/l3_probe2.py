import sys
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

print("=== cudaMemcpyAsync under aten::copy_ : grandparent (copy_'s parent) ===")
print(q("""
SELECT gp.name gpn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='aten::copy_'
JOIN slice gp ON p.parent_id=gp.id
WHERE s.name='cudaMemcpyAsync'
GROUP BY gpn ORDER BY s DESC LIMIT 20
""").to_string())

print("=== copy_ memcpys: great-grandparent ===")
print(q("""
SELECT ggp.name ggpn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='aten::copy_'
JOIN slice gp ON p.parent_id=gp.id
JOIN slice ggp ON gp.parent_id=ggp.id
WHERE s.name='cudaMemcpyAsync'
GROUP BY ggpn ORDER BY s DESC LIMIT 20
""").to_string())

print("=== copy_ memcpys: does the chain hit CheckpointFunction{,Backward}? ===")
# depth-limited ancestor walk via recursive CTE
print(q("""
WITH RECURSIVE anc(id, root_name, depth) AS (
  SELECT s.parent_id, 'x', 0 FROM slice s
  JOIN slice p ON s.parent_id=p.id AND p.name='aten::copy_'
  WHERE s.name='cudaMemcpyAsync'
  UNION ALL
  SELECT sp.parent_id, 'x', depth+1 FROM anc JOIN slice sp ON sp.id=anc.id WHERE depth<25
)
SELECT 'dummy'
LIMIT 0
""").to_string())

print("=== streamSync under nonzero: grandparent split ===")
print(q("""
SELECT gp.name gpn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='aten::nonzero'
JOIN slice gp ON p.parent_id=gp.id
WHERE s.name='cudaStreamSynchronize'
GROUP BY gpn ORDER BY s DESC LIMIT 10
""").to_string())

print("=== _local_scalar_dense syncs: grandparent ===")
print(q("""
SELECT gp.name gpn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='aten::_local_scalar_dense'
JOIN slice gp ON p.parent_id=gp.id
WHERE s.name IN ('cudaStreamSynchronize','cudaMemcpyAsync')
GROUP BY gpn ORDER BY s DESC LIMIT 10
""").to_string())

print("=== memcpy >1ms: parent split ===")
print(q("""
SELECT p.name pn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id
WHERE s.name='cudaMemcpyAsync' AND s.dur > 1000000
GROUP BY pn ORDER BY s DESC LIMIT 10
""").to_string())

print("=== aten::to: children API calls ===")
print(q("""
SELECT s.name sn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='aten::to'
GROUP BY sn ORDER BY s DESC LIMIT 10
""").to_string())

print("=== copy_ memcpys >1ms: grandparent ===")
print(q("""
SELECT gp.name gpn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id AND p.name='aten::copy_'
JOIN slice gp ON p.parent_id=gp.id
WHERE s.name='cudaMemcpyAsync' AND s.dur > 1000000
GROUP BY gpn ORDER BY s DESC LIMIT 15
""").to_string())
