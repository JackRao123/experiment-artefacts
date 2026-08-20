import sys
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

print("=== user_annotation distinct (top 30 by count) ===")
print(q("SELECT name, COUNT(*) c, SUM(dur)/1e9 s FROM slice WHERE category='user_annotation' GROUP BY name ORDER BY c DESC LIMIT 30").to_string())
print("=== cudaEventSynchronize immediate parents ===")
print(q("SELECT p.name pn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s JOIN slice p ON s.parent_id=p.id WHERE s.name='cudaEventSynchronize' GROUP BY pn ORDER BY s DESC LIMIT 15").to_string())
print("=== cudaStreamSynchronize immediate parents (top 15 by time) ===")
print(q("SELECT p.name pn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s JOIN slice p ON s.parent_id=p.id WHERE s.name='cudaStreamSynchronize' GROUP BY pn ORDER BY s DESC LIMIT 15").to_string())
print("=== cudaMemcpyAsync immediate parents (top 15 by time) ===")
print(q("SELECT p.name pn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s JOIN slice p ON s.parent_id=p.id WHERE s.name='cudaMemcpyAsync' GROUP BY pn ORDER BY s DESC LIMIT 15").to_string())
print("=== aten::nonzero immediate parents ===")
print(q("SELECT p.name pn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s JOIN slice p ON s.parent_id=p.id WHERE s.name='aten::nonzero' GROUP BY pn ORDER BY s DESC LIMIT 10").to_string())
