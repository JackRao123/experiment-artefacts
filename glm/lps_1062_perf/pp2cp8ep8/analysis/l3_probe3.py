import sys
from collections import Counter
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

# full cpu_op/annotation parent map
ops = q("SELECT id, parent_id, name, category FROM slice WHERE category IN ('cpu_op','user_annotation')")
op_name = dict(zip(ops.id, ops.name))
op_parent = dict(zip(ops.id, ops.parent_id))
del ops

def chain_names(sid, max_depth=30):
    out = []
    pid = op_parent.get(sid)
    d = 0
    while pid and pid in op_name and d < max_depth:
        out.append(op_name[pid])
        pid = op_parent.get(pid)
        d += 1
    return out

def sig(anc):
    # compress: keep aten/autograd/checkpoint milestones, drop repeats
    keep = [n for n in anc]
    out = []
    for n in keep:
        if not out or out[-1] != n:
            out.append(n)
    return " < ".join(out[:8])

for label, sql in [
    ("BIG DtoH copy_ (>1ms)", """
        SELECT s.id, s.dur FROM slice s
        JOIN slice p ON s.parent_id=p.id AND p.name='aten::copy_'
        JOIN slice gp ON p.parent_id=gp.id AND gp.name='aten::_to_copy'
        WHERE s.name='cudaMemcpyAsync' AND s.dur > 1000000"""),
    ("small DtoH copy_ (<=1ms, sample 3000)", """
        SELECT s.id, s.dur FROM slice s
        JOIN slice p ON s.parent_id=p.id AND p.name='aten::copy_'
        JOIN slice gp ON p.parent_id=gp.id AND gp.name='aten::_to_copy'
        WHERE s.name='cudaMemcpyAsync' AND s.dur <= 1000000 LIMIT 3000"""),
    ("item/_local_scalar_dense syncs (sample 3000)", """
        SELECT s.id, s.dur FROM slice s
        JOIN slice p ON s.parent_id=p.id AND p.name='aten::_local_scalar_dense'
        WHERE s.name IN ('cudaStreamSynchronize','cudaMemcpyAsync') LIMIT 3000"""),
]:
    rows = q(sql)
    c = Counter()
    t = Counter()
    for sid, dur in zip(rows.id, rows.dur):
        anc = chain_names(sid)
        sg = sig(anc)
        c[sg] += 1
        t[sg] += dur
    print(f"=== {label} — {len(rows)} rows ===")
    for sg, n in t.most_common(8):
        print(f"  {n/1e9:8.3f}s  {c[sg]:6d}x  {sg}")
    print()

# aten::to >1ms: direct parent names (the Aug-9 RoPE test)
print("=== aten::to slices >1ms: direct parents ===")
print(q("""
SELECT p.name pn, COUNT(*) c, SUM(s.dur)/1e9 s FROM slice s
JOIN slice p ON s.parent_id=p.id
WHERE s.name='aten::to' AND s.dur > 1000000
GROUP BY pn ORDER BY s DESC LIMIT 10
""").to_string())
