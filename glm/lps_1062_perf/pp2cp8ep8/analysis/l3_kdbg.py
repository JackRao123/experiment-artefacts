import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

# layout-builder aten::index ops (parents of layout nonzeros)
nz = q("""SELECT s.parent_id AS pid FROM slice s WHERE s.name='aten::nonzero'""")
idx_ops = q("""SELECT p.id, p.ts, p.dur FROM slice p
               WHERE p.name='aten::index' AND p.id IN (
                   SELECT s.parent_id FROM slice s WHERE s.name='aten::nonzero')""")
print("index ops:", len(idx_ops))
w = sorted((int(t), int(t + d)) for t, d in zip(idx_ops.ts, idx_ops.dur))
merged = []
for s_, e_ in w:
    if merged and s_ <= merged[-1][1]:
        if e_ > merged[-1][1]:
            merged[-1][1] = e_
    else:
        merged.append([s_, e_])
print("merged windows:", len(merged), "first:", merged[0] if merged else None)

kk = q("SELECT ts, dur FROM slice WHERE category='kernel' AND dur>0")
ka = kk.ts.to_numpy(dtype=np.int64)
ka.sort()
print("kernels:", len(ka), "ka[0]:", ka[0], "ka[-1]:", ka[-1])
ms = np.array([m[0] for m in merged], dtype=np.int64)
me = np.array([m[1] for m in merged], dtype=np.int64)
print("ms[0]:", ms[0], "me[-1]:", me[-1])
idx = np.searchsorted(me, ka, side="left")
ok = idx < len(ms)
idxc = np.minimum(idx, len(ms) - 1)
hit = ok & (ka >= ms[idxc]) & (ka < me[idxc])
print("kernels in layout windows:", int(hit.sum()))
# window union coverage
cov = int((me - ms).sum()) / 1e9
print("layout window union coverage s:", cov)
