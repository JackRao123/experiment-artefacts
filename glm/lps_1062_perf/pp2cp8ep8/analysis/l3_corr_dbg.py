import sys
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

# 5 kernel correlations
k = q("""
SELECT corr.display_value AS c FROM slice s
JOIN args corr ON corr.arg_set_id=s.arg_set_id AND corr.key='args.correlation'
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%' LIMIT 5
""")
corrs = [str(x) for x in k["c"]]
print("kernel corrs:", corrs)

for c in corrs[:3]:
    rows = q(f"""
    SELECT s.id, s.category, s.name, s.parent_id FROM slice s
    JOIN args a ON a.arg_set_id=s.arg_set_id AND a.key='args.correlation'
    AND a.display_value='{c}'
    """)
    print(f"--- corr {c}: {len(rows)} slices")
    for r in rows.itertuples():
        print(f"    id={r.id} cat={r.category} name={r.name} parent={r.parent_id}")

# which categories have correlation args at all
print(q("""
SELECT s.category, COUNT(*) c FROM slice s
JOIN args a ON a.arg_set_id=s.arg_set_id AND a.key='args.correlation'
GROUP BY s.category
""").to_string())
