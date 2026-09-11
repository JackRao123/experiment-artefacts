"""Verify a completed nsys capability capture, not its mere exit status."""
import json
import sqlite3
import sys

db = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
kernels = db.execute("SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
samples, gpus = db.execute("SELECT count(*),count(DISTINCT(typeId & 255)) FROM GPU_METRICS").fetchone()
assert kernels > 0 and samples > 0 and gpus == 8, (kernels, samples, gpus)
print(json.dumps({"kernels": kernels, "metric_samples": samples, "metric_gpus": gpus,
                  "builtin_nsys_valid": True}))
