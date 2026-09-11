"""Summarize unprofiled timings separately from observer-affected NCU counters."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).parent
results = root / "results"
timing = defaultdict(lambda: defaultdict(float))
max_error = 0
files = sorted(results.glob("timing.rank*.json"))
assert len(files) == 8
for path in files:
    data = json.loads(path.read_text())
    assert len(data["records"]) == 8
    for record in data["records"]:
        for error in record["parity"].values():
            max_error = max(max_error, error["relative_rms"])
        for implementation, metrics in record["measurements"].items():
            for phase in ("forward_ms", "backward_ms"):
                assert len(metrics[phase]["samples"]) == 20
                key = (record["config"], record["layer"], implementation, phase)
                timing[key][record["rank"]] += metrics[phase]["mean"]
summary = []
for (config, layer, implementation, phase), ranks in sorted(timing.items()):
    assert len(ranks) == 8
    summary.append(dict(config=config, layer=layer, implementation=implementation, phase=phase,
                        mean_ms=statistics.mean(ranks.values()), min_ms=min(ranks.values()), max_ms=max(ranks.values()), by_rank=ranks))
range_profiles = []
for path in sorted(results.glob("*range.csv")):
    with path.open() as stream:
        reader = csv.reader(stream)
        header = next(reader)
        units = dict(zip(header, next(reader)))
        records = [dict(zip(header, row)) for row in reader if row and row[0]]
    assert len(records) == 1, path
    record = records[0]
    def value(name):
        number = float(record[name].replace(",", ""))
        factor = {"": 1, "%": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9,
                  "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}[units[name]]
        return number * factor
    range_profiles.append({"file": path.name,
                           "profiled_seconds_not_benchmark": value("gpu__time_duration.sum"),
                           "tensor_ops": value("sm__ops_path_tensor_src_bf16_dst_fp32.sum"),
                           "dram_read_bytes": value("dram__bytes_read.sum"),
                           "dram_write_bytes": value("dram__bytes_write.sum")})
assert len(range_profiles) == 16
with (results / "counts.csv").open() as stream:
    routing = list(csv.DictReader(stream))
padding = []
for profile in range_profiles:
    if ".grouped." not in profile["file"]:
        continue
    parts = profile["file"].split(".")
    config, layer, rank, projection = parts[0].upper(), int(parts[1][1:]), int(parts[2][4:]), parts[3]
    rows = [int(r["tokens"]) for r in routing if r["config"] == config and int(r["moe_layer"]) == layer
            and int(r["rank"]) == rank and r["stage"] == "expert_input"]
    padded_rows = sum(math.ceil(m / 256) * 256 for m in rows)
    kin, nout = {"gate_up": (6144, 4096), "down": (2048, 6144)}[projection]
    assert profile["tensor_ops"] == 2 * padded_rows * kin * nout, profile["file"]
    padding.append({"file": profile["file"], "useful_rows": sum(rows), "effective_256_tile_rows": padded_rows,
                    "extra_arithmetic_percent": 100 * (padded_rows / sum(rows) - 1)})
(root / "summary.json").write_text(json.dumps({"timing": summary, "max_sampled_relative_rms_error": max_error,
                                              "ncu_range_profiles": range_profiles,
                                              "validated_grouped_padding": padding,
                                              "warning": "NCU range duration/utilization is observer-distorted; use unprofiled timings for performance."}, indent=2))
print("Maximum sampled relative RMS error:", max_error)
print("Config | MoE | TE forward | Grouped forward | TE input-grad | Grouped input-grad")
for config in ("CP8EP1", "CP8EP8"):
    for layer in (1, 2):
        values = [statistics.mean(timing[config, layer, impl, phase].values()) for phase in ("forward_ms", "backward_ms") for impl in ("te", "grouped")]
        print(config, layer, *[f"{v:.3f}" for v in values])
