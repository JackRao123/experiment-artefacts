"""Validate original-forward captures and export tidy counts for visualization."""
import csv
import json
from pathlib import Path

root = Path(__file__).parent
rows = []
checks = {}
hashes = set()
for config, ep in (("CP8EP1", 1), ("CP8EP8", 8)):
    case = root / config.lower()
    result = json.loads((case / "forward-result.json").read_text())
    hashes.add(result["input_sha256"])
    captures = {}
    for path in sorted((case / "counts").glob("*.json")):
        capture = json.loads(path.read_text())
        if max(capture["datum_lengths"], default=0) < 100000:
            continue  # Exclude the startup warmup.
        rank = capture["rank"]
        assert rank not in captures, (config, rank, "multiple captured requests")
        assert capture["original_model_forwards"] == 1, capture
        assert capture["ignored_outside_original_forward"] >= 4, capture
        assert len(capture["records"]) == 4, capture
        captures[rank] = capture
    assert set(captures) == set(range(8)), (config, captures.keys())
    layers = sorted({record["layer_number"] for c in captures.values() for record in c["records"]})
    assert len(layers) == 2, layers
    by_key = {}
    for rank, capture in captures.items():
        for record in capture["records"]:
            layer = record["layer_number"]
            stage = record["stage"]
            assert record["model_forward"] == 1
            assert record["top_k"] == 8 and record["num_experts"] == 256
            assert record["ep_size"] == ep
            counts = dict(zip(record["expert_ids"], record["counts"], strict=True))
            assert all(isinstance(value, int) and value >= 0 for value in counts.values())
            assert sum(counts.values()) == record["input_rows"] * (8 if stage == "router" else 1)
            by_key[rank, layer, stage] = counts
            for expert, count in counts.items():
                rows.append({"config": config, "rank": rank, "moe_layer": layers.index(layer) + 1,
                             "model_layer": layer, "stage": stage, "expert": expert, "tokens": count})
    for layer in layers:
        total = sum(sum(by_key[rank, layer, "router"].values()) for rank in range(8))
        assert total == 131072 * 8, (config, layer, total)
        for rank in range(8):
            actual = by_key[rank, layer, "expert_input"]
            expected = {
                expert: (by_key[rank, layer, "router"][expert] if ep == 1 else
                         sum(by_key[source, layer, "router"][expert] for source in range(8)))
                for expert in actual
            }
            assert actual == expected, (config, rank, layer, "dispatch counts differ from routed assignments")
    checks[config] = {"ranks": 8, "moe_layers": layers, "routed_assignments_per_layer": 131072 * 8,
                      "original_forward_only": True, "dispatch_conservation": True,
                      "ignored_recompute_hooks": {rank: c["ignored_outside_original_forward"] for rank, c in captures.items()}}
assert len(hashes) == 1, "Configurations used different input"
with (root / "counts.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "validation.json").write_text(json.dumps({"input_sha256": hashes.pop(), "checks": checks}, indent=2))
print(json.dumps(checks, indent=2))
