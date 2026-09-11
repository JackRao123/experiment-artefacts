"""Compare two analyzed captures. Mismatched workload settings fail closed."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("left", type=Path)
parser.add_argument("right", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
left, right = (json.loads(p.read_text()) for p in (args.left, args.right))
bl, br = left["benchmark"], right["benchmark"]
if not bl or not br:
    raise ValueError("Both analyses require --benchmark metadata")
for key in ("sequence_length", "num_gpus", "tokens_per_step", "input_sha256", "step_definition"):
    if bl[key] != br[key]:
        raise ValueError(f"Mismatched {key}: {bl[key]} vs {br[key]}")
for key in bl.get("runtime_options", {}).keys() | br.get("runtime_options", {}).keys():
    if key != "grouped_mm" and bl.get("runtime_options", {}).get(key) != br.get("runtime_options", {}).get(key):
        raise ValueError(f"Mismatched runtime option {key}")
if bl.get("source_revisions") != br.get("source_revisions"):
    raise ValueError("Source revisions differ; establish comparability before comparing")
allowed = {"expert_parallel_size", "checkpoint_dir"}
for key in bl["config"].keys() | br["config"].keys():
    if key not in allowed and bl["config"].get(key) != br["config"].get(key):
        raise ValueError(f"Mismatched config {key}")
if set(left["ranks"]) != set(right["ranks"]):
    raise ValueError("Different captured rank sets")
lines = ["# Matched Nsight comparison", "", f"Left: {bl['case']}; right: {br['case']}.", "",
         "Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.", "",
         "| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |", "|---|---:|---:|---:|---:|"]
for b in (bl, br):
    peak = max(w["peak_allocated_bytes"] for w in b["windows"] if w["phase"] == "control") / 2**30
    lines.append(f"| {b['case']} | {b['control']['n']} | {b['control']['mean_s']:.4f} | {b['control']['tps_per_gpu']:.1f} | {peak:.2f} |")
lines += ["", "## Per-rank reconciliation", "", "All deltas are right minus left, milliseconds per profiled FB.", "",
          "Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.", "",
          "| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
rows = []
for rank in sorted(left["ranks"], key=int):
    l, r = left["ranks"][rank], right["ranks"][rank]
    cats = l["categories"].keys() | r["categories"].keys()
    def category(c, key, field):
        return c["categories"].get(key, {}).get(field, {}).get("mean", 0)
    deltas = {c: category(r,c,"exclusive_ms") - category(l,c,"exclusive_ms") for c in cats}
    mean = lambda x,k: statistics.mean(s[k] for s in x["steps"])
    delta = mean(r,"step_ms")-mean(l,"step_ms")
    mixed = mean(r,"mixed_category_overlap_ms")-mean(l,"mixed_category_overlap_ms")
    idle = mean(r,"device_idle_ms")-mean(l,"device_idle_ms")
    residual = delta-sum(deltas.values())-mixed-idle
    assert abs(residual) < 1e-4, residual
    expert, dispatcher = deltas.get("expert_gemm_path",0), deltas.get("expert_dispatch_combine",0)
    other = sum(deltas.values())-expert-dispatcher
    values = [delta, expert, dispatcher, other, mixed, idle, residual]
    lines.append(f"| {rank} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    rows.append({"rank": rank, "step_delta_ms": delta, "exclusive_category_deltas_ms": deltas,
                 "mixed_overlap_delta_ms": mixed, "idle_delta_ms": idle, "residual_ms": residual})
lines += ["", "## Interpretation limits", "",
          "- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.",
          "- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.",
          "- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.",
          "- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.",
          "- Compare loss/gradients and profiler slowdown before accepting a performance conclusion."]
args.output.write_text("\n".join(lines)+"\n")
args.output.with_suffix(".json").write_text(json.dumps({"left": str(args.left), "right": str(args.right), "deltas": rows}, indent=2))
print(args.output)
