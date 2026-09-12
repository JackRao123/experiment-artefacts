"""Summarize headline controls separately from continued stability validation."""
import argparse
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("benchmarks", nargs="+", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()


def summarize(windows, tokens, gpus):
    if not windows:
        return None
    times = [w["fb_elapsed_s"] for w in windows]
    opt = [w["optim_elapsed_s"] for w in windows]
    return {"n":len(times), "mean_fb_s":statistics.mean(times),
        "sd_fb_s":statistics.stdev(times) if len(times)>1 else None,
        "median_fb_s":statistics.median(times), "min_fb_s":min(times), "max_fb_s":max(times),
        "tps_per_gpu":tokens/gpus/statistics.mean(times),
        "mean_optimizer_s":statistics.mean(opt),
        "tps_including_optimizer_per_gpu":tokens/gpus/statistics.mean(a+b for a,b in zip(times,opt)),
        "peak_allocated_gib":max(w["peak_allocated_bytes"] for w in windows)/2**30,
        "loss_first":windows[0]["loss"], "loss_last":windows[-1]["loss"],
        "grad_norm_last":windows[-1]["grad_norm"],
        "all_finite":all(math.isfinite(w[k]) for w in windows for k in ("loss","grad_norm"))}


rows = []
for path in args.benchmarks:
    data = json.loads(path.read_text())
    controls = [w for w in data["windows"] if w["phase"]=="control"]
    validation = [w for w in data["windows"] if w["phase"]=="validation"]
    if data.get("continued_from_case"):
        validation, controls = controls, []
    params = data["tokens_per_step"], data["num_gpus"]
    row = {"case":data["case"], "capture_complete":data["capture_complete"],
        "starting_optimizer_step":data["initial_status"]["step"],
        "ending_optimizer_step":data.get("final_status",{}).get("step"),
        "controls":summarize(controls,*params), "validation":summarize(validation,*params),
        "validation_last10":summarize(validation[-10:],*params),
        "source_revisions":data["source_revisions"], "runtime_options":data["runtime_options"]}
    rows.append(row)
text = json.dumps(rows,indent=2)+"\n"
if args.output:
    args.output.write_text(text)
print(text)
