"""Replay measured routing shapes, time both GEMM paths, or expose one NVTX range."""
import argparse
import csv
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from expert_gemm import ExpertGEMMPair

parser = argparse.ArgumentParser()
parser.add_argument("--counts", type=Path, default=Path(__file__).with_name("counts.csv"))
parser.add_argument("--mode", choices=("benchmark", "profile", "trace"), default="benchmark")
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--config", choices=("CP8EP1", "CP8EP8"))
parser.add_argument("--layer", type=int, choices=(1, 2))
parser.add_argument("--projection", choices=("gate_up", "down"))
parser.add_argument("--implementation", choices=("te", "grouped", "both"), default="both")
parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--repeats", type=int, default=20)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
if args.mode == "profile" and (not args.config or not args.layer or not args.projection or args.implementation == "both"):
    parser.error("profile mode requires one config/layer/projection/implementation")

torch.set_num_threads(1)
torch.manual_seed(328)
with args.counts.open() as stream:
    routing = list(csv.DictReader(stream))
records = []
configs = [args.config] if args.config else ["CP8EP1", "CP8EP8"]
layers = [args.layer] if args.layer else [1, 2]
projections = {"gate_up": (6144, 4096), "down": (2048, 6144)}
if args.projection:
    projections = {args.projection: projections[args.projection]}


def error_metrics(reference, candidate, indices):
    a, b = reference.detach()[indices].float(), candidate.detach()[indices].float()
    error = a - b
    result = {"max_absolute": error.abs().max().item(),
              "relative_rms": (error.square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-20)).item()}
    assert torch.isfinite(b).all().item() and result["relative_rms"] < 0.01, result
    return result


for config in configs:
    for layer in layers:
        selected = sorted((r for r in routing if r["config"] == config and int(r["rank"]) == args.rank
                           and int(r["moe_layer"]) == layer and r["stage"] == "expert_input"), key=lambda r: int(r["expert"]))
        assert len(selected) == (256 if config == "CP8EP1" else 32)
        rows = [int(r["tokens"]) for r in selected]
        expert_ids = [int(r["expert"]) for r in selected]
        sample_rows, start = [], 0
        for count in rows:
            if count:
                sample_rows.extend([start, start + count // 2, start + count - 1])
            start += count
        indices = torch.tensor(sample_rows, device="cuda")
        for projection, (kin, nout) in projections.items():
            pair = ExpertGEMMPair(rows, kin, nout)
            x = torch.randn(sum(rows), kin, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            grad = torch.randn(sum(rows), nout, device="cuda", dtype=torch.bfloat16)
            functions = {"te": pair.te_forward, "grouped": pair.grouped_forward}
            chosen = list(functions) if args.implementation == "both" else [args.implementation]
            parity = {}
            if args.mode == "benchmark":
                a = pair.te_forward(x)
                da = torch.autograd.grad(a, x, grad)[0]
                b = pair.grouped_forward(x)
                db = torch.autograd.grad(b, x, grad)[0]
                parity = {"forward": error_metrics(a, b, indices), "dgrad": error_metrics(da, db, indices)}
                del a, b, da, db
            for implementation in chosen:
                function = functions[implementation]
                for _ in range(args.warmup):
                    x.grad = None
                    y = function(x)
                    y.backward(grad)
                torch.cuda.synchronize()
            measurements = {key: {"forward_ms": [], "backward_ms": [], "forward_enqueue_ms": [], "backward_enqueue_ms": []} for key in chosen}
            if args.mode == "trace":
                torch.cuda.cudart().cudaProfilerStart()
            if args.mode == "profile":
                x.grad = None
                if args.phase == "backward":
                    y = functions[chosen[0]](x)
                    torch.cuda.synchronize()
                with torch.cuda.nvtx.range("profile"):
                    if args.phase == "forward":
                        y = functions[chosen[0]](x)
                    else:
                        y.backward(grad)
                    torch.cuda.synchronize()
            else:
                for iteration in range(args.repeats):
                    # Alternate implementation order to reduce monotonic clock/thermal bias.
                    order = chosen if iteration % 2 == 0 else list(reversed(chosen))
                    for implementation in order:
                        function = functions[implementation]
                        x.grad = None
                        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                        prefix = f"{config}/L{layer}/rank{args.rank}/{projection}/{implementation}"
                        events[0].record()
                        with torch.cuda.nvtx.range(prefix + "/forward"):
                            t0 = time.perf_counter()
                            y = function(x)
                            t1 = time.perf_counter()
                        events[1].record()
                        with torch.cuda.nvtx.range(prefix + "/backward"):
                            t2 = time.perf_counter()
                            y.backward(grad)
                            t3 = time.perf_counter()
                        events[2].record()
                        events[2].synchronize()
                        item = measurements[implementation]
                        item["forward_ms"].append(events[0].elapsed_time(events[1]))
                        item["backward_ms"].append(events[1].elapsed_time(events[2]))
                        item["forward_enqueue_ms"].append((t1 - t0) * 1000)
                        item["backward_enqueue_ms"].append((t3 - t2) * 1000)
            if args.mode == "trace":
                torch.cuda.cudart().cudaProfilerStop()
            record = {"config": config, "layer": layer, "rank": args.rank, "projection": projection,
                      "rows": rows, "expert_ids": expert_ids, "total_rows": sum(rows), "K": kin, "N": nout,
                      "flops_per_phase": 2 * sum(rows) * kin * nout,
                      "weight_bytes_per_implementation": len(rows) * kin * nout * 2,
                      "parity": parity, "mode": args.mode, "phase": args.phase,
                      "measurements": {key: {metric: {"mean": statistics.mean(values), "median": statistics.median(values),
                                                       "sd": statistics.stdev(values) if len(values) > 1 else 0,
                                                       "samples": values} for metric, values in item.items() if values}
                                       for key, item in measurements.items()}}
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ("rows", "expert_ids", "measurements")}), flush=True)
            del pair, x, grad, y, functions, function
            gc.collect()
            torch.cuda.empty_cache()
result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"), "records": records}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
