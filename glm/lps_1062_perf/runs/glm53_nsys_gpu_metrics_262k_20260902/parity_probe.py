"""Per-token logprob parity probe for the running trainer.

Protocol: with the trainer up, ``--reinit`` resets the LoRA adapter to a fresh
init (B = 0, so the model equals the frozen base model), then ``--runs N`` runs
``/forward`` (no gradient side effects) on the same synthetic datum (seed
0xB300, like profile_driver.py) N times and saves loss + per-token logprobs.
``--compare A B`` reports the difference between two saved runs. Two runs on
the same build give the noise floor (cuDNN DSA top-k nondeterminism); a run on
another build is compared against that floor.

  PYTHONPATH=$REMOTE_RUN python parity_probe.py --reinit --runs 2 --label fix1 --seq-len 262144
  PYTHONPATH=$REMOTE_RUN python parity_probe.py --compare ab/parity-fix1-run0.json ab/parity-fix1-run1.json
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import httpx

from profile_driver import BASE_URL, FB_TIMEOUT_S, OP_TIMEOUT_S, make_datum, submit_and_wait

OUT = Path("/root/.cache/user_artifacts/lps1062_bench/glm_nsys_gpu_metrics_262k_20260902/ab")


def _flatten_logprobs(result: dict) -> list[float]:
    vals: list[float] = []
    for out in result.get("loss_fn_outputs", []):
        for key, td in out.items():
            if "logprob" in key and isinstance(td, dict) and "data" in td:
                data = td["data"]
                if data and isinstance(data[0], list):
                    data = [x for row in data for x in row]
                vals.extend(float(x) for x in data)
    return vals


def run(args: argparse.Namespace) -> None:
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        if args.reinit:
            print("[reinit] /init_trainer_server lora_rank=32 lora_alpha=32", flush=True)
            r = submit_and_wait(
                client, "/init_trainer_server", {"lora_rank": 32, "lora_alpha": 32}, OP_TIMEOUT_S
            )
            print(f"[reinit] -> {r}", flush=True)
        for i in range(args.runs):
            rng = random.Random(args.seed)
            datum = make_datum(rng, args.seq_len)
            t0 = time.perf_counter()
            res = submit_and_wait(client, "/forward", {"data": [datum]}, FB_TIMEOUT_S)
            dt = time.perf_counter() - t0
            lp = _flatten_logprobs(res)
            keys = [list(o.keys()) for o in res.get("loss_fn_outputs", [])]
            path = OUT / f"parity-{args.label}-run{i}.json"
            path.write_text(json.dumps({"loss": res.get("loss"), "metrics": res.get("metrics"),
                                        "logprobs": lp, "output_keys": keys}))
            finite = [x for x in lp if math.isfinite(x)]
            print(
                f"[{args.label} run{i}] forward={dt:.1f}s loss={res.get('loss')} "
                f"logprobs n={len(lp)} finite={len(finite)} mean={sum(finite)/max(1,len(finite)):.6f} -> {path}",
                flush=True,
            )


def compare(a_path: str, b_path: str) -> None:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    la, lb = a["logprobs"], b["logprobs"]
    assert len(la) == len(lb), (len(la), len(lb))
    pairs = [(x, y) for x, y in zip(la, lb) if math.isfinite(x) and math.isfinite(y)]
    d = [abs(x - y) for x, y in pairs]
    n = len(d)
    d_sorted = sorted(d)
    eq = sum(1 for x in d if x == 0.0)
    print(f"compare {Path(a_path).name} vs {Path(b_path).name}")
    print(f"  loss  a={a['loss']}  b={b['loss']}  delta={b['loss'] - a['loss']:+.3e}")
    print(f"  tokens compared={n}  bitwise-equal fraction={eq / max(1, n):.4f}")
    print(
        f"  |delta logprob| mean={sum(d) / max(1, n):.3e}  p50={d_sorted[n // 2]:.3e}  "
        f"p99={d_sorted[int(n * 0.99)]:.3e}  p99.9={d_sorted[int(n * 0.999)]:.3e}  max={d_sorted[-1]:.3e}"
    )
    print(f"  mean logprob a={sum(x for x, _ in pairs) / n:.6f}  b={sum(y for _, y in pairs) / n:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="probe")
    ap.add_argument("--seq-len", type=int, default=262144)
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0xB300)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--reinit", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        run(args)


if __name__ == "__main__":
    main()
