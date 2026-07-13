#!/usr/bin/env python3
"""CP1-vs-CP2 parity driver for per-token logprobs + RL losses under THD CP.

Run mode (against a live trainer on --trainer-url):
  For each loss_fn in [cross_entropy, dppo, importance_sampling, ppo, cispo,
  dro]: submit ONE deterministic 3-datum /forward_backward, record loss +
  per-datum wire logprobs, then /optim_step (lr=1e-10) to record grad_norm
  and reset grad buffers between losses. Writes a JSON blob to --out.

Compare mode (--compare A.json B.json):
  A = cp1 run, B = cp2 run. Checks per loss_fn:
    - relative loss diff
    - per-datum logprob rows: max/mean abs diff (the logprob-stitch check)
    - grad_norm ratio (report; the cp2/cp1 ratio should match CE's ratio)

Datums are built from fixed seeds, so cp1 and cp2 runs see identical bytes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid

LOSS_FNS = ["cross_entropy", "dppo", "importance_sampling", "ppo", "cispo", "dro"]
DATUM_LENGTHS = [173, 96, 41]  # odd lengths exercise per-doc 2*cp padding
VOCAB = 1800  # debug GLM vocab is 2048


def _rng(seed: int):
    # Tiny deterministic LCG so both runs build identical datums with no deps.
    state = seed & 0x7FFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def build_datums() -> list[dict]:
    datums = []
    for di, L in enumerate(DATUM_LENGTHS):
        rnd = _rng(1000 + di)
        tokens = [1 + int(rnd() * (VOCAB - 2)) for _ in range(L)]
        targets = tokens[1:] + [-100]
        # CE mask: first ~half is prompt (weight 0), rest supervised.
        cut = L // 2
        weights = [0.0] * cut + [1.0] * (L - cut)
        # RL fields: plausible old-logprobs; advantages nonzero on the
        # supervised tail with a few interior zeros (mask holes).
        logprobs = [-0.05 - 2.5 * rnd() for _ in range(L)]
        advantages = [0.0] * L
        for i in range(cut, L - 1):
            advantages[i] = 0.0 if rnd() < 0.15 else round(2.0 * rnd() - 1.0, 4) or 0.5
        temperature = 0.7 if di == 0 else 1.0
        temperatures = [temperature] * L

        def td(data, dtype):
            return {"data": data, "dtype": dtype, "shape": [len(data)]}

        datums.append(
            {
                "model_input": {
                    "chunks": [{"type": "encoded_text", "tokens": tokens}]
                },
                "loss_fn_inputs": {
                    "target_tokens": td(targets, "int64"),
                    "weights": td(weights, "float32"),
                    "logprobs": td(logprobs, "float32"),
                    "advantages": td(advantages, "float32"),
                    "temperatures": td(temperatures, "float32"),
                },
            }
        )
    return datums


def submit_and_wait(client, op_path: str, *, body: dict, timeout: float = 1800.0):
    key = uuid.uuid4().hex
    r = client.post(op_path, json=body, headers={"Idempotency-Key": key})
    if r.status_code != 202:
        raise RuntimeError(f"{op_path} submit failed {r.status_code}: {r.text[:800]}")
    operation_id = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{operation_id}", timeout=35.0)
        if rr.status_code == 408:
            continue
        rr.raise_for_status()
        payload = rr.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{op_path} op error: {payload.get('error', '')[:4000]}")
    raise TimeoutError(op_path)


def run(args) -> None:
    import httpx

    datums = build_datums()
    results: dict = {"datum_lengths": DATUM_LENGTHS, "losses": {}}
    with httpx.Client(base_url=args.trainer_url, timeout=1800.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        status = client.get("/status", timeout=30.0).json()
        results["parallel"] = {
            k: status.get(k)
            for k in (
                "context_parallel_size",
                "data_parallel_size",
                "expert_parallel_size",
                "world_size",
            )
        }
        print(f"[parity] trainer parallel: {results['parallel']}", flush=True)

        for loss_fn in LOSS_FNS:
            t0 = time.perf_counter()
            fb = submit_and_wait(
                client,
                "/forward_backward",
                body={"data": datums, "loss_fn": loss_fn},
            )
            rows = [
                out.get("logprobs", {}).get("data")
                for out in fb.get("loss_fn_outputs", [])
            ]
            opt = submit_and_wait(
                client,
                "/optim_step",
                body={
                    "adam_params": {
                        "learning_rate": 1e-10,
                        "beta1": 0.9,
                        "beta2": 0.95,
                    }
                },
            )
            grad_norm = (opt.get("metrics") or {}).get("grad_norm", opt.get("grad_norm"))
            results["losses"][loss_fn] = {
                "loss": fb.get("loss"),
                "grad_norm": grad_norm,
                "n_rows": len(rows),
                "row_lengths": [len(r) if r else 0 for r in rows],
                "logprobs": rows,
            }
            print(
                f"[parity] {loss_fn}: loss={fb.get('loss'):.6f} "
                f"grad_norm={grad_norm} rows={[len(r) if r else 0 for r in rows]} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[parity] wrote {args.out}", flush=True)


def _mean_abs_diff(rows_a: list, rows_b: list, shift: int) -> tuple[float, int]:
    """Mean |Δ| over positions active (nonzero) in both, at the given relative
    shift. shift=0 is the real comparison; shift=±1 is the discriminator: any
    misalignment anywhere would push the aligned number toward the shifted one
    (~0.6+ nats on this model) instead of fp noise (~5e-3)."""
    total, n = 0.0, 0
    for rowa, rowb in zip(rows_a, rows_b):
        if shift >= 0:
            pa, pb = rowa[shift:], rowb[: len(rowb) - shift or None]
        else:
            pa, pb = rowa[: len(rowa) + shift], rowb[-shift:]
        for va, vb in zip(pa, pb):
            if va != 0.0 and vb != 0.0:
                total += abs(va - vb)
                n += 1
    return total / max(n, 1), n


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    print(f"A={path_a} parallel={a['parallel']}")
    print(f"B={path_b} parallel={b['parallel']}")
    hard_fail = False
    ce_gn_ratio = None
    gn_ratios: dict[str, float] = {}
    for loss_fn in LOSS_FNS:
        ra, rb = a["losses"][loss_fn], b["losses"][loss_fn]
        la, lb = ra["loss"], rb["loss"]
        loss_rel = abs(la - lb) / max(abs(la), 1e-9)
        gna, gnb = ra.get("grad_norm") or 0.0, rb.get("grad_norm") or 0.0
        gn_ratio = (gnb / gna) if gna else float("nan")
        gn_ratios[loss_fn] = gn_ratio
        if loss_fn == "cross_entropy":
            ce_gn_ratio = gn_ratio
        if len(ra["logprobs"]) != len(rb["logprobs"]) or any(
            x is None or y is None or len(x) != len(y)
            for x, y in zip(ra["logprobs"], rb["logprobs"])
        ):
            print(f"  {loss_fn}: ROW COUNT/SHAPE MISMATCH")
            hard_fail = True
            continue
        max_d = max(
            (
                abs(va - vb)
                for rowa, rowb in zip(ra["logprobs"], rb["logprobs"])
                for va, vb in zip(rowa, rowb)
            ),
            default=0.0,
        )
        mean_d, n = _mean_abs_diff(ra["logprobs"], rb["logprobs"], 0)
        shift_d = min(
            _mean_abs_diff(ra["logprobs"], rb["logprobs"], 1)[0],
            _mean_abs_diff(ra["logprobs"], rb["logprobs"], -1)[0],
        )
        # Gates: loss parity within fp-nondeterminism envelope; aligned
        # logprob diff at noise level AND far below the off-by-one floor.
        aligned_ok = mean_d < 2e-2 and (shift_d < 0.05 or mean_d < shift_d / 20)
        ok = loss_rel < 1e-2 and aligned_ok
        hard_fail |= not ok
        print(
            f"  {loss_fn:20s} loss A={la:.6f} B={lb:.6f} rel={loss_rel:.2e} | "
            f"logprobs mean|Δ|={mean_d:.2e} max|Δ|={max_d:.2e} "
            f"off-by-one|Δ|={shift_d:.2e} (n={n}) | "
            f"grad_norm A={gna:.5g} B={gnb:.5g} ratio={gn_ratio:.4f} | "
            f"{'OK' if ok else 'FAIL'}"
        )
    # RL gradients must transform under CP exactly like CE (the validated
    # baseline): every loss's cp-ratio within 1% of CE's.
    if ce_gn_ratio and not math.isnan(ce_gn_ratio):
        for loss_fn, r in gn_ratios.items():
            if math.isnan(r) or abs(r / ce_gn_ratio - 1.0) > 0.01:
                print(f"  grad-ratio FAIL: {loss_fn} ratio={r:.4f} vs CE {ce_gn_ratio:.4f}")
                hard_fail = True
        print(f"  (all grad_norm cp-ratios track CE's ratio = {ce_gn_ratio:.4f})")
    print("PARITY:", "FAIL" if hard_fail else "PASS")
    return 1 if hard_fail else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    p.add_argument("--out", default="parity.json")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    run(args)


if __name__ == "__main__":
    main()
