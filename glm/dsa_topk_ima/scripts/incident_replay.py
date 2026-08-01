#!/usr/bin/env python3
"""Exact-shape replay of the rwn24dw GLM-5.2 B300 incident session (2026-07-28).

Drives the historical operation sequence against a trainer over HTTP:

  step 1..N_WARMUP   warmup train steps (mirror the healthy 19:43-20:59 batches)
  crash step         ops shaped exactly like 6d8548d7 / 08cee77a / a369ffe9,
                     submitted pipelined 1s apart (qsize>=2), then optim_step,
                     then save_state
  post step          op shaped like 339c67dd with a pre-queued optim_step

Every request payload is serialized to disk with a SHA-256 manifest BEFORE
submission, so a crashing cycle is automatically frozen for replay. Datum
token content is synthesized per --data-mode and --data-seed; shapes follow
the incident aggregates (n_datums, total_tokens, min, max) recovered from
production logs (per-datum lengths were not logged; lengths are sampled to
match the aggregates exactly).

Historical facts encoded here (from Loki/ClickHouse recon 2026-07-29):
  - ops execute strictly serialized server-side; packing is deterministic,
    in submission order; DP=1 CP16 on 2x8 B300.
  - crash 1: op a369ffe9 (39 datums / 1,329,907 tok / 18,129..84,173),
    IMA ~24.5s into execution, rank 3.
  - crash 2: op 339c67dd (26 / 843,999 / 17,630..63,956), IMA ~33s in,
    rank 0, after optim_step + checkpoint save.
  - both crashed ops PASSED on byte-identical retry: failure is not
    shape-deterministic.
  - supervised fraction ~14.5% of tokens (448,460 loss tokens of 3,087,457).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

# --------------------------------------------------------------------------- #
# Historical op shapes
# --------------------------------------------------------------------------- #
# (name, n_datums, total_tokens, min_len, max_len)
CRASH_STEP_OPS = [
    ("6d8548d7", 11, 508_992, 28_355, 70_912),
    ("08cee77a", 26, 1_248_558, 33_621, 68_011),
    ("a369ffe9", 39, 1_329_907, 18_129, 84_173),  # crash 1 happened here
]
POST_STEP_OPS = [
    ("339c67dd", 26, 843_999, 17_630, 63_956),  # crash 2 happened here
]
# Healthy warmup steps before the crash step — EXACT aggregates of the
# 19:43 / 20:11 / 20:22 / 20:59 batches recovered from production logs.
WARMUP_STEPS: list[list[tuple[str, int, int, int, int]]] = [
    [("30710d28", 12, 770_829, 60_530, 73_616), ("035c4e71", 26, 1_277_053, 24_896, 74_163), ("1695c1fe", 27, 1_253_307, 25_829, 68_014)],
    [("eab2530c", 2, 100_933, 49_434, 51_499), ("9d8e096e", 30, 1_221_908, 27_150, 68_004), ("a724911d", 41, 1_373_306, 18_100, 62_031)],
    [("21f51b80", 28, 934_043, 17_895, 66_037)],
    [("29cc0dcb", 1, 64_063, 64_063, 64_063), ("b9da24a9", 29, 1_326_732, 25_686, 70_781), ("a3d6d0c2", 26, 1_374_941, 27_173, 66_994), ("e6ec2a4b", 30, 1_475_624, 27_200, 71_030), ("55485de3", 31, 1_273_306, 24_940, 65_303)],
]

LR = 1e-05
SUPERVISED_FRAC = 0.145


# --------------------------------------------------------------------------- #
# Length sampling: n lengths with exact total, containing min and max
# --------------------------------------------------------------------------- #
def sample_lengths(n: int, total: int, lo: int, hi: int, rng: random.Random) -> list[int]:
    if n == 1:
        assert lo <= total <= hi
        return [total]
    if n == 2:
        assert lo + hi <= total <= 2 * hi and total - lo <= hi
        return [total - hi, hi] if total - hi >= lo else [lo, total - lo]
    assert n >= 2 and lo <= hi and total >= lo * n
    for _ in range(10_000):
        mids = [int(rng.lognormvariate(0, 0.45) * ((total - lo - hi) / max(n - 2, 1))) for _ in range(n - 2)]
        mids = [max(lo, min(hi, m)) for m in mids]
        lens = [lo, hi] + mids
        drift = total - sum(lens)
        # spread the drift over the middle entries, clamped to [lo, hi]
        for _ in range(200):
            if drift == 0:
                break
            for i in range(2, len(lens)):
                if drift == 0:
                    break
                step = max(min(drift, hi - lens[i]), lo - lens[i])
                if abs(step) > abs(drift):
                    step = drift
                lens[i] += step
                drift -= step
        if drift == 0 and all(lo <= x <= hi for x in lens):
            rng.shuffle(lens)
            return lens
    raise RuntimeError(f"could not sample lengths n={n} total={total} lo={lo} hi={hi}")


# --------------------------------------------------------------------------- #
# Token synthesis
# --------------------------------------------------------------------------- #
class TokenSource:
    """Produces token id sequences of exact lengths in one of four modes."""

    def __init__(self, mode: str, seed: int, tokenizer=None, corpus_ids: list[int] | None = None):
        self.mode = mode
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.corpus_ids = corpus_ids or []
        self._cursor = self.rng.randrange(max(1, len(self.corpus_ids) - 1)) if self.corpus_ids else 0

    # -- chat mode helpers ---------------------------------------------------
    def _blob(self, kind: str, n: int) -> str:
        import base64 as b64

        r = self.rng
        if kind == "b64":
            return b64.b64encode(bytes(r.randrange(256) for _ in range(n))).decode()
        if kind == "hex":
            return "".join(f"{r.randrange(256):02x}" for _ in range(n))
        if kind == "ws":
            return "".join(r.choice([" ", "\t", "\n", "="]) for _ in range(n))
        if kind == "json":
            return json.dumps({f"key_{i}": {"status": "ok", "retries": 0, "data": "x" * r.randrange(2, 40)}
                               for i in range(n // 40 + 1)})
        return "x" * n

    def _chat_rounds(self, approx_tokens: int) -> list[dict]:
        r = self.rng
        msgs = [{"role": "system", "content": "You are bolt, an autonomous coding agent. Use tools precisely."}]
        # ~3.5 chars/token heuristic; build until we overshoot when tokenized
        budget_chars = int(approx_tokens * 3.6)
        used = 0
        while used < budget_chars:
            task = r.choice(["Fix the failing test", "Summarize this log", "Parse this payload",
                             "Refactor the module", "Analyze this trace"])
            blob_kind = r.choice(["b64", "hex", "ws", "json", "json", "b64"])
            blob = self._blob(blob_kind, r.randrange(400, 4000))
            user = f"{task}:\n```\n{blob}\n```"
            tool_json = json.dumps({"tool_call": {"name": r.choice(["run_tests", "read_file", "http_get"]),
                                                  "arguments": {"path": f"/src/mod_{r.randrange(99)}.py",
                                                                "payload": self._blob("b64", r.randrange(100, 1200))}}})
            result_blob = self._blob(r.choice(["json", "hex", "ws"]), r.randrange(400, 6000))
            assistant = (f"I'll handle that.\n<tool_call>{tool_json}</tool_call>\n"
                         f"Observation:\n```\n{result_blob}\n```\nDone. " + "Status: OK. " * r.randrange(1, 30))
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": assistant})
            used += len(user) + len(assistant)
        return msgs

    def sequence(self, length: int, vocab: int) -> list[int]:
        if self.mode == "chat":
            assert self.tokenizer is not None, "chat mode needs --tokenizer"
            ids: list[int] = []
            while len(ids) < length:
                msgs = self._chat_rounds(min(length - len(ids) + 512, 40_000))
                out = self.tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
                if hasattr(out, "get") and "input_ids" in out:
                    out = out["input_ids"]
                    if out and isinstance(out[0], list):
                        out = out[0]
                ids.extend(list(out))
            return ids[:length]
        return self._sequence_basic(length, vocab)

    def _sequence_basic(self, length: int, vocab: int) -> list[int]:
        if self.mode == "corpus" and self.corpus_ids:
            out = []
            while len(out) < length:
                take = min(length - len(out), len(self.corpus_ids) - self._cursor)
                out.extend(self.corpus_ids[self._cursor : self._cursor + take])
                self._cursor = (self._cursor + take) % max(1, len(self.corpus_ids) - 1)
                if self._cursor == 0:
                    self._cursor = self.rng.randrange(max(1, len(self.corpus_ids) // 2))
            return out[:length]
        if self.mode == "agent":
            # agent-transcript-like: repetitive structured blocks with bursts
            out: list[int] = []
            while len(out) < length:
                block_len = self.rng.choice([64, 128, 256, 512])
                base = self.rng.randrange(100, vocab - block_len - 1)
                block = [base + (i % 37) for i in range(block_len)]
                repeats = self.rng.choice([1, 1, 2, 4, 8])
                for _ in range(repeats):
                    out.extend(block)
                    if len(out) >= length:
                        break
                # burst of high-entropy separators
                out.extend(self.rng.randrange(100, vocab) for _ in range(self.rng.randrange(1, 32)))
            return out[:length]
        # random
        return [self.rng.randrange(100, vocab) for _ in range(length)]


def make_datum(tokens: list[int], sup_frac: float) -> dict:
    L = len(tokens)
    targets = [-100] * L
    sup = max(1, int(L * sup_frac))
    for i in range(L - sup - 1, L - 1):
        targets[i] = tokens[i + 1]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": targets, "dtype": "int64", "shape": [L]},
        },
    }


# --------------------------------------------------------------------------- #
# HTTP plumbing (submit async, poll; supports pipelined submits)
# --------------------------------------------------------------------------- #
class OpHandle:
    def __init__(self, name: str, operation_id: str, submitted_at: float):
        self.name = name
        self.operation_id = operation_id
        self.submitted_at = submitted_at
        self.result: dict | None = None
        self.error: str | None = None


def submit(client: httpx.Client, op_path: str, body: dict, name: str) -> OpHandle:
    key = uuid.uuid4().hex
    r = client.post(op_path, json=body, headers={"Idempotency-Key": key})
    if r.status_code != 202:
        raise RuntimeError(f"{name} submit failed {r.status_code}: {r.text[:500]}")
    return OpHandle(name, r.json()["operation_id"], time.time())


def wait(client: httpx.Client, h: OpHandle, timeout: float = 2400.0) -> OpHandle:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            rr = client.get(f"/operations/{h.operation_id}", timeout=40.0)
        except httpx.HTTPError as e:
            h.error = f"poll transport error: {e}"
            return h
        if rr.status_code == 408:
            continue
        if rr.status_code in (404, 502, 503):
            h.error = f"poll failed {rr.status_code}: {rr.text[:500]}"
            return h
        payload = rr.json()
        st = payload.get("status")
        if st == "done":
            h.result = payload["result"]
            return h
        if st == "error":
            h.error = str(payload.get("error", ""))[:6000]
            return h
    h.error = f"timeout after {timeout}s"
    return h


# --------------------------------------------------------------------------- #
# Payload freezing
# --------------------------------------------------------------------------- #
def freeze_payload(out_dir: Path, cycle: int, name: str, body: dict) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = out_dir / f"cycle{cycle:03d}_{name}.json"
    path.write_bytes(raw)
    lens = [len(d["model_input"]["chunks"][0]["tokens"]) for d in body["data"]]
    manifest = {
        "name": name,
        "cycle": cycle,
        "sha256": digest,
        "n_datums": len(lens),
        "total_tokens": sum(lens),
        "seq_lens": lens,
        "loss_fn": body.get("loss_fn"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / f"cycle{cycle:03d}_{name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return digest


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run_step(
    client: httpx.Client,
    ops: list[tuple[str, int, int, int, int]],
    *,
    cycle: int,
    src: TokenSource,
    vocab: int,
    shape_rng: random.Random,
    out_dir: Path,
    do_optim: bool,
    do_save: bool,
    save_name: str | None,
    log,
) -> None:
    """Build all op payloads, freeze, submit pipelined 1s apart, then wait in order."""
    bodies: list[tuple[str, dict]] = []
    for name, n, total, lo, hi in ops:
        lens = sample_lengths(n, total, lo, hi, shape_rng)
        data = [make_datum(src.sequence(L, vocab), SUPERVISED_FRAC) for L in lens]
        body = {"data": data, "loss_fn": "cross_entropy"}
        digest = freeze_payload(out_dir, cycle, name, body)
        log(f"frozen {name}: n={n} total={total} sha256={digest[:16]}...")
        bodies.append((name, body))

    handles: list[OpHandle] = []
    for i, (name, body) in enumerate(bodies):
        handles.append(submit(client, "/forward_backward", body, name))
        log(f"submitted {name} op={handles[-1].operation_id}")
        if i < len(bodies) - 1:
            time.sleep(1.0)

    for h in handles:
        wait(client, h)
        if h.error is not None:
            raise ReplayCrash(h.name, h.operation_id, h.error)
        log(f"done {h.name}: loss={h.result.get('loss'):.6f} ({time.time() - h.submitted_at:.1f}s)")

    if do_optim:
        oh = submit(client, "/optim_step", {"adam_params": {"learning_rate": LR, "beta1": 0.9, "beta2": 0.95}}, "optim")
        wait(client, oh)
        if oh.error is not None:
            raise ReplayCrash("optim_step", oh.operation_id, oh.error)
        log(f"optim_step done ({time.time() - oh.submitted_at:.1f}s)")
    if do_save and save_name:
        try:
            sh = submit(client, "/save_state", {"name": save_name, "run_id": "dsa-ab-replay"}, "save")
            wait(client, sh)
            if sh.error is not None:
                log(f"save_state failed (continuing): {sh.error[:300]}")
            else:
                log(f"save_state done ({time.time() - sh.submitted_at:.1f}s)")
        except RuntimeError as e:
            log(f"save_state submit failed (continuing): {e}")


class ReplayCrash(Exception):
    def __init__(self, op_name: str, operation_id: str, error: str):
        super().__init__(f"{op_name} ({operation_id}): {error[:2000]}")
        self.op_name = op_name
        self.operation_id = operation_id
        self.error = error


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainer-url", default="http://127.0.0.1:8001")
    p.add_argument("--out-dir", required=True, help="payload freeze directory")
    p.add_argument("--data-mode", choices=["corpus", "agent", "random", "chat"], default="agent")
    p.add_argument("--corpus-file", default=None, help="plain-text file for corpus mode")
    p.add_argument("--corpus-tokens-file", default=None,
                   help="JSON file with a pre-tokenized id pool for corpus mode (overrides --corpus-file)")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer path for corpus mode")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--shape-seed", type=int, default=0, help="seed for per-datum length sampling")
    p.add_argument("--vocab", type=int, default=150000, help="max synthetic token id (GLM-5.2 vocab ~154k)")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=4, help="healthy train steps before the crash step (history: 4)")
    p.add_argument("--warmup-cycles", type=int, default=1,
                   help="run warmup steps only on the first N cycles; later cycles go straight to the crash+post steps")
    p.add_argument("--skip-save", action="store_true")
    p.add_argument("--tag", default="replay")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    logf = open(out_root / f"{args.tag}.log", "a", buffering=1)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    tokenizer = None
    corpus_ids: list[int] | None = None
    if args.data_mode in ("corpus", "chat"):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if args.data_mode == "corpus":
        if args.corpus_tokens_file:
            corpus_ids = json.loads(Path(args.corpus_tokens_file).read_text())
            log(f"corpus ids loaded: {len(corpus_ids)} tokens from {args.corpus_tokens_file}")
        else:
            text = Path(args.corpus_file).read_text(errors="ignore")
            corpus_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            log(f"corpus tokenized: {len(corpus_ids)} tokens from {args.corpus_file}")

    with httpx.Client(base_url=args.trainer_url, timeout=2400.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        status = client.get("/status", timeout=30.0).json()
        log(f"trainer status: step={status.get('step')} world={status.get('world_size')} "
            f"CP={status.get('context_parallel_size')} EP={status.get('expert_parallel_size')}")

        for cycle in range(args.cycles):
            cyc_dir = out_root / f"{args.tag}_cycle{cycle:03d}"
            src = TokenSource(args.data_mode, args.data_seed + cycle, tokenizer, corpus_ids)
            shape_rng = random.Random(args.shape_seed + cycle)
            log(f"=== cycle {cycle} data-mode={args.data_mode} data-seed={args.data_seed + cycle} "
                f"shape-seed={args.shape_seed + cycle} ===")
            try:
                n_warm = args.warmup_steps if cycle < args.warmup_cycles else 0
                for wi in range(n_warm):
                    ws = WARMUP_STEPS[wi % len(WARMUP_STEPS)]
                    log(f"--- warmup step {wi + 1}/{n_warm} ({len(ws)} ops) ---")
                    run_step(client, ws, cycle=cycle, src=src, vocab=args.vocab, shape_rng=shape_rng,
                             out_dir=cyc_dir, do_optim=True, do_save=False, save_name=None, log=log)
                log("--- crash step (6d85/08ce/a369 shapes) ---")
                run_step(client, CRASH_STEP_OPS, cycle=cycle, src=src, vocab=args.vocab, shape_rng=shape_rng,
                         out_dir=cyc_dir, do_optim=True, do_save=not args.skip_save,
                         save_name=f"{args.tag}-c{cycle}", log=log)
                log("--- post step (339c shape) ---")
                run_step(client, POST_STEP_OPS, cycle=cycle, src=src, vocab=args.vocab, shape_rng=shape_rng,
                         out_dir=cyc_dir, do_optim=True, do_save=False, save_name=None, log=log)
                log(f"=== cycle {cycle} COMPLETED CLEAN ===")
            except ReplayCrash as e:
                log(f"!!! CYCLE {cycle} CRASHED: op={e.op_name} id={e.operation_id}")
                log(f"!!! error: {e.error[:3000]}")
                log(f"!!! frozen payloads preserved in {cyc_dir}")
                sys.exit(42)
            except httpx.HTTPError as e:
                log(f"!!! CYCLE {cycle} TRANSPORT FAILURE (server likely dead): {e}")
                log(f"!!! frozen payloads preserved in {cyc_dir}")
                sys.exit(43)

    log("all cycles completed without crash")
    sys.exit(0)


if __name__ == "__main__":
    main()
