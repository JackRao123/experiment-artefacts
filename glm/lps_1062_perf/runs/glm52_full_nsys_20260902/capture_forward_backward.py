"""Run one steady-state forward/backward operation for an nsys capture."""

import json
import random
import time
from argparse import Namespace
from pathlib import Path

import httpx

from profile_driver import BASE_URL, FB_TIMEOUT_S, make_datums, submit_and_wait


args = Namespace(seq_len=131_072, datums=1)
data = make_datums(random.Random(0xB300), args)
started = time.perf_counter()
with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
    result = submit_and_wait(
        client,
        "/forward_backward",
        {"data": data},
        FB_TIMEOUT_S,
    )
elapsed = time.perf_counter() - started
record = {
    "seq_len": args.seq_len,
    "datums": args.datums,
    "num_gpus": 8,
    "elapsed_seconds": elapsed,
    "tokens_per_second_per_gpu": args.seq_len / elapsed / 8,
    "result": result,
}
Path(__file__).with_name("capture_result.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2), flush=True)
