"""Run one steady-state forward/backward and optimizer step for nsys."""

from argparse import Namespace

import httpx
from profile_driver import BASE_URL, drive_window, make_datums
import random


args = Namespace(seq_len=131_072, datums=1)
with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
    record = drive_window(
        client,
        "glm52-1d1m-nsys",
        0,
        make_datums(random.Random(0xB300), args),
        args.seq_len,
        8,
        "capture",
    )
    print(record, flush=True)
