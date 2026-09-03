"""Drive one steady-state forward/backward and optimizer step."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import httpx
from profile_driver import BASE_URL, drive_window, make_datums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    datum_args = argparse.Namespace(seq_len=args.seq_len, datums=1)
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        record = drive_window(
            client,
            args.label,
            0,
            make_datums(random.Random(0xB300), datum_args),
            args.seq_len,
            args.num_gpus,
            "capture",
        )
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
