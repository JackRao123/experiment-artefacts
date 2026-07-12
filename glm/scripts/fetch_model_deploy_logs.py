#!/usr/bin/env python3
"""Fetch all logs for a Baseten model deployment (management API).

Same pagination approach as loops-quickstart/training/util/fetch_loops_logs.py,
pointed at /v1/models/{model_id}/deployments/{deployment_id}/logs.

Usage:
  python3 fetch_model_deploy_logs.py MODEL_ID DEPLOYMENT_ID START_ISO END_ISO OUT_PATH
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from typing import Any

API_URL = "https://api.baseten.co"
MAX_PAGE_SIZE = 1_000


def parse_time(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000)


def log_epoch_ms(log: dict[str, Any]) -> int:
    return int(log["timestamp"]) // 1_000_000


def log_identity(log: dict[str, Any]) -> tuple[Any, ...]:
    return (
        log.get("timestamp"),
        log.get("message"),
        log.get("replica"),
        log.get("request_id"),
        log.get("level"),
    )


def request_page(
    model_id: str, deployment_id: str, api_key: str, start_ms: int, end_ms: int
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "start_epoch_millis": start_ms,
            "end_epoch_millis": end_ms,
            "direction": "asc",
            "limit": MAX_PAGE_SIZE,
        }
    )
    request = urllib.request.Request(
        f"{API_URL}/v1/models/{model_id}/deployments/{deployment_id}/logs?{query}",
        headers={"Authorization": f"Api-Key {api_key}"},
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)["logs"]
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_all(model_id: str, deployment_id: str, api_key: str, start_ms: int, end_ms: int):
    logs: list[dict[str, Any]] = []
    cursor_ms = start_ms
    seen_at_cursor: Counter[tuple[Any, ...]] = Counter()

    while cursor_ms <= end_ms:
        page = request_page(model_id, deployment_id, api_key, cursor_ms, end_ms)
        if not page:
            break
        skip = seen_at_cursor.copy()
        new_logs = []
        for log in page:
            identity = log_identity(log)
            if log_epoch_ms(log) == cursor_ms and skip[identity]:
                skip[identity] -= 1
                continue
            new_logs.append(log)
        logs.extend(new_logs)
        if len(page) < MAX_PAGE_SIZE:
            break
        last_ms = log_epoch_ms(page[-1])
        if last_ms == cursor_ms:
            seen_at_cursor.update(
                log_identity(l) for l in new_logs if log_epoch_ms(l) == cursor_ms
            )
        else:
            cursor_ms = last_ms
            seen_at_cursor = Counter(
                log_identity(l) for l in page if log_epoch_ms(l) == cursor_ms
            )
    logs.sort(key=lambda l: int(l["timestamp"]))
    return logs


def main() -> int:
    model_id, deployment_id, start_iso, end_iso, out_path = sys.argv[1:6]
    sys.path.insert(
        0,
        os.path.expanduser("~/Documents/loops-quickstart/training/util"),
    )
    from truss_auth import ensure_baseten_api_key

    ensure_baseten_api_key()
    logs = fetch_all(
        model_id,
        deployment_id,
        os.environ["BASETEN_API_KEY"],
        parse_time(start_iso),
        parse_time(end_iso),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, sort_keys=True) + "\n")
    print(f"lines={len(logs)} path={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
