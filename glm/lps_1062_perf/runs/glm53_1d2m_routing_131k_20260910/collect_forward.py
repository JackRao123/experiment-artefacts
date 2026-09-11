"""Submit one fixed-input training forward/backward; hooks save original forward only."""
import hashlib
import json
import random
import time
import uuid
from pathlib import Path

import httpx

root = Path(__file__).parent
rng = random.Random(0xB300)
tokens = [rng.randrange(154880) for _ in range(131072)]
payload = {"data": [{"model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]}, "loss_fn_inputs": {}}]}
input_hash = hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
with httpx.Client(base_url="http://127.0.0.1:8001") as client:
    response = client.post("/forward_backward", json=payload, headers={"Idempotency-Key": uuid.uuid4().hex}, timeout=60)
    response.raise_for_status()
    operation = response.json()["operation_id"]
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        response = client.get(f"/operations/{operation}", timeout=60)
        if response.status_code == 408:
            continue
        response.raise_for_status()
        body = response.json()
        if body.get("status") == "error":
            raise RuntimeError(body)
        if body.get("status") == "done":
            result = {"seed": "0xB300", "sequence_length": len(tokens), "input_sha256": input_hash, "result": body["result"]}
            (root / "forward-result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({"status": "done", "sequence_length": len(tokens), "input_sha256": input_hash}), flush=True)
            break
    else:
        raise TimeoutError(operation)
