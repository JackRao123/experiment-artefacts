"""Sample step-0 loss / grad_norm N times: /init_trainer_server, then one fb + optim on the seed datum."""
import argparse, random, time
import httpx
from profile_driver import BASE_URL, FB_TIMEOUT_S, OP_TIMEOUT_S, make_datum, submit_and_wait

ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=3); ap.add_argument("--seq-len", type=int, default=262144); ap.add_argument("--label", default="gn")
args = ap.parse_args()
with httpx.Client(base_url=BASE_URL, timeout=60.0) as c:
    for i in range(args.runs):
        submit_and_wait(c, "/init_trainer_server", {"lora_rank": 32, "lora_alpha": 32}, OP_TIMEOUT_S)
        datum = make_datum(random.Random(0xB300), args.seq_len)
        t0 = time.perf_counter()
        fb = submit_and_wait(c, "/forward_backward", {"data": [datum]}, FB_TIMEOUT_S)
        opt = submit_and_wait(c, "/optim_step", {"adam_params": {"learning_rate": 1e-5}}, OP_TIMEOUT_S)
        print(f"[{args.label} sample{i}] fb={time.perf_counter()-t0:.1f}s loss={fb.get('loss')} gn={(opt.get('metrics') or {}).get('grad_norm')}", flush=True)
