# GLM-5.3 main baseline

Completed on 2026-09-09, trainers main `33d19a3542c3d67553e9dc9d30381d2bdf6db31b`.
One node, eight HGX B300 GPUs, CP8/EP8/TP1/PP1, 262,144-token datum,
LoRA rank/alpha 32, native-FP8 experts, HybridEP and full-layer recomputation.

The fresh trainer completed one warmup, five unprofiled controls, one
memory-profiled step and one runtime-profiled step. Initial step 0, final step 8.

| Metric | Result |
|---|---:|
| Control forward/backward mean | 22.133 s |
| Control forward/backward sample SD | 1.190 s |
| Control throughput per GPU | 1,480.5 tokens/s |
| Aggregate control throughput | 11,844.1 tokens/s |
| Maximum control allocated memory across ranks | 191.959 GiB |
| Maximum control reserved memory across ranks | 194.277 GiB |
| Memory-profiled forward/backward | ~21.8 s |
| Runtime-profiled forward/backward | ~23.1 s |

All five controls are included. The first control was slower (~24.2 s) than
the remaining four (~21.3–21.9 s). The runtime and memory windows do not
contribute to control throughput. Exact values are in the raw benchmark JSON.

The stock GLM helper reports ~17.2% useful MFU against 2.25 PFLOP/s/GPU;
this inherits the existing GLM analytic FLOP model, not an independently
audited GLM-5.3 operation count. The measurement above is the direct TPS result.

Rank-0 Kineto capture is in `runtime/`; rank-0 allocator snapshot is in
`memory/`. `summary.json` records exact sizes and SHA-256 hashes. The
benchmark memory metrics above are distributed maxima, not rank-0-only peaks.

Startup took several minutes to read native-FP8 weights from the shared cache.
A stack dump confirmed safetensors loading. During warmup, Dynamo reported a
graph break at HybridEP `setup_metadata` / `max_num_tokens_across_ep.item()`.
The main source was left unchanged. Full startup/operation logs are preserved.

The trainer remains running at port 8001. To stop it on the devbox:

```bash
bash /root/glm53-main-262k-20260909/.devbox_up/stop_trainer.sh
```
