# Optional metrics capture failed

The three warmups, five unprofiled controls and three timing-profiled FBs
completed. The subsequent, separate GPU-metrics FB stalled. Do not use that
partial metrics report for utilization, timing, or throughput conclusions.

Observed: GPU 4 idle, peers occupied at low power; PID 59428 (rank 4, verified
against NVTX rank markers) blocked in `_coalescing_manager.__exit__` during
FSDP parameter gather in checkpoint recompute. Rank 0 waited on an expert
metadata DtoH synchronization. Python stack dumps were preserved on the pod.

Cause is not established. This may be a profiler-triggered dependency issue
or an underlying scheduling race. CUDA event tracing was enabled; nsys help
explicitly warns it can introduce false dependencies across streams. Future
v2 captures disable that feature and have bounded profiled request deadlines.

The failed pass was interrupted, Nsight collection stopped, and the exact
run-owned trainer processes stopped through the adapted lifecycle. Prior
complete timing and control artifacts are retained.
