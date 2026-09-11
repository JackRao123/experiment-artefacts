# Full-model startup diagnostics

The first full-model MFSDP attempt was still inside construction after roughly
15 minutes. A live Python stack caught rank0 at the explicit gc.collect call
in param_and_grad_buffer.py. The implementation called gc.collect and
torch.cuda.empty_cache before each module whenever GPU usage exceeded 50%.
That threshold stays true for a model whose persistent shards alone use
approximately 173 GiB of a 268.6-GiB device.

The retained 10-second py-spy sampling window contains 388 samples and no
sampling errors: 151 top frames are empty_cache and 232 are all_gather while
constructing distributed parameters. The sampler warned that it fell behind,
so these are diagnostic sample counts, not a precise whole-startup time split.
The initial stack and source establish the repeated collection path; the
sampled window also includes subsequent initialization collectives.

Core c5de29803ca1ac8caf606a72a3300115c1b0fda0 replaces the utilization threshold
with an allocation-demand check: reclaim only if the next module's parameter
storage exceeds driver-free plus reusable allocator-cache bytes. The normal
allocator can still reuse or release cached memory. Automatic Python GC is
not disabled by this change.

After restarting, FSDP construction completed in 330.66–333.21 seconds across
the eight ranks. The checkpoint import is timed separately and is still in
progress. This is not a completed training benchmark or an exact end-to-end
startup speedup, since the earlier attempt was interrupted before completion.

A later stack also exposed repeated whole-model parameter counting in FSDP's
root-hook registration. Core e6c86ea3b75674dfff3769c7c92141629864dc0c caches that
unchanging count. This additional startup-only change has been committed but
is NOT deployed in the current CP8 attempt; it may be used on the next restart.

Files: startup-rank0.speedscope.json, cp8ep1/attempt1-startup-gc.trainer.log,
source-pins-before-startup-fix.json and source-pins.json. No tests changed.
