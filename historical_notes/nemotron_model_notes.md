# Historical note: Nemotron-3 trainer and sampler observations

> This is a record of past experiments, not an active runbook. Details may be
> outdated as model images, registries, and devbox provisioning evolve. Use
> `devbox-up` for all current trainer and sampler lifecycle operations.

## Trainer observations

- In past cached-weight runs, both `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` were needed to prevent `trust_remote_code` from
  triggering an expensive Hub fetch.
- Shared artifacts belonged in `user_artifacts` or `team_artifacts`; using
  `/b10/workspace` caused cross-node visibility failures.
- With pipeline parallelism, forward/backward microbatches padded to
  `max_seq_len`. Changing the configured maximum context required a restart to
  test meaningfully different sequence lengths.
- The first forward could be much slower than steady state due to kernel
  autotuning.
- A distributed OOM could wedge a request rather than cleanly terminating all
  ranks.

## Sampler observations

- A local mounted or cached `MODEL_PATH` better matched production-like
  validation than an HF model ID, which exercised a different download path.
- The sampler received golden runtime configuration through
  `OPAQUE_SAMPLER_PAYLOAD`; omitted fields fell back to vLLM defaults.
- `ENFORCE_EAGER=false` was the throughput-oriented setting. Enabling eager
  mode helped boot diagnostics but disabled CUDA graphs.
- A measured Mamba-based Ultra setup needed `MAX_NUM_SEQS <= 567`; larger
  values could fail CUDA-graph capture.

## Ultra BF16 exception

The production Ultra sampler served NVFP4. In an older local BF16 validation,
the sampler needed `max_loras: 1` rather than the NVFP4 golden value of four
to avoid an OOM during model load.
