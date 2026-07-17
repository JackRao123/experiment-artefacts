# Historical note: Nemotron-3-Ultra RL constraints

> This is a record of past experiments, not an active runbook. Details may be
> outdated as model images, SDKs, and devbox provisioning evolve. Use
> `devbox-up` for all current trainer and sampler lifecycle operations.

## Topology and weight sync

- An 8-GPU TP8 sampler needs a free node. Do not place it with a trainer that
  consumes all eight GPUs on that node.
- Past local trainer-to-sampler experiments used shared-FS
  `weight_sync: {"type":"local","path":"..."}` and a manifest whose
  `bt_weight_sync_uri` named the same shared path. `/b10/workspace` was
  node-local and could not be used.

## Ultra BF16 sampler exception

The production B200 entry served NVFP4 and supported its golden `max_loras: 4`.
When deliberately serving the much larger **BF16** Ultra checkpoint locally,
`max_loras: 1` avoided model-load OOMs. This was a local BF16 workaround; it
was not a reason to change the production golden configuration.

## RL experiment observations

- `max_seq_len: 2048` was appropriate for learning runs.
- 131k was a memory/performance-envelope test, not a learning test. With PP,
  padded microbatches made short examples expensive and small batches often
  became degenerate.
- A real batch such as batch 8 × group 8 produced usable signals. A healthy
  step had finite loss and nonzero datums; 100% degenerate groups performed no
  optimization.
- GSM8K could be saturated by the model, producing zero-advantage groups.
  Harder datasets such as DeepScaleR or Big-Math were more informative when
  available.
- Sampler policy version / `min_pv` advancing after saves was used as an
  indicator that weight sync worked.
- A dozen small LoRA GRPO steps demonstrated plumbing, not convergence.
