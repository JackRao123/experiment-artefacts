# FP32 LM-head memory experiment

The accepted candidate preserves functional addition and fuses its FP32
conversion, and fuses the high/residual gradient split. It is opt-in through
BT_MEMORY_EFFICIENT_LM_HEAD=1; the default behavior is unchanged.

All output tensors and hidden/LoRA-A/LoRA-B gradients were bitwise equal in
the numerical probes: BF16 17x64x97, FP16 65x256x1024, and BF16
4096x6144x154880 (M,K,N). The rounding-sensitive high/residual split was also
checked directly.

At the full logits-chunk shape, incremental forward/backward peak fell from
8.328 to 5.908 GiB. Small-shape peaks include compiler/workspace overhead and
are not evidence of a memory benefit for small requests. Normal training
measurements must still use warmed controls.

Rejected attempts are retained:

- Default compiler precision widening did not preserve the BF16/FP16
  downcast-upcast required by the residual. It failed the bitwise check and
  was never deployed. The accepted functions set emulate_precision_casts=True.
- In-place addition on the reshaped logits passed numerics but increased the
  full-size peak to 9.453 GiB. It was replaced by functional fusion and was
  never deployed to a trainer.

Files: probe.py, baseline.py, candidate.py, probe.log,
probe-default-rounding-failed.log, probe-inplace-higher-peak.log. No test-suite
files were added or modified.
