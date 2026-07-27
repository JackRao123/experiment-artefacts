# 256k profile data

Summaries and driver logs copied from devbox `w64072q`.

- `fresh_baseline_262144`: direct 256k fwd/bwd from a fresh boot.
- `nvls_off_262144`: same probe with `NCCL_NVLS_ENABLE=0`.
- `nccl_1ch_262144`: same probe with `NCCL_MAX_NCHANNELS=1`.
- `baseline_262144_optim2`: first two 256k SFT steps including Adam updates.
- `baseline_262144_optim3`: third step confirming steady-state memory.
- `torch_snapshot_196608`: allocation attribution from the 196k torch snapshot.

The raw 20 ms NVML CSVs and 50 MiB torch snapshot remain on shared devbox
storage under `/root/.cache/user_artifacts/glm256k/profile/results/`.
