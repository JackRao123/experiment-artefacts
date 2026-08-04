# cudnn-frontend develop-pin migration — validation record (trainers PR #910)

2026-08-03, B300 devbox b300-1-z7sxjdpi-0020 (tj-q8x5ky3), venv
trainers_main/server/.venv (torch 2.11+cu130, nvidia-cutlass-dsl 4.5.2),
CUDA_DEVICE_MAX_CONNECTIONS unset (prod default). Wheels compared:

- **pristine** = PyPI nvidia-cudnn-frontend 1.26.0 (sha256 1a1223c4…)
- **dev** = built from source at NVIDIA/cudnn-frontend develop
  @ 74785165de2da954a2c879a5e3e6f95411c2292d → 1.27.0+git7478516
  (the wheel committed in PR #910)
- **ctrl** = 1.26.0+dsatopk5 (the venv's installed wheel, our retired patch
  stack) — control arm for soak + parity attribution

Overlays were installed with `pip install --target` and selected via
PYTHONPATH so the venv stayed untouched.

## Files

- `build.log`, `build2.log` — wheel builds at the pin (untagged, then
  +git7478516 version-tagged). cmake/ninja from-source build, ~2 min, no GPU.
- `ab_pristine.log`, `ab_dev.log` — A/B race probe (lps1003_review
  probe_race.py, 12 reps x 5 geometries, double-exec + empty_cache per rep).
  Pristine: 12/12 disagreements at the 4 GiB out geometry, all with the
  erasure signature; 0/12 at ≤2.00 GiB (= 2^31 B exactly, sub-threshold).
  Dev: 0/12 everywhere.
- `soak_dev.log`, `soak_ctrl.log` — phase A of soak_identical_fwd.py:
  100 reps kernel-level indexer_fwd at the 4 GiB race geometry, empty_cache
  each rep, double-exec + bitwise-vs-rep0. 0/100 bad reps both arms.
  (Phase B errored in these runs — wrong import guess, fixed and re-run.)
- `soakB_dev.log`, `soakB_ctrl.log` — phase B: 100 reps fused absorbed-MLA
  DSA module forward (cudnn backend, THD seqlen 2048), identical input,
  bitwise-vs-rep0. 0/100 bad reps both arms.
- `parity_dev.log` — first parity run (-x): 31 passed, 8 skipped
  (tilelang unavailable), stopped at
  test_cudnn_indexer_topk_multi_packed_cp_uses_segmented_thd.
- `parity_dev_full.log`, `parity_ctrl_full.log` — full
  test_dsa_native_parity.py + test_attention_variant_dsa.py on dev and ctrl
  wheels: 169 passed / 15 failed on BOTH arms, failure sets identical
  (pre-existing at trainers_main checkout 0e0b65a: FakeDSA monkeypatch tests
  + TP tests; none touch the real wheel). All real-kernel cudnn fused
  absorbed-MLA fwd+bwd parity variants pass on the dev wheel.
- `soak_identical_fwd.py` — the soak harness (phases A/B, reps + phase
  selection via argv).
- `test_launch_stream_new.py` — the rewritten regression test as validated
  on-box against the dev wheel (3/3 passed); canonical copy lives at
  server/tests/unit/dp_worker/test_cudnn_dsa_indexer_launch_stream.py.

The repo regression suites (test_cudnn_dsa_indexer_launch_stream.py +
test_cudnn_dsa_indexer_topk.py) were run interactively: 4 passed + 1
expected failure (the old arrangement test asserting the +dsatopk5 source
text) on the dev wheel; the bitwise cold-start test FAILS on pristine in
31 s (positive control) and passes on dev.
