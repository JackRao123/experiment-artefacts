Experimental source dependency for basetenlabs/trainers#1355.

- Keep proven plain-causal CP1 DSA indexer chunks on the cuDNN path.
- Skip identity source/expert chunk sorts for EP=ETP=1.
- Unwrap the inner Megatron FSDP wrapper correctly.
- Avoid repeated GC/cache flushing and repeated root parameter counting at initialization.
- Add fsdp_param_gather_prefetch to reproduce the no-lookahead memory experiment
  without replacing runtime methods.
- Add opt-in moe_use_torch_grouped_mm and its frozen BF16, zero-copy expert
  implementation. Backward rebuilds the view from currently unsharded weights.

All implementations are committed library code. Configs, traces and reports
are kept in the separate experiment-artefacts repository, not trainers.

No tests added or modified, per the researcher's instruction. Changed files
pass formatting/import sorting, lint and Python 3.12 compilation checks.
Clean-branch trainer reproduction passed standard/grouped debug CP8,
no-prefetch debug CP4, and full GLM-5.3 CP8 at 131072 tokens. Draft research work;
full save/load and multi-adapter support are not claimed.
