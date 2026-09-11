# Original-forward routing capture

GLM-5.3 debug 1d2m, CP8EP1 and CP8EP8, TP1/PP1/ETP1, BF16 expert
storage, LoRA32, full one-layer activation recomputation. FSDP is disabled
in both configurations, matching the earlier CP/EP comparison.

Each configuration receives one identical synthetic 131,072-token datum
(seed `0xB300`) after the standard 64-token server startup warmup. The driver
sends a forward/backward request but does not take an optimizer step.

`BT_ROUTING_COUNTS_DIR` enables the implementation committed to trainers
PR #1355, commit `815fe3ed7a14677c6c76cbcf27506de059e1ee1c`.
Bridge remains `60b1570fb`, Core remains `cf81782b2`.
The hooks capture only while the original `GPTModel.forward` is executing;
layer recomputation in backward is outside that scope. No grad-mode heuristic
is used. Startup captures are excluded from `counts.csv`.

- `router`: assignments selected by each rank before dispatch, all 256 experts.
- `expert_input`: tokens presented to each rank's hosted experts after dispatch.
- `analyze.py`: requires all eight ranks, one original model forward, two MoE
  layers, excluded recompute hooks, and exact routing/dispatch conservation.
- `routing_counts.ipynb`: visualization only; `counts.csv` holds exact values.
- `visualize.py`: shared plotting helper; `render_notebook.py` executes and
  adds static fallback images alongside interactive Plotly output.

Collection runs on `jackrao-profiler-b300` under
`/root/glm53-1d2m-routing-131k-20260910`, using generated devbox lifecycle
scripts. No trainer was launched manually. Original `tools/profile_driver.py`
and `tools/mfu.py` are unchanged. These instrumented requests are not throughput
benchmarks.

Both captures completed and passed every conservation check in
`validation.json`. Each rank recorded one original forward and ignored four
recompute hook invocations. The CSV contains 12,800 exact counts. The executed
notebook was visually checked and opened in Cursor. Trainer processes
were stopped after collection; the profiler pod remains available.

The local `GLM routing (visualization)` Jupyter kernel uses this run's `.venv`;
its package versions are in `requirements-visualization.txt`.

After user feedback, the notebook now contains exactly eight per-expert bar
plots: (CP8EP1, CP8EP8) × (rank 0, rank 1) × (MoE 1, MoE 2). The x-axis is
global expert ID and the y-axis is that expert's raw token count on that rank.
No ranks are pooled and no percentages are used. The earlier heatmaps and
pooled histograms are no longer displayed. The user's Plotly-install line
was preserved. Plot arrays are plain lists for renderer compatibility.
