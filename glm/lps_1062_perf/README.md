# LPS-1062 — GLM-5.2 B300 throughput optimization

## Layout

- `NOTEBOOK.md` — chronological lab notebook across all experiments.
- `REPORT.md` — headline report from the initial optimization campaign.
- `DATA.md` — manifest and retention policy for large local artifacts.
- `B300_HOST_OFFLOAD_BANDWIDTH.md` — host-memory and NUMA bandwidth findings.
- `tools/` — canonical reusable benchmark, profiling, memory, and bandwidth
  utilities. This is the current experiment machinery.
- `runs/` — one self-contained folder per experiment or campaign, containing
  its configs, results, reports, patches, and copies of the scripts as used.
  This is the experiment history and evidence.
- `pp2cp8ep8/` — focused PP2/CP8/EP8 activation-placement workstream.
- `glm-5.2-moe-layer/` — editable architecture diagrams and rendered outputs.
- `*.original.sh` and `*.bundle` — frozen launch-script and branch snapshots;
  not normal entry points.
