# Raw logs

This folder holds stdout/stderr from the selected long NCCL P2P runs on
`tj-w5ylgj3`. The parent report contains the topology commands and condensed
results. Verbose per-rank NCCL debug logs remain in the devbox artifact cache
under `/root/.cache/user_artifacts/devboxes/w5ylgj3/network_bw/`.

The `*.log` files are intentionally gitignored under the artifact repository's
large-data policy; they remain in Jack's local artifact tree.
