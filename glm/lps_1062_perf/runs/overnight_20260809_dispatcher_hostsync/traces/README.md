# Oversized traces — not in git

Three kineto traces exceed GitHub's 100 MB limit and are gitignored by explicit
path. Canonical copies live on Jack's Mac at this repo path; round-3 traces
(arm-cprime, arm2-w1, anchor, W3 canary) live under `~/perf_profiles/lps-1062/round3/`.

| file | size | md5 |
|---|---:|---|
| patched-ABF-4mb131k.pt.trace.json | 617M | 9045359766bf08af3089c84f947cf4ba |
| unpatched-4mb-steady_qr4ggv3-gatesoff-step3.pt.trace.json | 1.9G | e5bbe1770d0dee87d5a6501f10904a32 |
| gated-v2-4mb-steady_qr4ggv3-step8.pt.trace.json | 676M | 4c8f2038681dc0938bcf026776a72850 |

Decision (Jack, 2026-08-10): keep local on the Mac, gitignored — no upload.
Full inventory of all Mac-only large artifacts (these + `~/perf_profiles/lps-1062/`):
`DATA.md` at the lps_1062_perf root.
