# NCCL ENV RE-SWEEP PLAN — lever 6 (bohr, 2026-08-13 ~20:0x PDT)

Design-only, Mac-side. Opportunistic-window material; rides the B/F d4 window if
fermi approves the boot-sharing in §4.

## 1. The ship env block and what each knob was buying (Aug-7, EP16/CP16 @256k, RoCE)

Standing block (on every run since exp06, incl. the fe127 number-of-record runs —
per BRIEF_DOPPLER §6):

```
NCCL_IB_QPS_PER_CONNECTION=8  NCCL_IB_SPLIT_DATA_ON_QPS=1  NCCL_NCHANNELS_PER_NET_PEER=8
```

Aug-7 evidence (NOTEBOOK.md exp05 ladder, anchor 416.5 tok/s/GPU, 6×400 Gb
LAG-bonded RoCE per node, NCCL 2.28.9):

| knob | run | result | what it was buying |
|---|---|---|---|
| `IB_QPS_PER_CONNECTION=4` + `IB_SPLIT_DATA_ON_QPS=1` | exp05a | 464 (+11.4%) | LAG bonds hash flows **per-QP** — multiple QPs spread each peer connection across the bond slaves; without it each connection pins to one slave |
| QPS=8 + `NCHANNELS_PER_NET_PEER=4` | exp05b | 559 (+34% cum.) | more parallel network streams per peer → higher fabric utilization for the **cross-node EP a2a** |
| `NCHANNELS_PER_NET_PEER=8` | exp05c | 583 (+40%) | channel scaling, already flattening (+4% for 4→8) |
| chan=16 + MIN/MAX_NCHANNELS | exp05d | 597 (+43%) | rejected for ship: +3.5 GiB NCCL buffers cost OOM margin |
| `SPLIT_DATA_ON_QPS=0` | exp05e | 603 ≈ 597 | split knob **immaterial at these sizes** even on the old topology |

Mechanism (exp05d trace): cross-node a2a SendRecv 44.9→24.5 s, p50 34→17.8 ms —
QP/channel spreading ~halved a2a time on the bond fabric. The lever was rich
because EP16/CP16 put the dominant collective (EP a2a) **on the wire**.

## 2. What each knob touches now (PP2/CP8/EP8 @131k — a2a on NVLink, RoCE carries PP p2p only)

Hardware-verified topology (group dump + NCCL_DEBUG=INFO on Run B): EP and CP
groups are **intra-node** ({0-7}/{8-15}, NVLink, "via P2P/CUMEM" ×448); the only
cross-node collective path is **PP p2p** ({r,r+8}, "via NET/IB/GDRDMA" ×192).

All three ship knobs are **NET/IB-scoped** (QPs are an RDMA construct;
`NCHANNELS_PER_NET_PEER` is per *network* peer) ⇒ on this topology they touch
only the PP p2p path. That path is **wire-speed and park-dominated**: 200 MB in
4.2 ms = 48 GB/s = line rate; the 8.1 s p2p cost is schedule parking, not
transfer. Measured confirmation: Run B (ship env on vs off, clean-tip d4,
pre-M=N) = **552 vs 550 = +0.4% = inert**.

Fixed-wheel d16 exposure map (erdos, fe127_d16_rank0): exposed EP a2a 34.3 s
(intra-node, **wait-dominated** — microbench: 600 GB/s/rank vs 30.5 ms avg calls
≈ 50× over bandwidth bound; imbalance wait, no NCCL knob touches it), pure idle
20.8 s (kepler's lane), exposed CP collectives 7.5 s (AllGather 5.6 + RS 1.9,
intra-node NVLink), compute 67.1 s.

**Per-knob verdict: all three ship knobs are INERT for every current exposure
class.** Keep them in the ship package regardless (load-bearing +51% for golden
EP16/CP16 prod; harmless here) unless Arm 1 says they hurt.

## 3. Ranked arms (≤4; bars sized to the 3–4% d4 noise floor)

### Arm 1 — ship-env ablation (null-hypothesis control). Rank 1.
- **Question:** does the ship env do *anything* on PP2/CP8/EP8 post-M=N, fixed
  wheel? (Run B's +0.4% predates M=N; the cross-node share has only shrunk.)
- **Hypothesis:** Δ ≈ 0 (mechanism §2).
- **Design:** d4, off-arm boot ×2 runs vs on-arm boot ×2 runs (env is
  launch-bound ⇒ cross-boot; x2/x2 bounds the boot term). Reference 878–886.
- **Accept/reject:** |Δmean| ≤ max(arm spreads) → INERT confirmed → document
  "env-neutral on PP2/CP8/EP8" in the ship package (one less thing to validate).
  Δ < −3% beyond spread → env *hurts* → drop from the PP2 ship env. Δ > +3%
  beyond spread → contradicts Run B + mechanism → investigate before believing.
- **Cost:** folds into the B/F window (§4). Certain information, tiny cost.

### Arm 2 — CP-collective intra-node tuning. **KILLED AT THE GATE (2026-08-13 ~20:5x PDT), zero box time spent.**

Gate executed (bohr; trace re-served locally, `query -f` path after this
build's HTTP server 404s `/query` — papercut-worthy but worked around).
Per-call payload × duration from the kernel slices' collective metadata
(`args.Out msg nelems`), fe127_d16_rank0, bf16, group 8, ring floor =
0.875 × out_bytes / 600 GB/s (microbench-measured NVLink rate):

| class | payload | n | avg | min | max | transfer floor | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| AllGather_RING_LL (KV) | 75.5M elem = 151 MB | 1216 | 4.13 ms | **0.225 ms** | 172 ms | ≈0.22 ms | **wait-dominated: min AT floor, avg 18× floor** |
| AllGather_RING_LL (ctrl) | 2 KB | 1120 | 0.26 ms | 0.010 ms | 48 ms | ~0 | latency/wait |
| AllGather_RING_LL | 16.8M elem = 33.6 MB | 352 | 0.76 ms | 0.078 ms | 95 ms | ≈0.05 ms | wait-dominated (15×) |
| ReduceScatter_RING_LL | 9.4M out (151 MB in) | 608 | 3.14 ms | **0.223 ms** | 138 ms | ≈0.22 ms | **wait-dominated: min AT floor, avg 14× floor** |

The hardware runs these payloads at wire speed when peers arrive together
(min = floor); the exposure is **peer-arrival skew inside the kernel**, which
no channel/chunk/protocol/NVLS knob addresses (the fix is upstream balance —
the overlap program, lebesgue's shim). ~4.7 s of the 5.6 s AllGather exposure
is wait, ~0.3 s is transfer floor. **No box arm. This was the last Mac-side
gate; lever 6 is now Arm 1 (ship-env ablation, riding the B/F 3-boot window)
only.**

### No-arm verdicts (documented, no box time)
- **PP p2p NET retuning (more QPs/channels for the cross-node path):** inert by
  mechanism — transfers already at line rate, cost is parking (schedule), and
  Run B measured the whole ship block at +0.4%. A knob cannot speed up waiting.
- **a2a ALGO/PROTO forcing (Ring/Tree, LL128/Simple) on the EP collectives:**
  the exposed a2a is peer-**wait** (imbalance), not transfer; large calls
  already run at 600–790 GB/s (spec). Inert.

## 4. Boot-economics option (fermi's call)

The B/F d4 A/B needs gates-on vs gates-off boots; Arm 1 needs NCCL-on vs
NCCL-off boots. Three boots serve both with one-variable discipline intact:

| boot | NCCL ship env | B/F gates | serves |
|---|---|---|---|
| 1 | on | off | shared off-arm (B/F off-arm **and** NCCL on-arm) |
| 2 | on | on | B/F on-arm |
| 3 | off | off | NCCL off-arm |

×2 d4 runs per boot = 6 runs, both experiments adjudicated, each arm pair
differing in exactly one env block. d2 canary on each fresh boot per house
rules. Arm 2 (if its gate passes) wants d16 — tag onto any d16 window.

## 5. Pre-registration summary

| arm | rung | reference | accept | reject |
|---|---|---|---|---|
| 1 (env ablation) | d4 | 878–886 | "inert" if \|Δ\| ≤ spread; drop-env if Δ < −3% beyond spread | Δ > +3% → investigate, don't believe |
| 2 (CP tuning) | d16 paired | 984 (133.2 s) | Δ > +2% beyond spread | else reject-inert; gate fail ⇒ never boots |

Noise discipline: d4 within-boot control spreads ran 3.0–3.8% tonight; d16
controls 7.4% with settle drift — mean-vs-spread rule throughout, per the L1
pre-registration pattern.
