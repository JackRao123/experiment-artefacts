# FLEX/DeepEP VERDICT — why it is not faster at 131k, and when it would be (gauss, 2026-08-13 ~08:4x CDT)

Jack's question, answered from the traces: `l5_flex_d4_rank0` (DeepEP/flex)
vs `mn_d4_rank0` (NCCL alltoall), same M=N d4 config, rank 0. Traced steps:
flex 41.67s vs alltoall 36.46s (untraced summary: 40.8 vs 35.7 = 5.1s;
802 vs 918 tok/s/GPU). All numbers below from the traced steps.

## The 5.1s, decomposed

| term | size | evidence |
|---|---:|---|
| DeepEP kernel residency vs NCCL a2a | **+1.37 s** | DeepEP dispatch+combine+notify kernels total 8.57s (dispatch 1.99 + combine 3.13 + cached_notify_combine 3.28 + notify/layout 0.17) vs the NCCL EP a2a SendRecv 7.20s. Same payload, more kernel work: the notify/layout kernels are pure overhead, and the dispatch/combine split pays two kernel boundaries per direction. |
| DeepEP never overlaps compute | **+0.6 s** | DeepEP comm/compute overlap = **0.0%** (0.00s of 8.57s); NCCL a2a overlapped 8.3% (0.59s of 7.20s). The dispatch/combine stages run as blocking phases between compute, not overlapped. |
| GPU-empty idle growth | **+3.14 s** | idle 4.24s → 7.38s; the growth is entirely in **64 gaps >10ms = 3.33s** (alltoall: 4 gaps, 0.64s) — the serialized dispatch/combine windows where the GPU drains and waits. Micro-gap launch drag is unchanged (2.65 vs ~2.5s). |
| CP-collective balloon (second-order convoy) | **+2.7 s on the main stream** | CP AllGather per-call 0.34ms → 6.5ms (19×; 0.23s → 2.55s total), ReduceScatter 2.2× (0.32 → 0.70s). These park on the main stream waiting for peers whose compute is delayed by the contention — a convoy effect of the primary serialization, not an independent cost. |

The terms interlock (the idle growth and the collective balloon are two views
of the same serialization), so they do not sum linearly to 5.1s; the primary
drivers are the first three.

**The SM-contention candidate, verified:** the main compute stream carries
the same work but totals 17.16s vs 13.74s (+3.42s). Of that, +2.70s is the
CP-collective balloon (parking, above); the pure-compute kernels grew only
modestly (~+0.7s, mixed per kernel) — so DeepEP's SM reservation slows
concurrent compute a little, but the dominant cost is the serialization, not
raw SM starvation of the math.

## The roofline check (cauchy's argument, verified)

At 131k the per-call token payload is ~1.6 GB (measured 813.7M elements avg,
BF16) and the NCCL a2a already achieves **611 GB/s p95 (733 max)** — ~85% of
the B300 NVLink per-direction line rate. So the a2a is **bandwidth-bound at
the wire**: no dispatcher can beat the transfer. DeepEP's levers are (a)
latency on small payloads — irrelevant at 1.6 GB/call, (b) fusing the
permute/layout into the comm kernel — tonight it REMOVED the TE chunk_sort
(−0.49s) but ADDED its own notify/layout kernels (+0.17s) plus the
dispatch/combine split overhead, a net wash-to-loss, and (c) enabling overlap
— realized as 0.0%, the opposite. So at this shape DeepEP cannot win on
transfer and tonight it loses on overlap and contention.

## Under what conditions DeepEP/flex WOULD be faster

1. **Latency-bound small-payload regimes** — the 16k customer shape (many
   more, much smaller a2a calls): per-call latency and host overhead dominate
   there, and DeepEP's fused low-latency path is built for exactly that. The
   Aug-9 16k regime is where the dispatcher host-syncs already mattered most.
2. **If its overlap machinery is actually engaged** — DeepEP's
   dispatch/combine are designed to overlap compute; tonight they ran fully
   serialized (0.0%). With a schedule that gives the dispatch/combine windows
   another microbatch's compute to hide behind (the L0b / combined-1f1b
   program), the exposed 8.57s could largely vanish — that flips the
   comparison.
3. **Cross-node (IB) EP**, where DeepEP's RDMA/NVLink split path matters —
   tonight's EP8 is intra-node NVLink, NCCL's home turf at line rate.
4. NOT when the payload is large and the NCCL path already saturates the
   wire — there is no transfer win to take, only overhead to add.

## Report-ready paragraph

> We also tested a DeepEP/flex dispatcher variant against the NCCL
> all-to-all at the 131k shape, same config. It was slower — 802 vs 918
> tokens/s/GPU (40.8s vs 35.7s steps). The traces show why, and it is not a
> bandwidth problem: the all-to-all already runs at ~85% of NVLink line rate
> at our ~1.6 GB per-call payloads, so no dispatcher can win on transfer.
> DeepEP lost on three counts: its dispatch/combine/notify kernels do ~1.4s
> more kernel work per step than the single NCCL collective for the same
> bytes; they ran with **zero** compute overlap (NCCL overlapped ~8%), so the
> GPU sat empty in 64 long (>10ms) serialized windows, +3.1s of idle; and
> their SM reservation stretched the downstream collectives (~+2.7s of
> convoy parking). The variant's real advantages are latency-bound
> small-payload shapes (the 16k customer regime) and schedules that actually
> overlap its dispatch/combine with compute — neither present at 131k tonight.
> Recommendation: keep the NCCL all-to-all at 131k; revisit DeepEP for the
> 16k regime or once the overlap schedule lands.
