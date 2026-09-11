# Isolated BF16 expert GEMM shape experiment

Transformer Engine 2.16.0 GroupedLinear, frozen BF16 weights, no communication,
no dequantization, no routing, no LoRA, no activation. One otherwise-idle B300.
Three warmups and ten measured repetitions for each projection. CUDA events
measure the complete GroupedLinear forward and input-gradient paths, including
GPU launch starvation where applicable. These are effective path rates, not
hardware-counter saturation measurements and not whole-model MFU.

Balanced expert counts are deliberately synthetic. Actual model routing is
uneven; these numbers isolate topology-dependent shape/weight-footprint effects,
not a byte-for-byte replay of the full-model trace.

| Shape | Experts | Rows/expert | Gate/up fwd ms | Down fwd ms | Gate/up dgrad ms | Down dgrad ms |
|---|---:|---:|---:|---:|---:|---:|
| CP8EP8-like | 32 | 4096 | 4.006 | 2.157 | 3.637 | 1.983 |
| CP8EP1-like | 256 | 512 | 6.438 | 3.914 | 5.013 | 4.984 |
| CP1EP1-like | 256 | 4096 | 37.836 | 18.851 | 36.590 | 18.938 |

The first two rows do the same total arithmetic: 6.597 TFLOP for gate/up and
3.299 TFLOP for down, in each forward or dgrad. Gate/up is K=6144,N=4096;
down is K=2048,N=6144. CP1 does eight times that arithmetic.

CP8EP1-like forward totals 10.353 ms versus 6.164 ms: 68% slower for the same
FLOPs. Dgrad totals 9.996 versus 5.620 ms: 78% slower. Therefore, removing
expert communication does not leave an otherwise equal expert-compute cost.
The two projections hold 18 GiB of weights at EP1 versus 2.25 GiB at EP8,
and EP1 distributes the rows across eight times as many GEMMs.

Gate/up effective forward throughput is 1025 TFLOP/s at the EP1 shape versus
1647 at EP8; down is 843 versus 1529. Input-gradient rates are 1316/662 versus
1814/1663 TFLOP/s, respectively. These rates include dispatching GEMMs inside
the GroupedLinear call; they do not establish whether individual kernels are
limited by memory, instruction issue, tile occupancy, or host launch overhead.

The CP1-like forward total divided by eight is 7.086 ms, and dgrad is 6.941 ms.
Its larger rows/expert recover substantial efficiency despite the replicated
weights. The exact loss of full-step efficiency is separately accounted for in
RESULTS.md; do not sum these synthetic probe times into that trace.

Source: grouped_gemm_probe.py. Raw samples and exact arithmetic:
grouped-gemm-probe.log. The optional single-grouped-weight experiment is not
included above. It was subsequently run and did not improve this BF16 path:
CP8EP1-like forward totals 10.341 ms versus 10.353 ms, and dgrad 10.211 ms
versus 9.996 ms. No trainer storage-layout change was made on that evidence.
Its raw samples are in grouped-gemm-single-weight.log.

## Alternative grouped-MM implementation (microbenchmark only)

Using torch.nn.functional.grouped_mm with an already-packed 3D BF16 weight
tensor gives the following. Packing is outside the timing window; this does
not justify stacking all expert weights afresh inside every model forward.

| Shape | Gate/up fwd ms | Down fwd ms | Gate/up dgrad ms | Down dgrad ms |
|---|---:|---:|---:|---:|
| CP8EP8-like | 3.856 | 1.801 | 3.794 | 1.856 |
| CP8EP1-like | 4.448 | 2.044 | 4.441 | 2.002 |
| CP1EP1-like | 40.302 | 18.801 | 40.634 | 19.041 |

The EP1-like smaller GEMMs benefit: 6.492 ms forward and 6.443 ms dgrad,
versus TE's 10.353 and 9.996. CP1's larger GEMMs do not improve. A usable
training optimization needs an appropriate persistent/sharded weight layout,
numerical checks, and an end-to-end run; this is not yet such a result.
Raw samples: grouped-gemm-torch.log.
