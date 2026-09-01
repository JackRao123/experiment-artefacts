# PR 1228 deslop validation

- Trainers: `59697238ecef0be568d4b8254bf873ff74afcf71`
- Megatron-Bridge: `a63844500ec94a8ca47fb2ebffd361a7e34fba77`
- Megatron-LM: `836602e11e5cf697efdefadbd9a46aad9c24f42e`
- Hardware: 8x B300 (`NVIDIA L20D`)
- Topology: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1
- Sequence length: 131,072
- LoRA rank: 32
- Persistent expert storage: native blockwise FP8

## Results

| Window | Control TPS/GPU | Mean forward-backward |
|---|---:|---:|
| steady10 | 1,196 | 13.7 s |
| steady10 repeat | 1,204 | 13.6 s |

- Combined 20-control TPS from mean forward-backward time: **1,200 tok/s/GPU**
- Median individual control: **1,209 tok/s/GPU**
- Individual control range: 1,128-1,214 tok/s/GPU
- All control losses and gradient norms were finite.
- Focused Megatron-LM frozen-FP8 tests: 4 passed with Transformer Engine 2.16.0.

Raw driver outputs are stored beside this file.
