# HUMAN WRITTEN
Goal
Scope: we are only doing this for the routed experts. Every other param stays the same as before.
Routed experts (aka MOE experts) are 97.77% of our FP8 params.  

There are two parts.  
1. Load weights
2. Runtime



Before, we were doing.
Load weights: 
W_bf16 = BF16(FP32(W_native_fp8) * expanded_scale_inv_fp32)

Runtime:
A_bf16_next = BF16(BF16_GEMM_FP32_ACCUM(A_bf16, W_bf16))


Now, what we want to do is
Load weights:
Store W_native_fp8 and scale_inv_fp32 on the GPU.

Runtime: 
W_equiv_bf16 = BF16(FP32(W_native_fp8) * block_scale_fp32)
A_bf16_next_new = BF16(
    BF16_GEMM_FP32_ACCUM(A_bf16, W_equiv_bf16)
)


For the 'runtime' part, ideally there is a kernel that can do that mixed-precision multiplication in one go, but unfortunately there isn't.  
How we achieve `A_bf16_next_new` actually is just an implementation detail.  


Acceptance criteria:
- Same forward numerics as old main
- BF16 activations
- BF16 LoRA
- ~90 ish GB/GPU lower expert storage
- Higher MFU than old main


We can use whatever kernel implementation achieves the same numerics, as long as its higher MFU than before.  




There are two categories. Load, and runtime. 

# AI WRITTEN Native FP8 Storage With BF16 Compute Plan 
## Objective

Store only GLM-5.2 routed-expert frozen weights in their compact native
checkpoint form while preserving old-main training numerics. Every other
parameter remains exactly as it is on old main:

```text
weight storage: E4M3 payload + FP32 scale per 128x128 block
activation storage: BF16
GEMM operands: BF16 activation and BF16-equivalent weight
accumulation: FP32, matching the old BF16 kernel
output: BF16
```

The intended result is old-main computation without permanently materializing
the routed-expert weight matrices as BF16 in GPU global memory.

## Numerical Contract

The old loaded parameter is:

```python
W_bf16 = BF16(FP32(W_native_fp8) * expanded_scale_inv_fp32)
```

The old forward is:

```python
C_reference = BF16(BF16_GEMM_FP32_ACCUM(A_bf16, W_bf16))
```

The new forward must reconstruct `W_bf16` tile-by-tile inside the kernel and
use the same BF16 MMA and FP32 accumulation semantics:

```python
W_tile_bf16 = BF16(FP32(W_native_fp8_tile) * scale_inv_fp32)
C_new = BF16(BF16_GEMM_FP32_ACCUM(A_bf16, W_tile_bf16))
```

Initial correctness target: `torch.equal(C_new, C_reference)` for fixed inputs
and a fixed algorithm. If a different reduction order makes bitwise equality
impossible, record the exact ULP distribution before considering any relaxed
gate.

## Important Non-Goals

- Do not cast the logical FP32 weight back to bare FP8 without a scale. That is
  a second lossy quantization.
- Do not quantize the BF16 activation to FP8.
- Do not use TE W8A8 FP8 training as the numerical-equivalence solution.
- Do not keep a persistent BF16 copy of the base weight.
- Do not keep both rowwise and transposed FP8 copies unless measurements prove
  the memory tradeoff is acceptable.

## Current Implementation Status

`jackrao/glm52-native-blockwise-fp8-v2` changes checkpoint reconstruction from
a BF16 intermediate to an FP32 intermediate. By itself it still copies into a
BF16 destination parameter and does not save runtime weight memory.

When combined with trainers `fp8_param=True`, TE stores an FP8 parameter and
quantizes activations during forward. That is W8A8 computation and does not
preserve old-main numerics. The tested TE blockwise path also retained extra
weight orientation storage, adding about 107.56 GB at full-model scale.

The desired design is a separate weight-only FP8 storage and BF16-compute path.

## Proposed Representation

For each routed-expert projection, retain two buffers:

```text
qweight: native E4M3 bytes, one byte per element
scale_inv: native FP32 grid, one value per 128x128 block
```

Do not create an ordinary BF16 `Parameter` for the frozen base weight. LoRA
parameters remain ordinary trainable BF16 parameters.

The loader must map the payload and scale grid together through all structural
operations:

```text
TP/EP sharding
expert selection
gate/up concatenation
transpose/layout conversion
padding for partial 128x128 edge blocks
```

No numerical requantization occurs during loading.

## Settled Design Decisions

### Native Checkpoint Pair Is The Stored Weight

Keep the original E4M3 payload and FP32 `scale_inv` grid together. Do not
dequantize and requantize during loading.

With `ETP=1`, expert parallelism assigns complete experts to EP ranks. Each
owning GPU can therefore receive the complete local expert payload and its
complete scale grid without sharding either tensor inside an expert. The scale
buffer follows the same expert placement as the payload.

### Exact Old Numerics Require BF16 Rounding Before Multiplication

Old main computes with:

```python
W_bf16 = BF16(FP32(q_e4m3) * scale_fp32)
```

A direct scaled product computes a different expression:

```python
A_bf16 * FP32(q_e4m3 * scale_fp32)
```

These differ because the second expression omits the old BF16 rounding of the
weight before multiplication. If exact old-main forward results remain the
goal, the kernel must recreate that BF16 rounding tile-by-tile before BF16 MMA.
The BF16 tile can live only in registers/shared memory; it does not need to be
stored persistently in GPU global memory.

A native mixed scaled GEMM without this rounding is a separate, close-but-not-
identical numerical mode. It may be worth benchmarking, but it does not satisfy
the exact-equivalence contract.

### Initial Scope Is Routed Experts Only

The trainer uses transformer layers 0-77; the appended checkpoint MTP layer 78
is disabled. The only tensors changing storage are:

```text
mlp.experts.*.gate_proj.weight [2048, 6144]  -> expert FC1
mlp.experts.*.up_proj.weight   [2048, 6144]  -> expert FC1
mlp.experts.*.down_proj.weight [6144, 2048]  -> expert FC2
```

There are 75 loaded MoE layers and 256 routed experts, so the trainer loads
19,200 of each projection. Routed experts account for 97.77% of native FP8
parameter elements, and 97.50% of all loaded logical model parameters. Across
all loaded tensors:

```text
total logical parameters:          743.377 B
routed-expert parameters:          724.776 B  (97.50%)
shared-expert parameters:            2.831 B
everything outside routed experts:  18.601 B
```

At `EP=8, ETP=1`, every GPU owns one eighth of the routed experts and each local
expert is complete. Approximate persistent base-weight storage per GPU is:

```text
old routed experts in BF16:            181.194 GB
native routed FP8 + FP32 scale grids:   90.619 GB
all other weights kept as before:       37.203 GB

old total:                              218.397 GB
routed-experts-only FP8 total:          127.822 GB
saving:                                  90.575 GB per GPU
```

The implementation must touch only the routed-expert FC1 and FC2 grouped GEMMs.
Shared experts, dense MLPs, MLA/attention projections, DSA indexers, routers,
norms, embeddings, LM head, activations, and LoRA remain unchanged and BF16/FP32
exactly as on old main. This captures most of the available memory reduction
while minimizing the numerical and integration surface.

## Kernel Strategy

First survey CUTLASS, CuTe DSL, NVJet, Transformer Engine, and existing grouped
MoE kernels for a weight-only operation with these semantics:

```text
input: BF16 activation
weight: E4M3 payload + FP32 128x128 scales
internal weight operand: round dequantized tile to BF16
MMA: BF16 with FP32 accumulation
output: BF16
```

If no existing kernel exposes those exact semantics, build a narrow B300
prototype:

```text
load one FP8 weight tile
load its FP32 scale
convert scaled values to BF16 in registers/shared memory
run BF16 tensor-core MMA
apply the same epilogue
store BF16 output
```

The FP8 weight must never be expanded into a persistent global-memory BF16
matrix.

## Available Kernel Inventory

Checked on `tj-w5y89m3` with B300/SM103, PyTorch 2.11.0+cu130,
Transformer Engine 2.16.0, and the installed FlashInfer/CUTLASS DSL packages.

### PyTorch

`torch.mm` requires matching operand dtypes. `torch._scaled_mm` accepts the
installed FP8 scaling layouts only when both matrix operands are FP8. A BF16 A
operand with an E4M3 B operand is rejected as an invalid scaling
configuration.

### Transformer Engine / cuBLASLt

`general_gemm` with BF16 activation and a TE current-scaled FP8 weight reaches
cuBLASLt but reports no supported algorithm. Standard TE FP8 linear instead
quantizes the activation and executes W8A8.

### FlashInfer Hopper Kernel

FlashInfer includes `fp8_blockscale_gemm_sm90`, whose API accepts:

```text
BF16 input
E4M3 weight
FP32 weight scale, including a 128x128 grid
BF16 output
```

However, it explicitly supports only SM90/SM90a. B300 reports SM103 and the API
rejects it. Its documentation also says BF16 inputs are internally quantized,
so it is not automatically the exact BF16-MMA contract even on Hopper.

### FlashInfer Blackwell Kernels

The installed SM100 groupwise and block-scaled APIs require both A and B to be
FP8. The SM100 dense block-scaled implementation explicitly rejects mixed
operand element types.

### CUTLASS DSL Starting Point

The installed CUTLASS DSL contains Blackwell `mixed_input_gemm` and
`grouped_mixed_input_gemm` examples. They convert a low-precision operand into
the BF16/FP16 MMA dtype inside the kernel and optionally apply a scale before
MMA. The ready examples expose Int8 and Int4 low-precision inputs, not E4M3
with a 128x128 FP32 scale grid.

This is the closest implementation starting point: extend the Blackwell mixed-
input template to accept E4M3 plus the native GLM FP32 scale layout, round the
transformed tile to BF16, and run BF16 MMA. This is adaptation work rather than
designing a GEMM from scratch.

## Execution Phases

### Phase 1: Isolated Numerical Prototype

Use native checkpoint tensors from one real routed expert. Benchmark both
expert projections independently:

```text
FC1 gate + up: [4096, 6144]
FC2 down:      [6144, 2048]
```

For FC1, concatenate the native gate and up E4M3 payloads along the output
dimension and concatenate their compact scale grids the same way. Generate the
old-main BF16 reference once from each native checkpoint pair. Compare the
prototype against that reference for fixed random BF16 activations.

Sweep expert token count `M` across:

```text
256, 512, 1024, 2048, 4096, 8192
```

`M=4096` is the balanced 131K/TopK8/256-expert operating point: each global
expert receives approximately `131072 * 8 / 256 = 4096` token assignments.
The sweep covers routing imbalance and shorter workloads.

Benchmark three arms:

```text
old reference: persistent BF16 weight + BF16 GEMM
naive upper bound: FP8 dequantized to a temporary BF16 tensor + BF16 GEMM
target: fused FP8 load/scale/BF16 conversion + BF16 GEMM
```

The naive arm is useful for correctness and a worst-case overhead bound, but is
not the intended memory implementation because it materializes the full BF16
weight temporarily.

Required measurements:

```text
bitwise equality
BF16 ULP histogram if not equal
maximum and mean absolute error
CUDA-event kernel duration after warmup
TFLOP/s and useful MFU
temporary and persistent GPU memory
```

### Phase 2: One-Layer Forward Integration

Integrate into the 0d1m GLM debug model for routed-expert FC1 and FC2 only.
Keep shared experts, DSA, routing, residuals, LoRA, and the LM head unchanged.

Compare old and new at every useful boundary:

```text
expert projection output
MoE output
transformer-block output
final logits
loss
```

### Phase 3: One-layer backward integration
same as phase 2; use the 0d1m GLM debug model, but for the backward.
Test a forward backward.

Forward:
FP8 -> BF16 -> GEMM -> discard BF16

Backward:
FP8 -> BF16 again -> dgrad GEMM -> discard BF16

### Phase 4: Full-Model fwdbwd Validation

Run full GLM-5.2 at sequence length 131072 with TP1/PP1/CP8/EP8/DP1. Compare
against old main using identical tokens and deterministic settings.


## Acceptance Gates

### Correctness

```text
exact BF16 reconstructed weight values
bitwise forward output target
matching logits and loss
later: matching activation and LoRA gradients
later: matching optimizer trajectory
```

### Memory

One native orientation costs approximately:

```text
1 byte per FP8 element
+ 4 bytes per 16,384-element scale block
= about 1.00024 bytes per weight
```

BF16 costs 2 bytes per weight, so the forward-only persistent-weight target is
approximately 50% lower. Reject implementations that retain a full BF16 copy

### Performance

```text
no material forward throughput regression
```

## Open Questions

- Does an existing B300 kernel support BF16 activation with block-scaled E4M3
  weight while rounding the weight operand to BF16 before MMA?
- Can that kernel match the old BF16 GEMM reduction order exactly?
- Can grouped routed-expert GEMM use one native weight orientation efficiently?
- What is the fastest dgrad design without a permanent transpose?
- Is bitwise equality required across all kernels, or is a documented BF16 ULP
  gate acceptable if reduction order is the only difference?

## PR Guidance

Keep the current FP8 PRs draft. The existing TE `fp8_param` path is useful as a
separate W8A8 performance experiment, but it must not be described as the
same-training memory-compression solution above.
