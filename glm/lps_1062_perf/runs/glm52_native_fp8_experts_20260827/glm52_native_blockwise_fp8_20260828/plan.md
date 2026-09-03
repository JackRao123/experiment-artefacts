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
- more than 1000TPS/GPU on full model.



# AI WRITTEN (human edited) Native FP8 Storage With BF16 Compute Plan



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

The new forward: 

```python
C_new = implementation...
```

Initial correctness target: `torch.equal(C_new, C_reference)` for fixed inputs
and a fixed algorithm. If a different reduction order makes bitwise equality
impossible, record the exact ULP distribution and make your judgement that its ok.

### Native Checkpoint Pair Is The Stored Weight

Keep the original E4M3 payload and FP32 `scale_inv` grid together. Do not
dequantize and requantize during loading.

With `ETP=1`, expert parallelism assigns complete experts to EP ranks. Each
owning GPU can therefore receive the complete local expert payload and its
complete scale grid without sharding either tensor inside an expert. The scale
buffer follows the same expert placement as the payload.

### Exact Old Numerics Require BF16 Rounding Before Multiplication

(exact, like, within noise)  

Old main computes with:

```python
W_bf16 = BF16(FP32(q_e4m3) * scale_fp32)
```

A direct scaled product computes a different expression:

```python
A_bf16 * FP32(q_e4m3 * scale_fp32)
```



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

"""
With full-layer activation recompute:
Initial forward:
create BF16 temporary
run GEMM
checkpointing releases BF16 storage
continue to next layer
The BF16 weights do not accumulate across all 75 layers.
During backward:
recompute one layer
create its BF16 temporary
TE retains it briefly
run that layer's dgrad
release it
move to the next layer

"""

we should just use TE's path that takes FP8 and dequants it at runtime and computes bf16 activations.



### Phase 3: One-layer backward integration

same as phase 2; use the 0d1m GLM debug model, but for the backward.
Test a forward backward.



### Phase 4: Full-Model fwdbwd Validation

Run full GLM-5.2 at sequence length 131072 with TP1/PP1/CP8/EP8/DP1. 
Record the TPS and MFU.

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

- roughly 90 GB/GPU lower persistent full-model storage;
- no BF16 expert tensor retained between forward and backward;



### Performance

```text
no material forward throughput regression
```

