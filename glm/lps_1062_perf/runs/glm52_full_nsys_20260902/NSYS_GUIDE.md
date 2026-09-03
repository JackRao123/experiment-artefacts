# Getting and reading an Nsight Systems trace

## What nsys records

Nsight Systems produces a time-ordered report. Its timeline aligns CPU threads,
CUDA API calls made by those threads, GPU kernels, memory copies, and named
NVTX ranges. A `.nsys-rep` file is the self-contained report opened by the
Nsight Systems GUI.

For distributed PyTorch, launch the complete `torchrun` process tree under one
nsys session. Each rank then appears as a separate Python process in the same
report.

## Check the installation

```bash
nsys --version
command -v nsys
```

The target-side collector alone can record a raw `.qdstrm`, but the full
Nsight Systems package is required to convert it to `.nsys-rep`. On the Ubuntu
CUDA 13 image used for this run, the package was:

```bash
apt-get install -y nsight-systems-2025.3.2
```

Use a GUI version at least as new as the CLI that produced the report.

## Recommended server workflow

The interactive workflow lets the trainer start and warm up before collection.
This avoids filling the report with model loading, compilation, and autotuning.

First, register the trainer process tree inside a named nsys session. The
command that normally starts `torchrun` is the final command here:

```bash
nsys launch \
  --session-new=my-trace \
  --trace=cuda,nvtx,osrt \
  --pytorch=autograd-nvtx \
  --show-output=true \
  --wait=all \
  torchrun --standalone --nproc_per_node=8 -m your_module
```

With nsys 2025.3.2, `nsys launch` leaves the session idle and the first
`nsys start` actually launches the registered application. Use a throwaway
startup collection:

```bash
nsys start \
  --session=my-trace \
  --sample=none \
  --cpuctxsw=none \
  --output=/tmp/startup-discard \
  --force-overwrite=true
```

Wait until the service is healthy, stop that collection, and discard its
report. The trainer process remains attached to the session:

```bash
nsys stop --session=my-trace
```

Now run the exact input shape once or twice without collecting. The first
execution often compiles or autotunes kernels and is not representative of
steady state.

Start collection only after warmup:

```bash
nsys start \
  --session=my-trace \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --output=/path/to/report \
  --force-overwrite=true
```

Run one representative operation, then stop immediately:

```bash
nsys stop --session=my-trace
```

The result should be `/path/to/report.nsys-rep`. CPU sampling and context-switch
collection are useful for unexplained host-side gaps but add overhead. For the
lowest-overhead GPU-focused capture, use `--sample=none --cpuctxsw=none` and
omit `osrt` from `--trace`.

## Alternative: application-controlled capture

For a normal training loop that you can edit, bracket the desired iterations:

```python
torch.cuda.cudart().cudaProfilerStart()
run_training_steps()
torch.cuda.cudart().cudaProfilerStop()
```

Then launch with:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --pytorch=autograd-nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --output=/path/to/report \
  python train.py
```

Every distributed rank must receive the start and stop calls. The interactive
session method is easier for this trainer server because the profiled operation
arrives later over HTTP.

Do not run Kineto and nsys simultaneously. Both consume CUDA profiling events,
and simultaneous collection can fail or distort the result.

## Validate and summarize the report

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  report.nsys-rep
```

Useful reports include:

- `cuda_gpu_kern_sum`: GPU time grouped by kernel name.
- `cuda_api_sum`: CPU time spent in CUDA API calls.
- `nvtx_gpu_proj_sum`: GPU work grouped under NVTX ranges.
- `cuda_gpu_trace`: every GPU event in chronological order.

## Open the report

Copy the `.nsys-rep` to the workstation and open it directly:

```bash
scp host:/path/to/report.nsys-rep .
open report.nsys-rep
```

## First GUI inspection

1. Find the short interval containing dense activity in the `CUDA HW` rows and
   zoom into it. Do not inspect a full model-startup-width timeline.
2. Expand one Python process and its `CUDA HW` row. Each Python process is one
   distributed rank; the CUDA row is actual GPU execution.
3. Look for white gaps in the CUDA row. A solid row is GPU-bound. A gap while
   CPU/NVTX work continues suggests launch, synchronization, or CPU overhead.
4. Click the widest kernels and inspect their names and durations. Confirm the
   same pattern across ranks rather than trusting one rank.
5. Inspect NCCL ranges. Communication overlapping GPU compute is generally
   healthy; communication followed by idle GPU space is exposed overhead.
6. Treat NVTX width as a CPU annotation, not automatically as GPU time. Use the
   CUDA row and event correlation to determine what actually ran on the GPU.

Colors identify event categories; they do not mean good or bad. Begin with
duration, gaps, overlap, and rank-to-rank consistency.
