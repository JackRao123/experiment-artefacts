# B300 host-offload topology and bandwidth

**Measured:** 2026-08-20
**Cluster:** `ali-apse7-prod-1`
**Node accessed through:** `tj-wlmlkeq`
**Purpose:** establish the host-memory topology and practical transfer limits for
activation or parameter offloading on an 8xB300 node.

## Headline

- GPU-to-host activation offload sustains approximately **57.3 GB/s per GPU**.
- Host-to-GPU reload sustains approximately **55.7 GB/s per GPU**.
- A GPU simultaneously transferring in both directions sustains approximately
  **48 GB/s in each direction**, or **96 GB/s combined**.
- A single GPU performs the same with local and remote NUMA memory on this node.
  Remote placement still crosses Intel UPI and may contend when several GPUs do
  it concurrently.
- With all eight GPUs using NUMA-local memory, aggregate throughput is
  **438.84 GB/s host-to-device** and **456.45 GB/s device-to-host**.
- These numbers require pinned host memory and large asynchronous copies. They
  are not representative of pageable buffers or small synchronous transfers.

All rates in this report are decimal GB/s.

## Node architecture

| component | measured configuration |
|---|---|
| CPU | 2x Intel Xeon 6767P |
| CPU cores | 64 physical cores/socket, SMT2; 128 physical and 256 logical CPUs/node |
| NUMA | 2 nodes, one per CPU socket |
| Host RAM | 3.93 TiB Linux-visible, split approximately evenly between sockets |
| NUMA 0 RAM | 2,113,159,176 kB, approximately 1.97 TiB |
| NUMA 1 RAM | 2,111,050,048 kB, approximately 1.97 TiB |
| Current job memory limit | 2,662,556,762,112 bytes, approximately 2.42 TiB / 2.66 TB decimal |
| GPUs | 8x B300; 275,040 MiB visible HBM per GPU |
| Host link per GPU | PCIe Gen6 x16, current and maximum |
| GPU fabric | All-to-all NVLink/NVSwitch; every pair reports `NV18` |
| NVLinks | 18 active links/GPU, each reporting 53.125 GB/s |
| RoCE | 8 bonds; one closest PCIe peer per GPU |
| Local workspace | 4x Solidigm SB5PH27X076T approximately 7 TB NVMe in RAID0, 27.9 TB total |

The Alibaba platform firmware and `nvidia-smi` identify these GPUs as
`NVIDIA L20D`. This is the known Alibaba B300 labeling issue; the device shape
and the cluster inventory are B300.

The physical node has approximately 3.93 TiB of host RAM. A workload's usable
amount is its cgroup/Kubernetes memory limit, which can be lower. The measured
training container was limited to 2.42 TiB.

## Topology

```text
8xB300 node
|
+-- NUMA 0 / CPU socket 0: Intel Xeon 6767P
|   +-- approximately 1.97 TiB host RAM
|   +-- logical CPUs 0-63,128-191
|   +-- GPU 0 -- PCIe Gen6 x16 -- RoCE bond 0 is closest peer
|   +-- GPU 1 -- PCIe Gen6 x16 -- RoCE bond 1 is closest peer
|   +-- GPU 2 -- PCIe Gen6 x16 -- RoCE bond 2 is closest peer
|   `-- GPU 3 -- PCIe Gen6 x16 -- RoCE bond 3 is closest peer
|
+-- Intel UPI inter-socket link
|
`-- NUMA 1 / CPU socket 1: Intel Xeon 6767P
    +-- approximately 1.97 TiB host RAM
    +-- logical CPUs 64-127,192-255
    +-- GPU 4 -- PCIe Gen6 x16 -- RoCE bond 4 is closest peer
    +-- GPU 5 -- PCIe Gen6 x16 -- RoCE bond 5 is closest peer
    +-- GPU 6 -- PCIe Gen6 x16 -- RoCE bond 6 is closest peer
    `-- GPU 7 -- PCIe Gen6 x16 -- RoCE bond 7 is closest peer

GPU 0-7 are also connected all-to-all through the NVLink/NVSwitch fabric.
```

`nvidia-smi topo -p2p n` reported P2P `OK` for every GPU pair. GPU-to-NIC
topology reports `PXB` for each matching GPU/bond pair, meaning the pair shares
the local PCIe-switch hierarchy without crossing the CPU host bridge. Other
same-socket GPU/NIC pairs report `NODE`; cross-socket pairs report `SYS`.

The four host-memory paths are:

| path | locality | interconnects crossed |
|---|---|---|
| GPU 0-3 to NUMA 0 RAM | local | PCIe, local memory controller |
| GPU 0-3 to NUMA 1 RAM | remote | PCIe, Intel UPI, remote memory controller |
| GPU 4-7 to NUMA 1 RAM | local | PCIe, local memory controller |
| GPU 4-7 to NUMA 0 RAM | remote | PCIe, Intel UPI, remote memory controller |

"Local PCIe DMA" or "NUMA-local host-memory transfer" describes the local
path. "Cross-socket DMA" or "NUMA-remote host-memory transfer" describes the
remote path. Both directions use the same physical route in reverse.

## Benchmark method

The reproducible benchmark is `tools/cuda_host_bw.cu`.

- The node's GPUs were idle before measurement.
- Host buffers used `cudaMallocHost`, matching CUDA pinned-memory offload.
- Each result copied a 256 MiB buffer 40 times, approximately 10.74 GB per
  direction.
- Copies used `cudaMemcpyAsync` and dedicated CUDA streams.
- The allocating and first-touch thread was pinned to a CPU on the intended
  NUMA node.
- Linux `move_pages` verified 64 sampled pages from every host buffer. Every
  sample was on the requested NUMA node, including the remote tests.
- Device synchronization bounded each timed region.
- The all-GPU aggregate is total bytes divided by the slowest worker's elapsed
  time because all workers begin at the same barrier.

### Reproduce

On an equivalent node:

```bash
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -Xcompiler=-pthread \
  tools/cuda_host_bw.cu -o /tmp/cuda_host_bw
```

The optional final argument is the CPU used to place host-memory pages. CPU 0
belongs to NUMA 0; CPU 64 belongs to NUMA 1.

```bash
# GPU 0 with local NUMA 0 memory
/tmp/cuda_host_bw 0 h2d 0
/tmp/cuda_host_bw 0 d2h 0
/tmp/cuda_host_bw 0 bidir 0

# GPU 0 with remote NUMA 1 memory
/tmp/cuda_host_bw 0 h2d 64
/tmp/cuda_host_bw 0 d2h 64
/tmp/cuda_host_bw 0 bidir 64

# All GPUs with the benchmark's built-in local NUMA mapping
/tmp/cuda_host_bw all h2d
/tmp/cuda_host_bw all d2h
```

## Single-GPU local and remote results

| path | RAM to GPU (H2D) | GPU to RAM (D2H) | simultaneous bidirectional |
|---|---:|---:|---:|
| GPU 0 to NUMA 0 RAM, local | 55.72 GB/s | 57.28 GB/s | 48.30 GB/s each way, 96.60 combined |
| GPU 0 to NUMA 1 RAM, remote over UPI | 55.68 GB/s | 57.27 GB/s | 48.03 GB/s each way, 96.07 combined |
| GPU 4 to NUMA 1 RAM, local | 55.73 GB/s | 57.27 GB/s | 48.25 GB/s each way, 96.50 combined |
| GPU 4 to NUMA 0 RAM, remote over UPI | 55.69 GB/s | 57.27 GB/s | 48.04 GB/s each way, 96.08 combined |

For one GPU, local versus remote NUMA placement differs by less than 1%. One
approximately 57 GB/s copy stream therefore does not exhaust the available UPI
bandwidth on this machine.

The PCIe links are full-duplex, but the measured directions are not perfectly
independent. Running one direction alone reaches 55.7-57.3 GB/s. Running both
directions concurrently reduces each to approximately 48 GB/s while increasing
combined throughput to approximately 96 GB/s. The PCIe directions are separate,
but the GPU copy engines, endpoint, host bridge, and memory subsystem still
share resources.

## All-eight-GPU NUMA-local results

### Host RAM to GPU HBM

| GPU | throughput |
|---:|---:|
| 0 | 55.26 GB/s |
| 1 | 55.28 GB/s |
| 2 | 55.28 GB/s |
| 3 | 55.11 GB/s |
| 4 | 55.05 GB/s |
| 5 | 54.85 GB/s |
| 6 | 54.90 GB/s |
| 7 | 55.08 GB/s |
| **aggregate** | **438.84 GB/s** |

### GPU HBM to host RAM

| GPU | throughput |
|---:|---:|
| 0 | 57.14 GB/s |
| 1 | 57.15 GB/s |
| 2 | 57.06 GB/s |
| 3 | 57.17 GB/s |
| 4 | 57.06 GB/s |
| 5 | 57.12 GB/s |
| 6 | 57.09 GB/s |
| 7 | 57.11 GB/s |
| **aggregate** | **456.45 GB/s** |

All eight GPUs retain almost all single-GPU bandwidth when each uses RAM from
its local socket. The node's two host-memory domains can therefore sustain the
measured activation-offload traffic without visible aggregate collapse in this
copy-only test.

## Application guidance

1. Allocate pinned host buffers on NUMA 0 for GPUs 0-3 and NUMA 1 for GPUs 4-7.
2. First-touch or explicitly bind the host pages before registering/using them.
   Pinning only the worker thread after allocation is too late.
3. Use large asynchronous copies on dedicated streams and overlap them with
   computation. Small copies and synchronization overhead will lower effective
   throughput.
4. Do not use pageable host memory for the high-volume activation path. CUDA
   must stage pageable transfers, and apparently asynchronous calls can block.
5. Budget approximately 57.3 GB/s/GPU for offload and 55.7 GB/s/GPU for reload
   before accounting for framework overhead or compute/memory contention.
6. Treat cross-socket placement as a scaling risk even though one remote GPU is
   unaffected. Several GPUs simultaneously crossing UPI have not been measured.
7. The all-GPU result is a copy-only ceiling. Real training can reduce it through
   HBM contention, PCIe traffic from NICs/storage, allocator overhead, copy-size
   distribution, and imperfect compute/copy overlap.

## Remaining measurements

- Multi-GPU cross-socket matrix: GPUs 0-3 using NUMA 1 simultaneously and GPUs
  4-7 using NUMA 0 simultaneously, to find the UPI saturation point.
- All-eight-GPU simultaneous bidirectional throughput.
- Copy-size sweep to determine the minimum activation chunk that approaches the
  large-transfer ceiling.
- Copy/compute overlap under representative GLM kernels rather than an idle GPU.
- End-to-end activation-offload throughput in the intended implementation,
  including allocation, queueing, and synchronization costs.
