# Ali B300 cross-node GPU P2P bandwidth

**Measured:** 2026-08-26 UTC  
**Devbox:** `tj-w5ylgj3`, 2x8 B300 (`ali-apse7-prod-1`)  
**Nodes:** `b300-1-izksekdp-0001`, `b300-1-s58nc356-0011`  
**Benchmark:** [`../../tools/cross_node_p2p_bw.py`](../../tools/cross_node_p2p_bw.py)

All bandwidths are decimal GB/s. The test moves payload directly from source
GPU HBM to destination GPU HBM with NCCL send/receive over GPUDirect RDMA. It
does not stage payload through host memory.

## Headline

| traffic pattern | best measured throughput | interpretation |
|---|---:|---|
| One GPU to matching GPU, one way | **74.83 GB/s** | one 2x400G LACP bond, 4 GiB messages |
| One GPU pair, simultaneous both ways | **51.59 GB/s each way, 103.17 GB/s combined** | full duplex; do not compare the combined number with a one-way ceiling |
| Eight matching GPU pairs, one way | **515.59 GB/s aggregate** | 64.45-78.13 GB/s per pair; 8 independent 2-rank communicators |
| Eight matching GPU pairs, simultaneous both ways | **771.72 GB/s combined** | 48.23-51.69 GB/s per pair per direction |

The direct one-way stack scales to 6.89x the isolated pair, not 8x. Relative
to eight copies of the isolated 74.83 GB/s result, it retains 86.1% of ideal
scaling. The full-duplex stack retains 93.5% of eight copies of the isolated
103.17 GB/s combined result.

These are the maxima observed in this direct-P2P sweep, not proofs that no
other NCCL protocol/channel setting can improve them.

## Topology

Each node has eight GPUs and eight matching logical RoCE bonds:

```text
node 0                                      node 1
GPU 0 -- PCIe Gen6 x16 -- mlx5_bond_0 ==== mlx5_bond_0 -- PCIe Gen6 x16 -- GPU 0
GPU 1 -- PCIe Gen6 x16 -- mlx5_bond_1 ==== mlx5_bond_1 -- PCIe Gen6 x16 -- GPU 1
...                                         ...
GPU 7 -- PCIe Gen6 x16 -- mlx5_bond_7 ==== mlx5_bond_7 -- PCIe Gen6 x16 -- GPU 7
```

Within each node, all GPU pairs are connected through NVLink/NVSwitch
(`NV18`). Cross-node traffic uses RoCE, not NVLink.

Measured topology facts on both nodes:

- Every GPU PCIe endpoint is currently negotiated at its maximum `64.0 GT/s
  x16`: PCIe Gen6 x16.
- Every matching NIC PCIe endpoint is also currently and maximally `64.0 GT/s
  x16`.
- `nvidia-smi topo -m` reports every GPU i to `mlx5_bond_i` path as `PXB`.
  The GPUDirect path crosses PCIe switches but not CPU cores, host DRAM, the
  CPU root bridge, or inter-socket UPI.
- GPUs/bonds 0-3 are on NUMA 0; GPUs/bonds 4-7 are on NUMA 1.
- Every `mlx5_bond_i` is an IEEE 802.3ad LACP group with two active 400,000
  Mb/s members. Raw bond line rate is therefore 800 Gbit/s = 100 GB/s in each
  direction.
- All eight RDMA devices are `PORT_ACTIVE`, Ethernet link layer, MTU 4096.

NCCL logs confirm `NET/IB/.../GDRDMA`, `GPU Direct RDMA Enabled`, RoCE GID 3,
and the matching `mlx5_bond_i` on every local GPU rank. There was no socket
fallback.

### Terminology

- **QP (queue pair):** one RDMA send queue plus one receive queue. Each QP is
  a separately hashable flow for LACP.
- **NCCL channel:** an independent NCCL communication pipeline. One channel
  underfilled this path; four channels were best in this test.
- **Rail:** an independent network path. Here the GPU-local logical rail is an
  `mlx5_bond_i` LACP group containing two physical 400G members.
- **GPUDirect RDMA:** the NIC reads/writes GPU HBM over PCIe without staging
  payload in host DRAM.
- **Full duplex:** simultaneous traffic in both directions. Combined
  full-duplex bandwidth sums the two directions and is not a one-way rate.

## Direct P2P results

The long tests move 200 GiB per active direction per pair: 200 iterations of
a 1 GiB payload. The 4 GiB rows move the same total volume with 50 iterations.

### Isolated GPU pair

| payload | QPs/channel | channels/peer | direction | GB/s per direction | combined GB/s |
|---:|---:|---:|---|---:|---:|
| 1 GiB | 1 | 1 | one way | 18.59 | 18.59 |
| 1 GiB | 8 | 1 | one way | 18.89 | 18.89 |
| 1 GiB | 2 | 4 | one way | 55.62 | 55.62 |
| 1 GiB | 1 | 4 | one way | **74.70** | **74.70** |
| 1 GiB | 1 | 8 | one way | 48.69 | 48.69 |
| 1 GiB | 1 | 16 | one way | 73.82 | 73.82 |
| 4 GiB | 1 | 4 | one way | **74.83** | **74.83** |
| 1 GiB | 1 | 4 | bidirectional | **51.59** | **103.17** |

The one-channel result is a software/flow underutilization control, not a
hardware expectation. Raising QPs alone did not help; raising NCCL channels
did. Four channels were best in this sweep. LACP hashes flows to members, so
more channels/QPs can expose the second 400G member, but excessive concurrency
also adds overhead and hash balance is not guaranteed.

### Eight independent GPU pairs

Each local GPU i uses its own two-rank NCCL communicator and is pinned to
`mlx5_bond_i`.

| payload | direction | effective aggregate | slowest pair/direction | pair rates |
|---:|---|---:|---:|---|
| 1 GiB | one way | **515.59 GB/s** | 64.45 GB/s | 64.59, 64.84, 78.13, 74.37, 64.45, 64.57, 64.67, 74.89 |
| 1 GiB | bidirectional | **771.72 GB/s combined** | 48.23 GB/s each way | 51.17, 48.23, 51.69, 51.14, 51.09, 50.85, 51.10, 51.06 per direction |
| 4 GiB | one way | 389.64 GB/s | 48.70 GB/s | no improvement from the larger direct-send operation |

The effective aggregate is total equal work divided by the slowest pair's
completion time. For the one-way row, the sum of independently measured pair
rates is 550.51 GB/s, but faster pairs finish and wait for the slowest pair;
515.59 GB/s is the synchronized workload throughput.

## Communicator limitation

Using one shared 16-rank communicator for eight simultaneous P2P pairs was a
large software bottleneck:

| communicator layout | one-way aggregate | bidirectional combined |
|---|---:|---:|
| One shared 16-rank communicator | 86.44 GB/s | 172.90 GB/s |
| Eight independent 2-rank communicators | **515.59 GB/s** | **771.72 GB/s** |

The shared-communicator result must not be interpreted as the physical fabric
limit. Communicator/channel topology matters as much as QP count for this
traffic pattern.

## Existing measurements

### Exact fabric, different NCCL traffic pattern

`mp/networking/network_precheck/README.md` in the
`baseten-wt-loops-current-release` worktree records prior Lingjun L20D/B300
NCCL `all_reduce_perf` measurements:

- **99.3 GB/s busbw** on one forced 2x400G bond at a 4 GiB message.
- **803-817 GB/s busbw** across all eight bonds at 4-8 GiB.
- 1 GiB was explicitly insufficient to saturate that collective test.

Those are valid fabric ceilings but are not direct GPU-i to GPU-i P2P rates.
All-reduce uses a different schedule and can feed a bond from multiple GPUs
through NVLink. This run's direct-pair results must remain separate.

### Existing LPS-1062 application evidence

- `runs/overnight_20260813_overlap_campaign/NCCL_RESWEEP_PLAN.md` measured a
  200 MB cross-node pipeline P2P at **48 GB/s** (`4.2 ms`), consistent with a
  smaller application transfer using one 400G member effectively.
- Root `REPORT.md` and `NOTEBOOK.md` record that QP/channel tuning reduced the
  EP all-to-all SendRecv p50 from **34 ms to 17.8 ms**. The mechanism was LACP
  flow spreading.
- `runs/overnight_20260807_baseline_shipconfig/glm52-b300-s256k/REPORT.md`
  estimated approximately **47 GB/s effective** for the original application
  all-to-all path.
- `B300_HOST_OFFLOAD_BANDWIDTH.md` measured the local PCIe/host-memory path at
  55.7-57.3 GB/s per GPU and 438.8-456.5 GB/s across eight GPUs. That is host
  offload, not GPUDirect network throughput, but corroborates the Gen6 x16 and
  GPU/NIC locality topology.

No prior artifact found in the LPS-1062 tree measured the exact eight fixed
cross-node GPU-pair pattern used here.

## Interpretation

1. A GPU does have 800 Gbit/s of raw bond capacity, but one NCCL channel does
   not expose it. The measured direct-pair maximum is 74.83 GB/s, not 18.6.
2. PCIe is not the one-pair line-rate bottleneck. Both GPU and NIC endpoints
   are Gen6 x16, whose usable one-way capacity exceeds the bond's 100 GB/s raw
   line rate.
3. Eight pairs do contend or lose efficiency: one-way per-pair throughput
   falls from 74.83 isolated to a slowest-pair 64.45 GB/s under full load.
4. The stack still scales well when communicators are independent: 515.59
   GB/s one way and 771.72 GB/s combined full duplex.
5. A shared communicator can impose a far lower software ceiling. Training
   process-group layout must be considered before blaming PCIe or RoCE.

## Software and environment

- PyTorch `2.11.0+cu130`
- CUDA `13.0`
- NCCL `2.28.9`
- `NCCL_SOCKET_IFNAME=eth0`
- `NCCL_IB_GID_INDEX=3`
- `NCCL_IB_MERGE_VFS=0`
- `NCCL_NET_PLUGIN=none`
- Local rank i sets `NCCL_IB_HCA=mlx5_bond_i` before communicator creation
- Best direct setting: `NCCL_IB_QPS_PER_CONNECTION=1`,
  `NCCL_NCHANNELS_PER_NET_PEER=4`

The GPUs were otherwise idle. Payload samples were checked after every run.
