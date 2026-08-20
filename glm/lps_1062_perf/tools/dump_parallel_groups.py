#!/usr/bin/env python3
"""Dump megatron parallel-state process groups for a TP/PP/CP/EP/ETP layout.

Pre-flight check for LPS-1062 PP2/CP8/EP8 @131k bring-up: verifies that with
the bridge's default rank order (tp-cp-ep-dp-pp, pp outermost), PP is the
CROSS-NODE dimension and CP/EP groups stay INTRA-node — the perf thesis of
the whole push (EP a2a + CP collectives on NVLink, only PP p2p cross-node).

Run on the 2-node box with the trainer venv active and MEGATRON_LM_PATH
pointing at the vendored mcore
(<clone>/server/vendor/megatron-bridge/3rdparty/Megatron-LM):

    srun --overlap --nodes=2 --ntasks-per-node=8 \
        torchrun --nnodes=2 --nproc_per_node=8 \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        python3 dump_parallel_groups.py --pp 2 --cp 8 --ep 8

Exit 0 iff CP and EP groups never span an 8-rank node boundary and every PP
group pairs ranks across the two nodes. Prints a full rank->group table.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

import torch
import torch.distributed as dist


def group_ranks(group) -> list[int]:
    return sorted(dist.get_process_group_ranks(group))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--cp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--etp", type=int, default=1)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    args = ap.parse_args()

    mcore = os.environ.get("MEGATRON_LM_PATH")
    if mcore:
        sys.path.insert(0, mcore)
    import megatron.core.parallel_state as mpu

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % args.gpus_per_node))
    torch.cuda.set_device(local_rank)

    if world % (args.tp * args.pp * args.cp) != 0:
        raise SystemExit(
            f"world {world} not divisible by tp*pp*cp "
            f"({args.tp}*{args.pp}*{args.cp})"
        )

    # Same call the megatron-bridge initialize path makes
    # (order="tp-cp-ep-dp-pp" when use_tp_pp_dp_mapping=False, the default).
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        expert_model_parallel_size=args.ep,
        expert_tensor_parallel_size=args.etp,
        order="tp-cp-ep-dp-pp",
    )

    info = {
        "rank": rank,
        "host": socket.gethostname(),
        "slurm_nodeid": os.environ.get("SLURM_NODEID"),
        "tp": group_ranks(mpu.get_tensor_model_parallel_group()),
        "pp": group_ranks(mpu.get_pipeline_model_parallel_group()),
        "cp": group_ranks(mpu.get_context_parallel_group()),
        "ep": group_ranks(mpu.get_expert_model_parallel_group()),
        "dp": group_ranks(mpu.get_data_parallel_group()),
        "dp_with_cp": group_ranks(
            mpu.get_data_parallel_group(with_context_parallel=True)
        ),
        "pp_rank": mpu.get_pipeline_model_parallel_rank(),
        "cp_rank": mpu.get_context_parallel_rank(),
        "ep_rank": mpu.get_expert_model_parallel_rank(),
    }
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(info, gathered, dst=0)

    if rank != 0:
        dist.destroy_process_group()
        return

    n = args.gpus_per_node
    node_of = {}
    for g in gathered:
        node_of[g["rank"]] = g["rank"] // n
        print(
            f"rank {g['rank']:>2} host={g['host']} SLURM_NODEID={g['slurm_nodeid']} "
            f"pp_rank={g['pp_rank']} cp_rank={g['cp_rank']} ep_rank={g['ep_rank']} | "
            f"pp={g['pp']} cp={g['cp']} ep={g['ep']} dp={g['dp']} dp+cp={g['dp_with_cp']}",
            flush=True,
        )

    def spans_node(group: list[int]) -> bool:
        return len({r // n for r in group}) > 1

    cp_groups = {tuple(g["cp"]) for g in gathered}
    ep_groups = {tuple(g["ep"]) for g in gathered}
    pp_groups = {tuple(g["pp"]) for g in gathered}

    ok = True
    for name, groups, must_be_intra in (("CP", cp_groups, True), ("EP", ep_groups, True)):
        for grp in sorted(groups):
            bad = spans_node(list(grp))
            if bad and must_be_intra:
                ok = False
            print(f"{name} group {list(grp)}: {'CROSS-NODE !!' if bad else 'intra-node OK'}")
    for grp in sorted(pp_groups):
        cross = spans_node(list(grp))
        if not cross:
            ok = False
        print(f"PP group {list(grp)}: {'cross-node OK (expected)' if cross else 'INTRA-NODE !!'}")

    # Hostname cross-check: contiguous rank blocks really are distinct hosts.
    hosts = {g["rank"]: g["host"] for g in gathered}
    block_hosts = [{hosts[r] for r in range(b * n, (b + 1) * n)} for b in range(world // n)]
    if any(len(h) != 1 for h in block_hosts):
        ok = False
        print(f"!! rank blocks do not map 1:1 to hosts: {block_hosts}")
    else:
        print(f"rank blocks -> hosts: {[next(iter(h)) for h in block_hosts]}")

    print("VERDICT:", "OK — PP cross-node, CP/EP intra-node" if ok else "FAILED")
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
