#!/usr/bin/env python3
"""Probe (a): three-way CP-split convention check for the THD tail-pad hunt.

gauss, 2026-08-13. Read-only w.r.t. the trainer. Run on the box with the
trainer venv python (has transformer_engine + the editable vendored mcore):

    /root/.cache/user_artifacts/trainers_main/server/.venv/bin/python probe_a_split_convention.py

What it checks, per CP geometry (CP8 and CP16) and padding mode (per-doc pad
only vs + tail-fill to 131072):

  leg 1 (mcore position map, CPU): dsa_layout.build_packed_allgather_cp_local_positions
          per rank -> the absolute padded coordinate assigned to each local row.
  leg 2 (TE data sharding, CUDA, tiny): tex.thd_get_partitioned_indices per rank
          -> the global padded row index each local slot holds.

INVARIANT: the two must be EXACTLY equal per rank (both are "the global padded
coordinate of this rank's i-th local row"). Any mismatch is a layout
misalignment; the report localizes it per document and tests whether the
offset accumulates toward the tail (the sawtooth signature).

The cuDNN fused indexer's internal split (third leg) is not introspectable
from user space; if legs 1==2 but corruption persists, the cuDNN internal
split is the residual suspect (needs the cuDNN source or a kernel-level dump).

SCOPE NOTE: this tests the DETERMINISTIC layout convention. The separately
observed run-to-run nondeterminism of the padded path (4 samples, range 0.030)
is the uninit-memory class and is NOT what this probe measures.

GPU note: leg 2 allocates a few-KB CUDA tensors (TE's binding is CUDA-only).
Run it in a boot gap / pre-boot window at the box owner's discretion. Use
--skip-te to run only the mcore/CPU leg (self-consistency only).
"""
import argparse
import sys

import torch

# --- replicate the packer's boundary construction (thd_cp.py:258-330,353-387) ---
def build_cu(doc_lens, pad_multiple, pad_to_length=None):
    cu_real = [0]
    cu_pad = [0]
    for n in doc_lens:
        padded = -(-n // pad_multiple) * pad_multiple
        cu_real.append(cu_real[-1] + n)
        cu_pad.append(cu_pad[-1] + padded)
    if pad_to_length is not None:
        tail = pad_to_length - cu_pad[-1]
        assert tail >= 0, "pad_to_length below padded total"
        cu_pad[-1] += tail
    return cu_real, cu_pad


def doc_of_row(cu_pad, row):
    # which doc global padded row belongs to
    for i in range(len(cu_pad) - 1):
        if cu_pad[i] <= row < cu_pad[i + 1]:
            return i
    return len(cu_pad) - 2


def run_case(name, doc_lens, cp_size, pad_to_length, skip_te):
    print(f"\n=== case {name}: {len(doc_lens)} docs, cp={cp_size}, pad_to={pad_to_length} ===")
    pad_multiple = 2 * cp_size  # TP1 GLM rule (thd_cp.py:58-79)
    cu_real, cu_pad = build_cu(doc_lens, pad_multiple, pad_to_length)
    total = cu_pad[-1]
    assert total % (2 * cp_size) == 0, f"total {total} not 2cp-divisible"
    local = total // cp_size
    print(f"  real={cu_real[-1]} padded={total} local/rank={local} tailfill={pad_to_length - sum((-(-n // pad_multiple) * pad_multiple) for n in doc_lens) if pad_to_length else 0}")

    from megatron.core.transformer.experimental_attention_variant import dsa_layout

    cu_pad_t = torch.tensor(cu_pad, dtype=torch.int64)
    mismatches_total = 0
    for rank in range(cp_size):
        # mcore position map (CPU path exercises the divisibility guard too)
        pos = dsa_layout.build_packed_allgather_cp_local_positions(
            cu_pad_t, cp_size, rank, torch.device("cpu"), output_size=local
        )
        assert pos.numel() == local, f"mcore positions {pos.numel()} != local {local}"
        # self-consistency: positions must stay within their doc's padded range
        bad_self = 0
        for i, p in enumerate(pos.tolist()):
            d = doc_of_row(cu_pad, p)
            if not (cu_pad[d] <= p < cu_pad[d + 1]):
                bad_self += 1
        if bad_self:
            print(f"  [rank {rank}] mcore SELF-consistency: {bad_self} rows outside their doc range")
        if skip_te:
            continue
        import transformer_engine_torch as tex

        idx = tex.thd_get_partitioned_indices(
            torch.tensor(cu_pad, dtype=torch.int32, device="cuda"),
            total,
            cp_size,
            rank,
        ).cpu()
        assert idx.numel() == local, f"TE indices {idx.numel()} != local {local}"
        neq = (idx != pos).nonzero().flatten().tolist()
        if not neq:
            continue
        mismatches_total += len(neq)
        first = neq[0]
        # per-doc localization + accumulation test
        by_doc = {}
        for i in neq:
            d = doc_of_row(cu_pad, int(idx[i]))
            by_doc.setdefault(d, []).append((i, int(idx[i]), int(pos[i])))
        print(f"  [rank {rank}] MISMATCH: {len(neq)}/{local} rows; first local row {first}: "
              f"TE={int(idx[first])} vs mcore={int(pos[first])}")
        for d, rows in sorted(by_doc.items()):
            deltas = [te - mc for _, te, mc in rows]
            print(f"    doc {d} (padded [{cu_pad[d]},{cu_pad[d+1]})): {len(rows)} rows, "
                  f"TE-mcore delta min {min(deltas)} max {max(deltas)}")
    if skip_te:
        print("  TE leg skipped (--skip-te)")
    elif mismatches_total == 0:
        print("  TE == mcore on every rank: PASS")
    return mismatches_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-te", action="store_true", help="CPU/mcore leg only")
    args = ap.parse_args()

    # Real datum lengths from tonight's parity set (262,032 real tokens, 9 docs)
    real_docs = [31733, 48201, 12997, 63555, 8003, 27111, 40449, 19231, 10752]
    part_a = real_docs[:3]            # 92,931 real -> padded ~93k, tail-fill to 131072
    part_b = real_docs[3:5]           # 63,555 + 8,003
    awkward = [1, 17, 1000, 4097, 33]  # rounding stress

    total_mm = 0
    for cp in (8, 16):
        for name, docs, pad_to in [
            ("realA-tailfill", part_a, 131072),
            ("realA-notailfill", part_a, None),
            ("realB-tailfill", part_b, 131072),
            ("awkward-tailfill", awkward, 131072),
        ]:
            total_mm += run_case(f"{name}@cp{cp}", docs, cp, pad_to, args.skip_te)
    print(f"\nTOTAL MISMATCHED ROWS: {total_mm}")
    sys.exit(1 if total_mm else 0)


if __name__ == "__main__":
    main()
