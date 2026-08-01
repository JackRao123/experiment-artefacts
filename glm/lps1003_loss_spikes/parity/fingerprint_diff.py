#!/usr/bin/env python3
"""Diff two fingerprint.py outputs (prod pod vs devbox rank node).

Reports:
  - driver / vbios / GPU name deltas
  - loaded .so set diff keyed by soname (basename up to .so suffix), with
    sha256 equality for names present on both sides
  - package version deltas
  - env var deltas for a curated allowlist of numerics-relevant vars
"""
import argparse
import json
import os
import re

ENV_KEYS = re.compile(
    r"^(NVTE_|NCCL_|CUDA_|CUBLAS|CUDNN|TORCH_|PYTORCH_|OMP_|TRITON|BT_WARMUP|"
    r"BT_SKIP|LD_LIBRARY_PATH|PYTHONPATH|MEGATRON|GLOO|UCX_)")


def soname(path):
    b = os.path.basename(path)
    # collapse version suffixes: libcudnn.so.9.19.0 -> libcudnn.so
    return re.sub(r"\.so[.\d]*$", ".so", b)


def load(fp):
    return json.load(open(fp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    la, lb = args.labels

    print(f"### hosts: {la}={A.get('hostname')}  {lb}={B.get('hostname')}")
    for k in ("nvidia", "nvidia_smi_version", "uname", "os_release"):
        va, vb = A.get(k, ""), B.get(k, "")
        mark = "SAME" if va == vb else "DIFF"
        print(f"\n[{mark}] {k}:\n  {la}: {va}\n  {lb}: {vb}")

    sa = {soname(e["path"]): e for e in A.get("mapped_so", [])}
    sb = {soname(e["path"]): e for e in B.get("mapped_so", [])}
    names = sorted(set(sa) | set(sb))
    print(f"\n### mapped .so ({la}: {len(sa)}, {lb}: {len(sb)})")
    same = diff = 0
    for n in names:
        ea, eb = sa.get(n), sb.get(n)
        if ea and eb:
            ha, hb = ea.get("sha256"), eb.get("sha256")
            if ha == hb:
                same += 1
                continue
            diff += 1
            print(f"[SHA DIFF] {n}")
            print(f"   {la}: {ea['path']} size={ea.get('size')} sha={str(ha)[:16]}")
            print(f"   {lb}: {eb['path']} size={eb.get('size')} sha={str(hb)[:16]}")
        elif ea:
            print(f"[{la} ONLY] {n}  ({ea['path']})")
        else:
            print(f"[{lb} ONLY] {n}  ({eb['path']})")
    print(f"summary: {same} identical sha256, {diff} differing, "
          f"{len(names)-same-diff} one-sided")

    pa, pb = A.get("packages", {}), B.get("packages", {})
    print("\n### packages")
    for k in sorted(set(pa) | set(pb)):
        va, vb = pa.get(k), pb.get(k)
        print(f"[{'SAME' if va == vb else 'DIFF'}] {k}: {la}={va} {lb}={vb}")

    ea = {e.split("=", 1)[0]: e.split("=", 1)[1] for e in A.get("environ", []) if "=" in e}
    eb = {e.split("=", 1)[0]: e.split("=", 1)[1] for e in B.get("environ", []) if "=" in e}
    print("\n### numerics-relevant env")
    for k in sorted(set(ea) | set(eb)):
        if not ENV_KEYS.match(k):
            continue
        va, vb = ea.get(k), eb.get(k)
        if va != vb:
            print(f"[DIFF] {k}: {la}={va!r} {lb}={vb!r}")


if __name__ == "__main__":
    main()
