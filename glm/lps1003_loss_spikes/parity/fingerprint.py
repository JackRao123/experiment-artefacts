#!/usr/bin/env python3
"""LPS-1003 environment fingerprint. Stdlib only. Run next to a live trainer.

Captures, as JSON on stdout (or --out FILE):
  - trainer PID(s) (pattern dp_worker.main), argv, cwd
  - every unique .so mapped by the trainer process: path, size, sha256
  - the process environment (/proc/PID/environ)
  - nvidia driver/vbios (nvidia-smi), uname, os-release, hostname
  - python version + importlib.metadata versions for key packages when a
    --python interpreter is given (runs a small subprocess)
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys


def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"


def find_pids(pattern="dp_worker.main"):
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\x00", b" ").decode(errors="replace")
            if pattern in cmd and "grep" not in cmd:
                pids.append((int(d), cmd))
        except OSError:
            continue
    return pids


def sha256_file(path, max_bytes=None):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(1 << 22)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError as e:
        return f"ERROR: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, default=0, help="trainer pid (default: autodetect)")
    ap.add_argument("--pattern", default="dp_worker.main")
    ap.add_argument("--python", default="", help="interpreter to query package versions with")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-hash", action="store_true", help="skip sha256 (fast)")
    args = ap.parse_args()

    fp = {"hostname": sh("hostname"), "uname": sh("uname -a"),
          "os_release": sh("cat /etc/os-release | head -3"),
          "nvidia": sh("nvidia-smi --query-gpu=name,driver_version,vbios_version --format=csv,noheader | sort -u"),
          "nvidia_smi_version": sh("nvidia-smi --version 2>/dev/null | head -4"),
          "ldconfig_cuda": sh("ldconfig -p 2>/dev/null | grep -E 'cudnn|cublas|nccl|cudart|cufft' | head -40")}

    pids = find_pids(args.pattern) if not args.pid else [(args.pid, "(given)")]
    fp["trainer_pids"] = [{"pid": p, "cmd": c[:400]} for p, c in pids]

    sos = {}
    for p, _ in pids[:1] if not args.pid else [(args.pid, "")]:
        try:
            with open(f"/proc/{p}/environ", "rb") as fh:
                env = fh.read().split(b"\x00")
            fp["environ"] = sorted(e.decode(errors="replace") for e in env if e)
        except OSError as e:
            fp["environ_error"] = str(e)
        try:
            with open(f"/proc/{p}/maps") as fh:
                for line in fh:
                    m = re.search(r"(/\S+\.so[\w.]*)$", line.strip())
                    if m:
                        sos[m.group(1)] = True
        except OSError as e:
            fp["maps_error"] = str(e)
    so_list = []
    for so in sorted(sos):
        real = os.path.realpath(so)
        ent = {"path": so, "realpath": real if real != so else None,
               "size": os.path.getsize(so) if os.path.exists(so) else None}
        if not args.no_hash:
            ent["sha256"] = sha256_file(so)
        so_list.append(ent)
    fp["mapped_so"] = so_list

    if args.python:
        code = ("import sys,json;import importlib.metadata as md\n"
                "pk=['torch','nvidia-cudnn-frontend','transformer-engine','triton',"
                "'nvidia-cudnn-cu13','nvidia-cudnn-cu12','nvidia-cublas','nvidia-cublas-cu12',"
                "'transformers','nvidia-cutlass-dsl','megatron-core','flash-attn','numpy']\n"
                "out={'python':sys.version}\n"
                "for p in pk:\n"
                "    try: out[p]=md.version(p)\n"
                "    except Exception: pass\n"
                "print(json.dumps(out))")
        try:
            r = subprocess.run([args.python, "-c", code], capture_output=True,
                               text=True, timeout=120)
            fp["packages"] = json.loads(r.stdout.strip() or "{}")
            if r.returncode != 0:
                fp["packages_stderr"] = r.stderr[-500:]
        except Exception as e:
            fp["packages_error"] = str(e)

    out = json.dumps(fp, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
        print(f"wrote {args.out} ({len(out)} bytes)")
    else:
        print(out)


if __name__ == "__main__":
    main()
