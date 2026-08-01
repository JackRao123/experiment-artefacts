"""LPS-1003 layer-bisection activation tracer.

Purpose: localize WHERE in the forward pass the destruction enters, instead of
assuming the DSA top-k. Hooks every residual-stream stage of the GLM GPTModel
(embedding, each TransformerLayer output, each layer's self_attention / mlp /
DSA indexer output, final layernorm, output_layer) and records per-token-bin
statistics (rms / absmax / mean / nonfinite-count) per forward call, per rank.

Offline analysis then diffs an EVENT forward against a HEALED forward of the
byte-identical payload; the first hook where the delta exceeds the
healed-vs-healed noise floor is the entry point of the corruption, and every
stage before it is exonerated.

Deploy: put this file's directory on PYTHONPATH of the trainer ranks (devbox:
boot wrapper exports PYTHONPATH; prod: LWS pod-template edit or image override).
Inert unless BT_LTRACE=1.

Env:
  BT_LTRACE=1            enable
  BT_LTRACE_DIR=/path    output dir (required): rank<R>_pid<P>.jsonl.gz +
                         rank<R>_pid<P>_modmap.json
  BT_LTRACE_BIN=2048     token-bin size along the local sequence dim
  BT_LTRACE_MAXOPS=5000  auto-disable after N GPTModel.forward calls (safety)
  BT_LTRACE_NNF_MAXCOLS=32768  skip nonfinite scan for tensors wider than this
  BT_LTRACE_HB=20        stderr heartbeat every N forward calls

Design constraints: no GPU->CPU sync inside layer hooks (stats stay on-GPU,
flushed with ONE .cpu() at GPTModel.forward exit); every hook body is
exception-guarded; self-disables after 20 hook errors. Costs ~10-20ms per
partition forward — validated against the 0.760-0.771 devbox batch-0 gate
before any result is trusted.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time

_ON = os.environ.get("BT_LTRACE") == "1"
_DIR = os.environ.get("BT_LTRACE_DIR")
_TARGET = "megatron.core.models.gpt.gpt_model"

if _ON and _DIR:

    def _install(mod) -> None:
        import torch

        rank = os.environ.get("RANK", "?")
        BIN = int(os.environ.get("BT_LTRACE_BIN", "2048"))
        MAXOPS = int(os.environ.get("BT_LTRACE_MAXOPS", "5000"))
        NNF_MAXCOLS = int(os.environ.get("BT_LTRACE_NNF_MAXCOLS", "32768"))
        HB = int(os.environ.get("BT_LTRACE_HB", "20"))

        def say(msg: str) -> None:
            print(f"[ltrace r{rank}] {msg}", file=sys.stderr, flush=True)

        _PATTERNS = [
            re.compile(r"^embedding$"),
            re.compile(r"^decoder\.layers\.\d+$"),
            re.compile(r"^decoder\.layers\.\d+\.self_attention$"),
            re.compile(r"^decoder\.layers\.\d+\.self_attention\.core_attention$"),
            re.compile(r"^decoder\.layers\.\d+\.self_attention\.core_attention\.indexer$"),
            re.compile(r"^decoder\.layers\.\d+\.self_attention\.core_attention\.indexer\.linear_wk$"),
            re.compile(r"^decoder\.layers\.\d+\.self_attention\.core_attention\.indexer\.linear_wq_b$"),
            re.compile(r"^decoder\.layers\.\d+\.self_attention\.core_attention\.indexer\.linear_weights_proj$"),
            re.compile(r"^decoder\.layers\.\d+\.mlp$"),
            re.compile(r"^decoder\.final_layernorm$"),
            re.compile(r"^output_layer$"),
        ]

        st = {"ops": 0, "hooks": 0, "hook_errors": 0, "disabled": False,
              "records": 0, "skipped_nontensor": 0}

        class _Collector:
            def __init__(self):
                self.buf = []      # list of (hook_id, nb, kind, gpu_tensor[nstats*nb])
                self.active = False

            def rec(self, hook_id: int, kind: str, t) -> None:
                # stats along dim0 token bins; all ops stay on-GPU
                T = t.shape[0]
                x = t.detach().reshape(T, -1)
                ncols = x.shape[1]
                nb = (T + BIN - 1) // BIN
                rows = []
                for b in range(nb):
                    xb = x[b * BIN:(b + 1) * BIN]
                    n = xb.numel()
                    if kind == "int":
                        xf = xb.float()
                        rows.append(torch.stack([
                            xf.amax() if n else torch.zeros((), device=x.device),
                            xf.amin() if n else torch.zeros((), device=x.device),
                            xf.mean() if n else torch.zeros((), device=x.device),
                            (xb < 0).sum().float(),
                        ]))
                    else:
                        rms = torch.linalg.vector_norm(xb.flatten(), dtype=torch.float32) / max(n, 1) ** 0.5
                        amax = xb.abs().amax().float() if n else torch.zeros((), device=x.device)
                        mean = xb.mean(dtype=torch.float32) if n else torch.zeros((), device=x.device)
                        if ncols <= NNF_MAXCOLS and n:
                            nnf = torch.logical_not(torch.isfinite(xb)).sum().float()
                        else:
                            nnf = torch.full((), -1.0, device=x.device)
                        rows.append(torch.stack([rms, amax, mean, nnf]))
                self.buf.append((hook_id, nb, ncols, torch.stack(rows)))  # [nb,4]

        col = _Collector()
        modmap: dict[int, dict] = {}
        handles = []
        jf = None

        def _open_out():
            nonlocal jf
            os.makedirs(_DIR, exist_ok=True)
            jf = gzip.open(os.path.join(_DIR, f"rank{rank}_pid{os.getpid()}.jsonl.gz"),
                           "at")

        def _pick_tensor(out):
            import torch
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, (tuple, list)):
                for o in out:
                    if isinstance(o, torch.Tensor) and o.dim() >= 1 and o.numel():
                        return o
            if isinstance(out, dict):
                for k in ("indices", "output", "hidden_states"):
                    v = out.get(k)
                    if isinstance(v, torch.Tensor):
                        return v
            return None

        def _mk_hook(hook_id: int):
            def hook(module, args, output):
                if not col.active or st["disabled"]:
                    return
                try:
                    t = _pick_tensor(output)
                    if t is None:
                        st["skipped_nontensor"] += 1
                        return
                    kind = "int" if not t.is_floating_point() else "f"
                    if modmap[hook_id].get("kind") != kind:
                        modmap[hook_id]["kind"] = kind
                    col.rec(hook_id, kind, t)
                except Exception as e:  # noqa: BLE001 — tracing must never break training
                    st["hook_errors"] += 1
                    if st["hook_errors"] <= 5:
                        say(f"hook {hook_id} ({modmap[hook_id]['path']}) error: {type(e).__name__}: {e}")
                    if st["hook_errors"] > 20:
                        st["disabled"] = True
                        say("too many hook errors -> tracer disabled")
            return hook

        def _register(model) -> None:
            n = 0
            for path, m in model.named_modules():
                if any(p.match(path) for p in _PATTERNS):
                    hid = len(modmap)
                    modmap[hid] = {"path": path, "cls": type(m).__name__}
                    handles.append(m.register_forward_hook(_mk_hook(hid)))
                    n += 1
            st["hooks"] = n
            _open_out()
            with open(os.path.join(_DIR, f"rank{rank}_pid{os.getpid()}_modmap.json"), "w") as fh:
                json.dump(modmap, fh, indent=1)
            say(f"registered {n} hooks; bin={BIN} maxops={MAXOPS} dir={_DIR}")

        def _find_cu(args, kwargs):
            psp = kwargs.get("packed_seq_params")
            if psp is None:
                for a in list(args) + list(kwargs.values()):
                    if hasattr(a, "cu_seqlens_q"):
                        psp = a
                        break
            if psp is None:
                return None
            cu = getattr(psp, "cu_seqlens_q", None)
            try:
                return cu.tolist() if cu is not None else None
            except Exception:  # noqa: BLE001
                return None

        GPT = mod.GPTModel
        orig_forward = GPT.forward

        def forward(self, *args, **kwargs):
            if st["disabled"] or st["ops"] >= MAXOPS:
                if not st["disabled"]:
                    st["disabled"] = True
                    for h in handles:
                        h.remove()
                    say(f"maxops {MAXOPS} reached -> tracer disabled")
                return orig_forward(self, *args, **kwargs)
            if not getattr(self, "_bt_ltrace_hooked", False):
                try:
                    _register(self)
                except Exception as e:  # noqa: BLE001
                    say(f"hook registration failed: {type(e).__name__}: {e}")
                self._bt_ltrace_hooked = True
            st["ops"] += 1
            n = st["ops"]
            col.buf = []
            col.active = True
            t0 = time.time()
            try:
                out = orig_forward(self, *args, **kwargs)
            finally:
                col.active = False
            try:
                import torch
                cu = _find_cu(args, kwargs)
                if col.buf:
                    flat = torch.cat([b[3].flatten() for b in col.buf]).cpu().tolist()
                    entries = [[b[0], b[1], b[2]] for b in col.buf]
                    rec = {"op": n, "ts": round(time.time(), 3), "wall": round(time.time() - t0, 3),
                           "cu": cu, "entries": entries,
                           "data": [float(f"{v:.6g}") for v in flat]}
                    if jf is not None:
                        jf.write(json.dumps(rec) + "\n")
                        if n % 5 == 0:
                            jf.flush()
                    st["records"] += 1
                if HB and n % HB == 0:
                    say(f"op#{n} hooks={st['hooks']} buf={len(col.buf)} "
                        f"errors={st['hook_errors']} wall={time.time()-t0:.2f}s")
            except Exception as e:  # noqa: BLE001
                st["hook_errors"] += 1
                say(f"flush error op#{n}: {type(e).__name__}: {e}")
            return out

        GPT.forward = forward
        say("GPTModel.forward wrapped (ltrace armed)")

    class _Finder:
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            import importlib.util
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _install(module)
                except Exception as e:  # noqa: BLE001
                    print(f"[ltrace] install failed: {e}", file=sys.stderr, flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
