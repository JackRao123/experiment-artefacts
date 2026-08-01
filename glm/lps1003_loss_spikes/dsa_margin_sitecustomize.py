"""Env-gated DSA top-k *selection quality* instrumentation (LPS-1003 Phase D).

Deploy as <dir>/sitecustomize.py and prepend <dir> to PYTHONPATH of the trainer
ranks. Complements dsa_capture_sitecustomize.py (which probes the crash-flood
regime); this one measures how *marginal* the top-k selection is and lets
selection sets be diffed across identical repeats.

Modes (combinable, all off unless env set):

  BT_DSA_MARGIN=1
      Per _indexer_top_k_one_chunk call: for every row, boundary margin
      m = s_(k) - s_(k+1) (k-th vs (k+1)-th score, descending). Logs per-call
      aggregates: min/p1/p10 margin, count of rows with margin < eps ladder
      (0, 1e-3, 1e-2, 1e-1), count of exact boundary ties. One extra topk +
      GPU->CPU sync per call — diagnostic runs only.

  BT_DSA_SEL_DIR=/path
      Per call, save a compact per-row fingerprint of the SELECTION SET
      (sum & xor-hash of selected indices) + margin stats to
      $BT_DSA_SEL_DIR/rank<R>_pid<P>.jsonl. Byte-identical repeats then align
      call-by-call; differing fingerprints pinpoint selection flips.

  BT_DSA_FULLDUMP_DIR=/path  [+ BT_DSA_DUMP_CALLS=lo:hi]
      For calls in [lo,hi) (default all — beware size), torch.save per call:
      top-(k+16) indices+scores per row, seq_lens, k. Enables offline
      near-boundary score distribution + exact selection overlap analysis.

Targets megatron.core.transformer.experimental_attention_variant.
dsa_cudnn_kernels._indexer_top_k_one_chunk via a meta-path hook (same
mechanism as dsa_capture_sitecustomize.py; do not deploy both files as the
same sitecustomize.py — merge dirs on PYTHONPATH instead, python only honors
the first sitecustomize found).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time

_TARGET = "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"
_MARGIN = os.environ.get("BT_DSA_MARGIN") == "1"
_SEL_DIR = os.environ.get("BT_DSA_SEL_DIR")
_FULL_DIR = os.environ.get("BT_DSA_FULLDUMP_DIR")

if _MARGIN or _SEL_DIR or _FULL_DIR:

    def _install(mod) -> None:
        import torch

        orig = mod._indexer_top_k_one_chunk
        rank = os.environ.get("RANK", "?")
        state = {"n": 0}
        lo, hi = 0, 1 << 62
        if os.environ.get("BT_DSA_DUMP_CALLS"):
            a, b = os.environ["BT_DSA_DUMP_CALLS"].split(":")
            lo, hi = int(a), int(b)
        sel_fh = None
        if _SEL_DIR:
            os.makedirs(_SEL_DIR, exist_ok=True)
            sel_fh = open(os.path.join(_SEL_DIR, f"rank{rank}_pid{os.getpid()}.jsonl"),
                          "a", buffering=1)

        def wrapper(scores_flat, seq_lens, topk_k, return_topk_scores):
            state["n"] += 1
            n = state["n"]
            out = orig(scores_flat, seq_lens, topk_k, return_topk_scores)
            try:
                with torch.no_grad():
                    k = int(topk_k)
                    nkeys = scores_flat.shape[-1]
                    L = seq_lens.to(scores_flat.device)
                    valid = (torch.arange(nkeys, device=scores_flat.device)[None, :]
                             < L[:, None])
                    # mask invalid keys to -inf so they never enter the boundary
                    masked = torch.where(valid, scores_flat.float(),
                                         torch.full_like(scores_flat, float("-inf"), dtype=torch.float32))
                    kk = min(k + 16, nkeys)
                    top_s, top_i = torch.topk(masked, kk, dim=-1)  # descending
                    # rows where the datum actually has > k valid keys (real selection)
                    full = L > k
                    rec: dict = {"call": n, "k": k, "rows": int(scores_flat.shape[0]),
                                 "rows_full": int(full.sum().item()), "nkeys": nkeys}
                    if full.any() and k < nkeys:
                        m = top_s[full, k - 1] - top_s[full, k]  # boundary margin per row
                        finite = m[torch.isfinite(m)]
                        if finite.numel():
                            rec.update({
                                "margin_min": float(finite.min().item()),
                                "margin_p1": float(torch.quantile(finite, 0.01).item()),
                                "margin_p10": float(torch.quantile(finite, 0.10).item()),
                                "margin_med": float(finite.median().item()),
                                "rows_tie0": int((finite == 0).sum().item()),
                                "rows_lt_1e3": int((finite < 1e-3).sum().item()),
                                "rows_lt_1e2": int((finite < 1e-2).sum().item()),
                                "rows_lt_1e1": int((finite < 1e-1).sum().item()),
                            })
                    if sel_fh is not None:
                        sel = top_i[:, :k].long()
                        # cheap order-free fingerprints of each row's selection set
                        s_sum = sel.sum(dim=-1)
                        s_xor = sel
                        # xor-fold: shift-multiply hash then xor-reduce
                        h = ((s_xor * 0x9E3779B97F4A7C15) & ((1 << 63) - 1))
                        fp = h[:, 0]
                        for c in range(1, h.shape[1]):
                            fp = fp ^ h[:, c]
                        rec["sel_sum_sha"] = hash(tuple(s_sum.tolist())) & ((1 << 62) - 1)
                        rec["sel_fp_sha"] = hash(tuple(fp.tolist())) & ((1 << 62) - 1)
                        rec["ts"] = time.time()
                        sel_fh.write(json.dumps(rec) + "\n")
                    if _MARGIN and (rec.get("rows_lt_1e2", 0) > 0 or n % 50 == 0):
                        print(f"[dsa_margin r{rank}] {json.dumps(rec)}",
                              file=sys.stderr, flush=True)
                    if _FULL_DIR and lo <= n < hi:
                        d = os.path.join(_FULL_DIR, f"rank{rank}_pid{os.getpid()}")
                        os.makedirs(d, exist_ok=True)
                        torch.save({"call": n, "k": k,
                                    "seq_lens": seq_lens.detach().cpu(),
                                    "top_idx": top_i.detach().to("cpu", torch.int32),
                                    "top_scores": top_s.detach().to("cpu", torch.float16)},
                                   os.path.join(d, f"call{n:05d}.pt"))
            except Exception as e:  # noqa: BLE001 — instrumentation must never break training
                print(f"[dsa_margin r{rank}] probe failed call {n}: {e}",
                      file=sys.stderr, flush=True)
            return out

        mod._indexer_top_k_one_chunk = wrapper
        print(f"[dsa_margin r{rank}] installed on {_TARGET} "
              f"(margin={_MARGIN} sel={'on' if _SEL_DIR else 'off'} "
              f"full={'on' if _FULL_DIR else 'off'})",
              file=sys.stderr, flush=True)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
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
                    print(f"[dsa_margin] install failed: {e}", file=sys.stderr, flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
